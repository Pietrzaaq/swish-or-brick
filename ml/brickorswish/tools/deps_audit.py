"""Audit installed Python packages against OSV.dev (CVE/GHSA/PYSEC/malicious-package database) and pin them.

    python -m brickorswish.tools.deps_audit rfdetr opencv-python --pins requirements-colab.txt [--fix]

Walks the installed dependency tree of the given root packages, queries OSV for the exact installed
versions, prints the result and optionally writes an `==`-pinned requirements file. Exit code 1 if
anything is vulnerable. --fix upgrades vulnerable packages to their fixed releases first (needs pip).
Standard library only; sends only package names and versions to api.osv.dev.
"""

import argparse
import json
import re
import subprocess
import sys
import urllib.request
from importlib import metadata

OSV_BATCH = "https://api.osv.dev/v1/querybatch"
_NAME = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)")


def normalize(name: str) -> str:
    """PEP 503 name: lowercase, runs of -_. become one '-'."""
    return re.sub(r"[-_.]+", "-", name).lower()


def requirement_name(req: str) -> str | None:
    """Name of a Requires-Dist entry, or None for extras-only requirements (`; extra == "x"`)."""
    if re.search(r"""extra\s*==""", req):
        return None
    m = _NAME.match(req)
    return normalize(m.group(1)) if m else None


def installed_closure(roots: list[str]) -> dict[str, str]:
    """{normalized name: installed version} for the roots and everything they require that is installed.

    Environment markers other than `extra` are not evaluated (that needs `packaging`); a requirement
    that is installed is included, one that is not is skipped. That can only over-include.
    """
    found: dict[str, str] = {}
    todo = [normalize(r) for r in roots]
    while todo:
        name = todo.pop()
        if name in found:
            continue
        try:
            dist = metadata.distribution(name)
        except metadata.PackageNotFoundError:
            continue
        found[name] = dist.version
        for req in dist.requires or []:
            dep = requirement_name(req)
            if dep and dep not in found:
                todo.append(dep)
    return dict(sorted(found.items()))


def osv_vulns(packages: dict[str, str]) -> dict[str, list[str]]:
    """{name: [OSV ids]} for every package with at least one known vulnerability at its version."""
    names = list(packages)
    queries = [{"package": {"name": n, "ecosystem": "PyPI"}, "version": packages[n]} for n in names]
    out: dict[str, list[str]] = {}
    for start in range(0, len(queries), 500):
        body = json.dumps({"queries": queries[start:start + 500]}).encode()
        req = urllib.request.Request(OSV_BATCH, body, {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            results = json.load(resp)["results"]
        for name, res in zip(names[start:start + 500], results):
            ids = [v["id"] for v in res.get("vulns", [])]
            if ids:
                out[name] = ids
    return out


def fixed_version(name: str, version: str) -> str | None:
    """Smallest release that fixes every OSV advisory affecting name==version, or None if some advisory
    has no fixed release yet. For each advisory the lowest stable 'fixed' above the installed version is
    taken (the patch on the nearest line); the answer is the highest of those."""
    body = json.dumps({"package": {"name": name, "ecosystem": "PyPI"}, "version": version}).encode()
    req = urllib.request.Request("https://api.osv.dev/v1/query", body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return fixed_version_from(name, version, json.load(resp).get("vulns", []))


def fixed_version_from(name: str, version: str, vulns: list[dict]) -> str | None:
    """fixed_version() on already fetched OSV vulnerability records."""
    current = version_tuple(version)
    target: str | None = None
    for vuln in vulns:
        fixes = [
            ev["fixed"]
            for aff in vuln.get("affected", [])
            if normalize(aff.get("package", {}).get("name", "")) == normalize(name)
            for rng in aff.get("ranges", [])
            for ev in rng.get("events", [])
            if "fixed" in ev and re.fullmatch(r"[\d.]+", ev["fixed"]) and version_tuple(ev["fixed"]) > current
        ]
        if not fixes:
            return None
        nearest = min(fixes, key=version_tuple)
        if target is None or version_tuple(nearest) > version_tuple(target):
            target = nearest
    return target


def version_tuple(version: str) -> tuple[int, ...]:
    """'2.10.0+cu126' -> (2, 10, 0). Good enough for minimum-version checks."""
    return tuple(int(p) for p in re.findall(r"\d+", version.split("+")[0])[:3])


def write_pins(packages: dict[str, str], path: str, header: str = "") -> None:
    """Local version labels are dropped ('2.10.0+cu128' -> '2.10.0'): they are not on PyPI, and
    `==2.10.0` is still satisfied by an installed 2.10.0+cu128, so nothing gets reinstalled."""
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        if header:
            fh.write("".join(f"# {line}\n" for line in header.splitlines()))
        fh.writelines(f"{n}=={v.split('+')[0]}\n" for n, v in packages.items())


CLEAN, VULNERABLE, NOTHING_INSTALLED, RESTART_NEEDED = 0, 1, 2, 3


def audit(roots: list[str], pins: str | None = None, fix: bool = False) -> int:
    """Prints the OSV result for the installed tree of `roots`; returns one of the codes above.

    fix=True upgrades each vulnerable package to the smallest release that fixes all its advisories
    (with pip, from PyPI), then audits again. RESTART_NEEDED: what is installed is clean, but this Python
    process already imported an older version of some package, so restart it before going on.
    """
    packages = installed_closure(roots)
    if not packages:
        print("none of the root packages are installed", file=sys.stderr)
        return NOTHING_INSTALLED
    vulns = osv_vulns(packages)
    print(f"{len(packages)} packages checked against OSV.dev, {len(vulns)} with known vulnerabilities")
    for name, ids in vulns.items():
        print(f"  {name}=={packages[name]}: {len(ids)} advisories, e.g. https://osv.dev/vulnerability/{ids[0]}")
    if vulns and fix:
        targets = {n: fixed_version(n, packages[n]) for n in vulns}
        for n in [n for n, t in targets.items() if t is None]:
            print(f"  {n}: no fixed release yet, cannot upgrade")
        upgrades = {n: t for n, t in targets.items() if t}
        if upgrades:
            print("upgrading: " + ", ".join(f"{n} {packages[n]} -> {t}" for n, t in upgrades.items()))
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", *(f"{n}>={t}" for n, t in upgrades.items())],
                           check=True)
            return audit(roots, pins, fix=False)
    if pins:
        header = (f"Exact versions installed on Colab for: {' '.join(roots)}\n"
                  f"Generated by brickorswish.tools.deps_audit; {len(vulns)} known vulnerabilities at generation time.")
        write_pins(packages, pins, header)
        print(f"pins written to {pins}")
    if vulns:
        return VULNERABLE
    stale = stale_imports(packages)
    if stale:
        print("loaded in memory with an older version than installed: " + ", ".join(stale))
        return RESTART_NEEDED
    return CLEAN


def stale_imports(packages: dict[str, str]) -> list[str]:
    """Already imported modules whose __version__ differs from the installed version of their package."""
    out = set()
    for mod, owners in metadata.packages_distributions().items():
        loaded = getattr(sys.modules.get(mod), "__version__", None)
        if not isinstance(loaded, str):
            continue
        for owner in owners:
            installed = packages.get(normalize(owner))
            if installed and loaded.split("+")[0] != installed.split("+")[0]:
                out.add(f"{mod} {loaded} (installed {installed})")
    return sorted(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("roots", nargs="+", help="installed root packages, e.g. rfdetr opencv-python")
    ap.add_argument("--pins", help="write an ==-pinned requirements file of the whole tree")
    ap.add_argument("--fix", action="store_true", help="upgrade vulnerable packages to their fixed releases")
    args = ap.parse_args()
    return audit(args.roots, args.pins, args.fix)


if __name__ == "__main__":
    sys.exit(main())
