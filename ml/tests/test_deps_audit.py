import unittest

from brickorswish.tools.deps_audit import (
    fixed_version_from,
    installed_closure,
    normalize,
    requirement_name,
    version_tuple,
)


class DepsAuditTest(unittest.TestCase):
    def test_requirement_name(self):
        self.assertEqual(requirement_name("transformers<6.0.0,>=5.1.0"), "transformers")
        self.assertEqual(requirement_name("pyDeprecate<0.10,>=0.9"), "pydeprecate")
        self.assertEqual(requirement_name('torch==2.9.*; sys_platform == "linux"'), "torch")
        self.assertIsNone(requirement_name('peft; extra == "lora"'))

    def test_normalize(self):
        self.assertEqual(normalize("Typing_Extensions"), "typing-extensions")
        self.assertEqual(normalize("zope.interface"), "zope-interface")

    def test_version_tuple(self):
        self.assertEqual(version_tuple("2.10.0+cu126"), (2, 10, 0))
        self.assertLess(version_tuple("2.9.1"), (2, 10))

    def test_missing_root_gives_empty_closure(self):
        self.assertEqual(installed_closure(["surely-not-installed-package-xyz"]), {})


def _vuln(*fixed, name="pillow"):
    events = [{"introduced": "0"}] + [{"fixed": f} for f in fixed]
    return {"affected": [{"package": {"name": name, "ecosystem": "PyPI"}, "ranges": [{"type": "ECOSYSTEM", "events": events}]}]}


class FixedVersionTest(unittest.TestCase):
    def test_takes_nearest_fix_per_advisory_then_highest(self):
        vulns = [_vuln("10.3.0", "11.3.1"), _vuln("12.1.0")]
        self.assertEqual(fixed_version_from("pillow", "11.3.0", vulns), "12.1.0")

    def test_ignores_fixes_below_installed_and_prereleases(self):
        self.assertEqual(fixed_version_from("Pillow", "11.3.0", [_vuln("9.0.0", "12.0.0rc1", "12.0.0")]), "12.0.0")

    def test_unfixed_advisory_gives_none(self):
        self.assertIsNone(fixed_version_from("pillow", "11.3.0", [_vuln("11.3.1"), _vuln()]))

    def test_other_package_ranges_ignored(self):
        self.assertIsNone(fixed_version_from("pillow", "11.3.0", [_vuln("99.0", name="other")]))


if __name__ == "__main__":
    unittest.main()
