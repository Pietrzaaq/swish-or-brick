import unittest

from brickorswish.tools.deps_audit import installed_closure, normalize, requirement_name, version_tuple


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


if __name__ == "__main__":
    unittest.main()
