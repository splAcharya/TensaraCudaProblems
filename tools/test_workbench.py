import importlib.util
from pathlib import Path
import tempfile
import unittest


SPEC = importlib.util.spec_from_file_location(
    "workbench", Path(__file__).with_name("workbench.py"))
workbench = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(workbench)


class WorkbenchTests(unittest.TestCase):
    def test_extractor_ignores_braces_in_comments_and_strings(self):
        source = '''
static void owned(int x) {
  // }
  const char* value = "{";
  if (x) { x += 1; }
}
'''
        body = workbench.extract_function(source, "owned")
        self.assertIn("x += 1", body)
        self.assertTrue(body.rstrip().endswith("}"))

    def test_extractor_rejects_declaration_without_definition(self):
        with self.assertRaises(ValueError):
            workbench.extract_function("void owned(int);", "owned")

    def test_protected_baseline_matches_approved_sources(self):
        expected = workbench.load_json(workbench.BASELINE)
        self.assertEqual(expected, workbench.make_baseline())

    def test_legacy_import_is_deterministic(self):
        manifest = workbench.load_json(workbench.MANIFEST)
        problem = manifest["problems"][0]
        path = workbench.ROOT / "p1_with_cpu.txt"
        first = workbench.import_log(path, problem)
        second = workbench.import_log(path, problem)
        self.assertEqual(first, second)
        self.assertTrue(first)
        self.assertTrue(all(x["provenance_quality"] == "legacy_partial"
                            for x in first))


if __name__ == "__main__":
    unittest.main()
