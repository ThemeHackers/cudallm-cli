import importlib.util
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VERIFY_SCRIPT = ROOT / 'scripts' / 'verify.py'


def load_verify_module():
    spec = importlib.util.spec_from_file_location('verify_harness', VERIFY_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


class VerifyHarnessTest(unittest.TestCase):
    def test_candidate_libs_prefers_explicit_path(self):
        module = load_verify_module()
        with tempfile.TemporaryDirectory() as tmp_dir:
            explicit = str(Path(tmp_dir) / 'module.dll')
            candidates = list(module._candidate_libs(explicit))

        self.assertEqual(candidates[0], explicit)
        self.assertIn('module.dll', candidates[1])
        self.assertIn('module.so', candidates[2])

    def test_candidate_libs_without_explicit_path(self):
        module = load_verify_module()
        candidates = list(module._candidate_libs())

        self.assertEqual(len(candidates), 2)
        self.assertTrue(candidates[0].endswith('module.dll'))
        self.assertTrue(candidates[1].endswith('module.so'))


if __name__ == '__main__':
    unittest.main()