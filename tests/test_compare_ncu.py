import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "compare_ncu.py"
BASELINE = ROOT / "ci" / "baselines" / "baseline.csv"


class CompareNCUTest(unittest.TestCase):
    def run_compare(self, current_csv: Path, column: str = "cycles") -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, str(SCRIPT), str(BASELINE), str(current_csv), "-c", column],
            capture_output=True,
            text=True,
        )

    def test_identical_csv_passes(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            current_csv = Path(tmp_dir) / "current.csv"
            current_csv.write_text(BASELINE.read_text(encoding="utf-8"), encoding="utf-8")

            result = self.run_compare(current_csv)

        self.assertEqual(result.returncode, 0, msg=result.stdout + result.stderr)

    def test_regression_exits_nonzero(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            current_csv = Path(tmp_dir) / "current.csv"
            current_csv.write_text(
                "Kernel Name,sm__cycles_elapsed.avg,dram__throughput.avg,sm__sass_thread_inst_executed_avg\n"
                "kernel_a,130.0,200.0,300.0\n"
                "kernel_b,150.0,210.0,330.0\n",
                encoding="utf-8",
            )

            result = self.run_compare(current_csv)

        self.assertEqual(result.returncode, 3, msg=result.stdout + result.stderr)
        self.assertIn("Performance regression detected", result.stdout)


if __name__ == "__main__":
    unittest.main()