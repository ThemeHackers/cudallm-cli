import tempfile
import unittest
from pathlib import Path

from src.profile_parsers import extract_metrics_from_ncu_csv, parse_ncu_csv


class ProfileParsersTest(unittest.TestCase):
    def test_extract_metrics_from_ncu_csv(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / 'ncu.csv'
            csv_path.write_text(
                'Kernel Name,sm__throughput.avg.pct_of_peak_sustained_elapsed,mem__throughput.avg.pct_of_peak_sustained_elapsed,sm__efficiency\n'
                'kernel_a,20.0,30.0,40.0\n'
                'kernel_b,50.0,60.0,70.0\n',
                encoding='utf-8',
            )

            result = extract_metrics_from_ncu_csv(str(csv_path), ['sm__throughput', 'mem__throughput', 'sm__efficiency'])

        self.assertEqual(result['sm__throughput'], 50.0)
        self.assertEqual(result['mem__throughput'], 60.0)
        self.assertEqual(result['sm__efficiency'], 70.0)

    def test_parse_ncu_csv_hotspot(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / 'ncu.csv'
            csv_path.write_text(
                'Kernel Name,cycles\n'
                'kernel_a,10\n'
                'kernel_b,12\n',
                encoding='utf-8',
            )

            result = parse_ncu_csv(str(csv_path))

        self.assertEqual(result['hotspot_kernel'], 'kernel_b')
        self.assertEqual(result['hotspot_value'], 12.0)


if __name__ == '__main__':
    unittest.main()