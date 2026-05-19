Place baseline `ncu` CSV exports here. Name them clearly, e.g. `baseline_kernel_v1.csv`.

`baseline.csv` is a tiny synthetic fixture used by the unit test harness for `tools/compare_ncu.py`.

CI workflows and `ci/ncu_regression_check.sh` expect a baseline CSV to compare against current runs. Keep small, focused baseline captures for CI to keep runtime reasonable.
