from pathlib import Path
from setuptools import setup


def read_requirements():
	req_path = Path(__file__).parent / "requirements.txt"
	if not req_path.exists():
		return []

	deps = []
	for line in req_path.read_text(encoding="utf-8").splitlines():
		item = line.strip()
		if not item or item.startswith("#"):
			continue
		deps.append(item)
	return deps


setup(
	install_requires=read_requirements(),
)
