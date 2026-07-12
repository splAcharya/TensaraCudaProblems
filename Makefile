.PHONY: check build import-legacy results-index

check:
	python3 -m unittest tools/test_workbench.py
	python3 tools/workbench.py check

build:
	python3 tools/workbench.py build

import-legacy:
	python3 tools/workbench.py import-legacy

results-index: import-legacy
	python3 tools/workbench.py generate-index /tmp/tensara-legacy.jsonl
