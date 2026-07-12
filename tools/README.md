# Workbench tooling

`workbench.py` is read-only by default and writes build/results output under
`/tmp`. It protects owner implementations, compiles all problem files, imports
legacy logs into versioned JSONL evidence, validates evidence, and generates a
repository-level results index.

```text
python3 tools/workbench.py protect
python3 tools/workbench.py build
python3 tools/workbench.py import-legacy
python3 tools/workbench.py validate /tmp/tensara-legacy.jsonl
python3 tools/workbench.py generate-index /tmp/tensara-legacy.jsonl
python3 tools/workbench.py check --evidence /tmp/tensara-legacy.jsonl
python3 tools/workbench.py capture-run --source P8_SIGMOID.cu \
  /tmp/tensara-build/p8 -- --skip-cpu
```

The protected baseline is owner-approved and must never be regenerated as part
of a normal check. Legacy imports retain missing provenance as unknown rather
than inferring it from documentation.

The warning baseline records four protected P6 warnings and three protected P9
warnings. Builds fail if new warnings are introduced without requiring changes
to the owner's implementations.

`capture-run` creates a unique immutable-style run directory containing a
manifest plus captured stdout and stderr. It never selects or overwrites an
approved historical result.
