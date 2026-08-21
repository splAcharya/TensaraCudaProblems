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

## Nsight Compute profiling

`profile_ncu.py` builds a self-contained CUDA problem with `-O3 -lineinfo`,
discovers its `__global__` kernel symbols, prompts for one, and runs Nsight
Compute with `--kernel-name`. When a source file also contains a recognizable
harness dispatch, the script derives the executable's kernel alias and passes
it to `--profile --kernel=...`. It prints the report to the terminal and saves
the same output as a timestamped text file under
`/tmp/ncu_profiles`:

```text
python3 tools/profile_ncu.py P11_L1_NORM.cu
```

For a non-interactive run or a persistent output directory:

```text
python3 tools/profile_ncu.py P11_L1_NORM.cu \
  --kernel=warp_float4 --output-dir ./ncu_profiles
```

By default it skips the five harness warmup launches and profiles one launch.
Use `--launch-skip=0` for a standalone program that does not perform warmups.

The batch profiler scans the source for `__global__` definitions and uses
Nsight Compute's `--kernel-name` filter. It does not depend on
`--list-kernels`. To profile every discovered kernel and create individual
logs plus combined `.txt` and comparison `.md` reports, run:

```text
python3 tools/profile_ncu_all.py <problem>.cu \
  --output-dir ./ncu_profiles
```

The batch command compiles once, profiles each kernel, sorts the Markdown
comparison by profiled duration, and returns a failure status if any
individual profile fails. Use `--extra-arg` for arguments needed by a
standalone CUDA program.

The Markdown report contains an overview plus separate cross-kernel tables for
GPU Speed of Light, Memory Workload Analysis, Launch Statistics, and
Occupancy. NCU durations reported in microseconds are normalized to
milliseconds, and raw-log links are intentionally omitted from the Markdown
summary.
