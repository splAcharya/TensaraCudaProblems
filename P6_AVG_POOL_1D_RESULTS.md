# `P6_AVG_POOL_1D.cu` Results

## At a Glance

- Recommendation: `basic_ldg`.
- Correctness: all 282 current CPU-backed rows pass.
- Performance: wins 25 of 50 comparable current benchmark table rows.
- Status: current raw logs; benchmark rows remain unverified.

Current skip-CPU evidence has 24 exact checked rows and 396 benchmark-only
rows. Published Tensara rows split: `basic_ldg` wins 3, `basic` 2, and
`coop_shared` 1.

Benchmark-only rows are timing evidence, not correctness evidence. See the
[repository index](RESULTS_INDEX.md) for cross-problem status.

Updated from regenerated local logs:

- CPU-backed correctness run: [p6_with_cpu.txt](p6_with_cpu.txt)
- Benchmark run: [p6_skip_cpu.txt](p6_skip_cpu.txt)

## Harness Context

- GPU: NVIDIA GeForce RTX 3050 Laptop GPU
- Default launch: `block_x=256`, `grid_x=64`
- Timing: median `kernel_ms` from `5` timed samples after `1` warmup
- Scaling sweep:
  - `block_x in [64, 128, 256, 512]`
  - `grid_x in [8, 16, 32, 64, 128]`

## Kernel Variants

- `basic`: direct average-pooling kernel
- `basic_ldg`: direct kernel using read-only loads
- `coop_shared`: cooperative shared-memory input staging

## Correctness Summary

From [p6_with_cpu.txt](p6_with_cpu.txt):

- `24 PASS/PASS` rows, `258 REF/PASS` rows
- `0 FAIL` rows
- All kernels pass small exact tests.
- All kernels pass generated medium and large CPU-reference cases.
- All kernels pass CPU-backed launch sweeps.

From [p6_skip_cpu.txt](p6_skip_cpu.txt):

- `24 SKIP/PASS` rows, `396 SKIP/SKIP` rows
- `0 FAIL` rows
- Small exact rows are still checked.
- Tensara and scale rows are benchmark-only.

## Performance Summary

From [p6_skip_cpu.txt](p6_skip_cpu.txt):

- Across comparable table rows, `basic_ldg` wins `25` of `50`; `basic` wins
  `14`; `coop_shared` wins `2`; `9` are ties.
- Across published Tensara rows, `basic_ldg` wins `3`; `basic` wins `2`; and
  `coop_shared` wins `1`.
- The direct kernels are generally strongest in the sweep. `coop_shared`
  only wins one default Tensara row.

## Tensara Summary

Published-size benchmark rows:

- `tensara_1`: `H=2097152`, `K=7`, `S=4`, `P=3`
  - default best: `basic_ldg = 0.065 ms`
  - scale best: `basic_ldg = 0.062 ms` at `(128, 128)`
- `tensara_2`: `H=4194304`, `K=2`, `S=1`, `P=0`
  - default best: `basic = 0.259 ms`
  - scale best: `basic_ldg = 0.224 ms` at `(512, 128)`
- `tensara_3`: `H=8388608`, `K=3`, `S=2`, `P=1`
  - default best: `basic = 0.302 ms`
  - scale best: `basic = 0.296 ms` at `(512, 32)`
- `tensara_4`: `H=16777216`, `K=4`, `S=2`, `P=1`
  - default best: `basic_ldg = 0.608 ms`
  - scale best: `basic_ldg = 0.602 ms` at `(128, 128)`
- `tensara_5`: `H=33554432`, `K=3`, `S=1`, `P=1`
  - default best: `coop_shared = 2.120 ms`
  - scale best: `basic_ldg = 1.802 ms` at `(512, 128)`
- `tensara_6`: `H=67108864`, `K=5`, `S=3`, `P=2`
  - default best: `basic_ldg = 2.111 ms`
  - scale best: `basic = 2.100 ms` at `(512, 32)`

Default-launch `kernel_ms` by variant:

- `tensara_1`: `basic 0.066`, `basic_ldg 0.065`,
  `coop_shared 0.085`
- `tensara_2`: `basic 0.259`, `basic_ldg 0.262`,
  `coop_shared 0.271`
- `tensara_3`: `basic 0.302`, `basic_ldg 0.305`,
  `coop_shared 0.362`
- `tensara_4`: `basic 0.613`, `basic_ldg 0.608`,
  `coop_shared 0.828`
- `tensara_5`: `basic 2.193`, `basic_ldg 2.187`,
  `coop_shared 2.120`
- `tensara_6`: `basic 2.167`, `basic_ldg 2.111`,
  `coop_shared 2.468`

## Notes

- `--skip-cpu` rows are timing evidence, not correctness evidence.
- Complete row-by-row data lives in the raw `.txt` logs.
