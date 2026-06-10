# `P6_AVG_POOL_1D.cu` Results

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

- Across all comparable skip-CPU configurations, `basic_ldg` wins `104`
  of `140`; `basic` wins `21`; `coop_shared` wins `7`; `8` are ties.
- Across default-launch non-scale rows, `basic_ldg` wins `9` of `20`;
  `coop_shared` wins `5`; `basic` wins `2`; `4` are ties.
- The direct kernels are generally strongest in the sweep. `coop_shared`
  only wins one default Tensara row.

## Tensara Summary

Published-size benchmark rows:

- `tensara_1`: `H=2097152`, `K=7`, `S=4`, `P=3`
  - default best: `basic_ldg = 0.063 ms`
  - scale best: `basic_ldg = 0.062 ms` at `(256, 64)`
- `tensara_2`: `H=4194304`, `K=2`, `S=1`, `P=0`
  - default best: `basic_ldg = 0.249 ms`
  - scale best: `basic = 0.225 ms` at `(512, 128)`
- `tensara_3`: `H=8388608`, `K=3`, `S=2`, `P=1`
  - default best: `basic_ldg = 0.299 ms`
  - scale best: `basic = 0.295 ms` at `(128, 128)`
- `tensara_4`: `H=16777216`, `K=4`, `S=2`, `P=1`
  - default best: `basic_ldg = 0.611 ms`
  - scale best: `basic = 0.601 ms` at `(128, 128)`
- `tensara_5`: `H=33554432`, `K=3`, `S=1`, `P=1`
  - default best: `coop_shared = 2.112 ms`
  - scale best: `basic = 1.794 ms` at `(512, 128)`
- `tensara_6`: `H=67108864`, `K=5`, `S=3`, `P=2`
  - default best: `basic_ldg = 2.107 ms`
  - scale best: `basic = 2.092 ms` at `(512, 32)`

Default-launch `kernel_ms` by variant:

- `tensara_1`: `basic 0.068`, `basic_ldg 0.063`,
  `coop_shared 0.080`
- `tensara_2`: `basic 0.304`, `basic_ldg 0.249`,
  `coop_shared 0.276`
- `tensara_3`: `basic 0.310`, `basic_ldg 0.299`,
  `coop_shared 0.358`
- `tensara_4`: `basic 0.648`, `basic_ldg 0.611`,
  `coop_shared 0.833`
- `tensara_5`: `basic 2.508`, `basic_ldg 2.183`,
  `coop_shared 2.112`
- `tensara_6`: `basic 2.169`, `basic_ldg 2.107`,
  `coop_shared 2.471`

## Notes

- `--skip-cpu` rows are timing evidence, not correctness evidence.
- Complete row-by-row data lives in the raw `.txt` logs.
