# `P4_MVM.cu` Results

## At a Glance

- Recommendation: `warp_per_row`.
- Correctness: all 186 current CPU-backed rows pass.
- Performance: wins 49 of 56 comparable current benchmark table rows.
- Status: current raw logs; benchmark rows remain unverified.

Current skip-CPU evidence has 24 exact checked rows and 1,032 benchmark-only
rows. `warp_per_row` wins all five published Tensara rows.

Benchmark-only rows are timing evidence, not correctness evidence. See the
[repository index](RESULTS_INDEX.md) for cross-problem status.

Updated from regenerated local logs:

- CPU-backed correctness run: [p4_with_cpu.txt](p4_with_cpu.txt)
- Benchmark run: [p4_skip_cpu.txt](p4_skip_cpu.txt)

## Harness Context

- GPU: NVIDIA GeForce RTX 3050 Laptop GPU
- Default launch: `block_x=256`, `grid_x=64`
- Timing: median `kernel_ms` from `5` timed samples after `1` warmup
- Scaling sweep:
  - `block_x in [64, 128, 256, 512]`
  - `grid_x in [8, 16, 32, 64, 128]`

## Kernel Variants

- `basic`: direct global-memory implementation
- `constant_b`: direct implementation with `B` in constant memory
- `shared_ab`: shared-memory tiles for `A` and `B`
- `warp`: register accumulation with warp-level reduction
- `warp_const_b`: warp-level mapping with `B` in constant memory
- `warp_per_row`: one warp computes one output row

## Correctness Summary

From [p4_with_cpu.txt](p4_with_cpu.txt):

- `24 PASS/PASS` rows, `162 REF/PASS` rows
- `0 FAIL` rows
- All six kernels pass the small exact tests.
- All six kernels pass generated medium and large CPU-reference cases.
- The CPU-backed launch sweep passes for every tested variant.

From [p4_skip_cpu.txt](p4_skip_cpu.txt):

- `24 SKIP/PASS` rows, `1056 SKIP/SKIP` rows
- `0 FAIL` rows
- Small exact rows are still checked.
- Tensara, shape, and scale rows are benchmark-only.

## Performance Summary

From [p4_skip_cpu.txt](p4_skip_cpu.txt):

- Across comparable table rows, `warp_per_row` wins `49` of `56`; `warp`
  wins `3`; `constant_b` wins `2`; `2` are ties.
- Across published Tensara rows, `warp_per_row` wins `5` of `5`.
- `warp_per_row` is still the strongest overall direction, with `warp`
  close on the published `k=4096` rows.

## Tensara Summary

Published-size benchmark rows:

- `tensara_1`: `4096 x 4096`
  - default best: `warp_per_row = 0.365 ms`
  - scale best: `warp_per_row = 0.364 ms` at `(128, 128)`
- `tensara_2`: `6144 x 4096`
  - default best: `warp_per_row = 0.543 ms`
  - scale best: `warp_per_row = 0.542 ms` at `(256, 64)`
- `tensara_3`: `7168 x 4096`
  - default best: `warp_per_row = 0.634 ms`
  - scale best: `warp_per_row = 0.632 ms` at `(128, 128)`
- `tensara_4`: `8192 x 4096`
  - default best: `warp_per_row = 0.722 ms`
  - scale best: `warp_per_row = 0.722 ms` at `(512, 32)`
- `tensara_5`: `9216 x 4096`
  - default best: `warp_per_row = 0.812 ms`
  - scale best: `warp_per_row = 0.812 ms` at `(512, 32)`

Default-launch `kernel_ms` by variant:

- `tensara_1`: `basic 1.128`, `constant_b 1.090`,
  `shared_ab 2.451`, `warp 0.393`, `warp_const_b 1.912`,
  `warp_per_row 0.365`
- `tensara_2`: `basic 1.533`, `constant_b 1.516`,
  `shared_ab 3.674`, `warp 0.546`, `warp_const_b 2.822`,
  `warp_per_row 0.543`
- `tensara_3`: `basic 1.859`, `constant_b 1.733`,
  `shared_ab 4.292`, `warp 0.636`, `warp_const_b 3.309`,
  `warp_per_row 0.634`
- `tensara_4`: `basic 2.064`, `constant_b 1.960`,
  `shared_ab 4.913`, `warp 0.726`, `warp_const_b 3.789`,
  `warp_per_row 0.722`
- `tensara_5`: `basic 2.269`, `constant_b 2.204`,
  `shared_ab 5.512`, `warp 0.815`, `warp_const_b 4.229`,
  `warp_per_row 0.812`

## Notes

- `--skip-cpu` rows are timing evidence, not correctness evidence.
- Complete row-by-row data lives in the raw `.txt` logs.
