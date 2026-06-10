# `P4_MVM.cu` Results

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

- Across all comparable skip-CPU configurations, `warp_per_row` wins `132`
  of `180`; `constant_b` wins `23`; `warp` wins `13`; `basic` wins `1`;
  `11` are ties.
- Across default-launch non-scale rows, `warp_per_row` wins `8` of `20`;
  `warp` wins `6`; `6` are ties.
- `warp_per_row` is still the strongest overall direction, with `warp`
  close on the published `k=4096` rows.

## Tensara Summary

Published-size benchmark rows:

- `tensara_1`: `4096 x 4096`
  - default best: `warp_per_row = 0.362 ms`
  - scale best: `warp_per_row = 0.362 ms` at `(128, 128)`
- `tensara_2`: `6144 x 4096`
  - default best: `warp_per_row = 0.541 ms`
  - scale best: `warp_per_row = 0.542 ms` at `(128, 128)`
- `tensara_3`: `7168 x 4096`
  - default best: `warp_per_row = 0.634 ms`
  - scale best: `warp_per_row = 0.632 ms` at `(256, 64)`
- `tensara_4`: `8192 x 4096`
  - default best: `warp_per_row = 0.722 ms`
  - scale best: `warp_per_row = 0.721 ms` at `(128, 128)`
- `tensara_5`: `9216 x 4096`
  - default best: `warp_per_row = 0.811 ms`
  - scale best: `warp_per_row = 0.810 ms` at `(128, 128)`

Default-launch `kernel_ms` by variant:

- `tensara_1`: `basic 1.482`, `constant_b 1.089`,
  `shared_ab 2.473`, `warp 0.366`, `warp_const_b 1.880`,
  `warp_per_row 0.362`
- `tensara_2`: `basic 1.535`, `constant_b 1.510`,
  `shared_ab 3.680`, `warp 0.547`, `warp_const_b 2.828`,
  `warp_per_row 0.541`
- `tensara_3`: `basic 1.893`, `constant_b 1.730`,
  `shared_ab 4.289`, `warp 0.637`, `warp_const_b 3.280`,
  `warp_per_row 0.634`
- `tensara_4`: `basic 2.088`, `constant_b 1.979`,
  `shared_ab 4.901`, `warp 0.726`, `warp_const_b 3.748`,
  `warp_per_row 0.722`
- `tensara_5`: `basic 2.293`, `constant_b 2.212`,
  `shared_ab 5.516`, `warp 0.815`, `warp_const_b 4.214`,
  `warp_per_row 0.811`

## Notes

- `--skip-cpu` rows are timing evidence, not correctness evidence.
- Complete row-by-row data lives in the raw `.txt` logs.
