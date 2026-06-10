# `P1_1D_CONVOLUTIONS.cu` Results

Updated from regenerated local logs:

- CPU-backed correctness run: [p1_with_cpu.txt](p1_with_cpu.txt)
- Benchmark run: [p1_skip_cpu.txt](p1_skip_cpu.txt)

## Harness Context

- GPU: NVIDIA GeForce RTX 3050 Laptop GPU
- Default launch: `block_x=256`, `grid_x=32`
- Timing: median `kernel_ms` from `5` timed samples after `1` warmup
- Scaling sweep:
  - `block_x in [64, 128, 256, 512]`
  - `grid_x in [8, 16, 32, 64, 128]`

## Kernel Variants

- `basic`: direct global-memory implementation
- `basic_c`: direct implementation with `B` in constant memory
- `tiled`: shared-memory tiled implementation with halo loads
- `tiled_c`: tiled implementation with `B` in constant memory
- `bstride`: tiled implementation with block-stride loading
- `bstride_c`: block-stride tiled implementation with `B` in constant memory

## Correctness Summary

From [p1_with_cpu.txt](p1_with_cpu.txt):

- `18 PASS/PASS` rows, `168 REF/PASS` rows
- `0 FAIL` rows
- All kernels pass small exact tests.
- All kernels pass generated medium cases.
- All kernels pass selected large odd-size verification cases.
- All kernels pass large-filter `K=8191` CPU-reference checks.
- The CPU-backed launch sweep passes across all tested block/grid pairs.

From [p1_skip_cpu.txt](p1_skip_cpu.txt):

- `18 SKIP/PASS` rows, `378 SKIP/SKIP` rows
- `0 FAIL` rows
- Small exact rows are still checked.
- Tensara, tile, odd, and scale rows are benchmark-only.

## Performance Summary

From [p1_skip_cpu.txt](p1_skip_cpu.txt):

- Across all comparable skip-CPU configurations, `bstride_c` wins `60`
  of `66`; the remaining `6` are ties.
- Across default-launch non-scale rows, `bstride_c` wins `20` of `26`;
  the remaining `6` are ties.
- Constant memory remains most useful on the large-filter rows.

## Tensara Summary

Published-size benchmark rows:

- `tensara_1`: `N=32768`, `K=8191`
  - default best: `bstride_c = 0.556 ms`
  - scale best: `bstride_c = 0.483 ms` at `(256, 128)`
- `tensara_2`: `N=65536`, `K=8191`
  - default best: `bstride_c = 1.103 ms`
  - scale best: `bstride_c = 0.948 ms` at `(512, 128)`

Default-launch `kernel_ms` by variant:

- `tensara_1`: `basic 1.137`, `basic_c 1.063`, `tiled 1.433`,
  `tiled_c 0.866`, `bstride 1.096`, `bstride_c 0.556`
- `tensara_2`: `basic 2.394`, `basic_c 2.117`, `tiled 2.890`,
  `tiled_c 1.769`, `bstride 2.184`, `bstride_c 1.103`

## Notes

- `--skip-cpu` rows are timing evidence, not correctness evidence.
- Complete row-by-row data lives in the raw `.txt` logs.
