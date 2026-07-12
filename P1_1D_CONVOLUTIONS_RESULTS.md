# `P1_1D_CONVOLUTIONS.cu` Results

## At a Glance

- Recommendation: `bstride_c`.
- Correctness: all 186 current CPU-backed rows pass.
- Performance: wins 21 of 26 comparable current benchmark table rows.
- Status: current raw logs; benchmark rows remain unverified.

Current skip-CPU evidence has 18 exact checked rows and 378 benchmark-only
rows. `bstride_c` wins both published Tensara rows.

Benchmark-only rows are timing evidence, not correctness evidence. See the
[repository index](RESULTS_INDEX.md) for cross-problem status.

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

- Across comparable table rows, `bstride_c` wins `21` of `26`; `5` are ties.
- Across published Tensara rows, `bstride_c` wins `2` of `2`.
- Constant memory remains most useful on the large-filter rows.

## Tensara Summary

Published-size benchmark rows:

- `tensara_1`: `N=32768`, `K=8191`
  - default best: `bstride_c = 0.617 ms`
  - scale best: `bstride_c = 0.537 ms` at `(256, 128)`
- `tensara_2`: `N=65536`, `K=8191`
  - default best: `bstride_c = 1.185 ms`
  - scale best: `bstride_c = 1.097 ms` at `(256, 128)`

Default-launch `kernel_ms` by variant:

- `tensara_1`: `basic 1.179`, `basic_c 1.168`, `tiled 1.574`,
  `tiled_c 0.962`, `bstride 1.239`, `bstride_c 0.617`
- `tensara_2`: `basic 2.587`, `basic_c 2.398`, `tiled 3.283`,
  `tiled_c 1.938`, `bstride 2.398`, `bstride_c 1.185`

## Notes

- `--skip-cpu` rows are timing evidence, not correctness evidence.
- Complete row-by-row data lives in the raw `.txt` logs.
