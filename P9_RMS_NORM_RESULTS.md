# `P9_RMS_NORM.cu` Results

## At a Glance

- Recommendation among passing variants: `shared_mem`.
- Correctness: all 28 current CPU-backed rows pass.
- Performance: `shared_mem` wins all four default Tensara rows and all four
  launch-sweep comparisons.
- Status: current logs include completed `float4` correctness and benchmark
  coverage.

Current skip-CPU evidence has 8 exact checked rows and 336 benchmark-only rows.
`shared_mem` wins all four published Tensara rows. Benchmark rows are timing
evidence only. See the [repository index](RESULTS_INDEX.md).

Updated from regenerated local logs:

- CPU-backed correctness run: [p9_with_cpu.txt](p9_with_cpu.txt)
- Benchmark run: [p9_skip_cpu.txt](p9_skip_cpu.txt)

## Harness Context

- GPU: NVIDIA GeForce RTX 3050 Laptop GPU
- Default launch: `block_x=256`, `grid_x=64`
- Timing: median `kernel_ms` from `5` timed samples after `1` warmup
- Scaling sweep:
  - `block_x in [64, 128, 256, 512]`
  - `grid_x in [8, 16, 32, 64, 128]`

## Kernel Variants

- `basic`: flattened grid-stride kernel that recomputes the row RMS for
  each output element
- `float4`: aligned vector loads with scalar prefix and tail handling; each
  output thread still recomputes its row RMS
- `shared_mem`: one block per row with shared-memory row reduction
- `warp`: one warp per row with warp-shuffle row reduction

## Correctness Summary

From [p9_with_cpu.txt](p9_with_cpu.txt):

- `8 PASS/PASS` rows and `20 REF/PASS` rows
- `basic`, `float4`, `shared_mem`, and `warp` pass small exact, medium, and
  large cases.

From [p9_skip_cpu.txt](p9_skip_cpu.txt):

- `8 SKIP/PASS` rows, `336 SKIP/SKIP` rows
- `0 FAIL` rows
- Small exact rows are still checked.
- Tensara and scale rows are benchmark-only.
- Skip-CPU rows include the `float4` benchmark variant.

## Performance Summary

From [p9_skip_cpu.txt](p9_skip_cpu.txt):

- Across the four default Tensara rows, `shared_mem` wins `4` of `4`.
- Across the four launch sweeps, `shared_mem` wins `4` of `4`.
- `float4` is faster than `basic` in the published-size benchmarks, but both
  remain slower than the row-owned kernels.
- `warp` stays close to `shared_mem` on the larger rows.
- `basic` is much slower than the row-owned kernels and is only useful as
  a baseline.

## Tensara Summary

Published-size benchmark rows:

- `tensara_1`: `1024 x 1024`
  - default best: `shared_mem = 0.062 ms`
  - scale best: `shared_mem = 0.056 ms` at `(128, 128)`
- `tensara_2`: `1024 x 4096`
  - default best: `shared_mem = 0.251 ms`
  - scale best: `shared_mem = 0.225 ms` at `(512, 32)`
- `tensara_3`: `2048 x 8192`
  - default best: `shared_mem = 1.164 ms`
  - scale best: `shared_mem = 0.972 ms` at `(512, 32)`
- `tensara_4`: `512 x 16384`
  - default best: `shared_mem = 0.590 ms`
  - scale best: `shared_mem = 0.587 ms` at `(128, 128)`

Default-launch `kernel_ms` by variant:

- `tensara_1`: `basic 3.657`, `float4 1.449`, `shared_mem 0.062`,
  `warp 0.077`
- `tensara_2`: `basic 49.180`, `float4 21.862`, `shared_mem 0.251`,
  `warp 0.305`
- `tensara_3`: `basic 385.907`, `float4 172.210`, `shared_mem 1.164`,
  `warp 1.213`
- `tensara_4`: `basic 382.852`, `float4 170.662`, `shared_mem 0.590`,
  `warp 0.604`

Best launch-sweep `kernel_ms` by variant:

- `scale_tensara_1`: `basic 1.993`, `float4 1.250` at `(512, 128)`,
  `shared_mem 0.056` at `(128, 128)`, `warp 0.077` at `(256, 64)`
- `scale_tensara_2`: `basic 32.290`, `float4 19.702` at `(512, 128)`,
  `shared_mem 0.225` at `(512, 32)`, `warp 0.304` at `(256, 64)`
- `scale_tensara_3`: `basic 263.470`, `float4 152.318` at `(512, 128)`,
  `shared_mem 0.972` at `(512, 32)`, `warp 1.217` at `(512, 128)`
- `scale_tensara_4`: `basic 262.852`, `float4 151.472` at `(512, 128)`,
  `shared_mem 0.587` at `(128, 128)`, `warp 0.604` at `(512, 128)`

## Notes

- `--skip-cpu` rows are timing evidence, not correctness evidence.
- Complete row-by-row data lives in the raw `.txt` logs.
- The row-owned kernels dominate P9. `shared_mem` is the current best
  direction, with `warp` close behind on larger feature counts.
