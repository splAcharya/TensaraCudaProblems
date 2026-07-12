# `P9_RMS_NORM.cu` Results

## At a Glance

- Recommendation among passing variants: `shared_mem`.
- Correctness: 21 current rows pass; all 7 `float4` CPU-backed rows fail.
- Performance: wins 21 of 26 comparable current benchmark table rows.
- Status: current logs are incomplete for `float4` benchmark coverage.

Current skip-CPU evidence has 6 exact checked rows and 252 benchmark-only rows.
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
- `float4`: currently an empty stub and expected to fail until implemented
- `shared_mem`: one block per row with shared-memory row reduction
- `warp`: one warp per row with warp-shuffle row reduction

## Correctness Summary

From [p9_with_cpu.txt](p9_with_cpu.txt):

- `6 PASS/PASS` rows, `15 REF/PASS` rows, `2 PASS/FAIL` rows, and
  `5 REF/FAIL` rows
- The current `float4` stub fails every CPU-backed case.
- `basic`, `shared_mem`, and `warp` pass small exact, medium, and large cases.

From [p9_skip_cpu.txt](p9_skip_cpu.txt):

- `6 SKIP/PASS` rows, `252 SKIP/SKIP` rows
- `0 FAIL` rows
- Small exact rows are still checked.
- Tensara and scale rows are benchmark-only.
- No skip-CPU rows cover the current `float4` stub.

## Performance Summary

From [p9_skip_cpu.txt](p9_skip_cpu.txt):

- Across comparable table rows, `shared_mem` wins `21` of `26`; `warp` wins
  `4`; `basic` wins `1`.
- Across published Tensara rows, `shared_mem` wins `4` of `4`.
- `warp` stays close to `shared_mem` on the larger rows.
- `basic` is much slower than the row-owned kernels and is only useful as
  a baseline.

## Tensara Summary

Published-size benchmark rows:

- `tensara_1`: `1024 x 1024`
  - default best: `shared_mem = 0.061 ms`
  - scale best: `shared_mem = 0.056 ms` at `(128, 128)`
- `tensara_2`: `1024 x 4096`
  - default best: `shared_mem = 0.251 ms`
  - scale best: `shared_mem = 0.224 ms` at `(512, 32)`
- `tensara_3`: `2048 x 8192`
  - default best: `shared_mem = 1.162 ms`
  - scale best: `shared_mem = 0.964 ms` at `(512, 32)`
- `tensara_4`: `512 x 16384`
  - default best: `shared_mem = 0.591 ms`
  - scale best: `shared_mem = 0.587 ms` at `(128, 128)`

Default-launch `kernel_ms` by variant:

- `tensara_1`: `basic 3.654`, `shared_mem 0.061`, `warp 0.078`
- `tensara_2`: `basic 48.306`, `shared_mem 0.251`, `warp 0.304`
- `tensara_3`: `basic 385.591`, `shared_mem 1.162`, `warp 1.202`
- `tensara_4`: `basic 384.537`, `shared_mem 0.591`, `warp 0.605`

Best launch-sweep `kernel_ms` by variant:

- `scale_tensara_1`: `basic 1.998` at `(512, 128)`,
  `shared_mem 0.056` at `(128, 128)`, `warp 0.077` at `(512, 32)`
- `scale_tensara_2`: `basic 32.153` at `(512, 128)`,
  `shared_mem 0.224` at `(512, 32)`, `warp 0.303` at `(128, 128)`
- `scale_tensara_3`: `basic 261.627` at `(512, 128)`,
  `shared_mem 0.964` at `(512, 32)`, `warp 1.198` at `(256, 64)`
- `scale_tensara_4`: `basic 261.982` at `(512, 128)`,
  `shared_mem 0.587` at `(128, 128)`, `warp 0.608` at `(512, 32)`

## Notes

- `--skip-cpu` rows are timing evidence, not correctness evidence.
- Complete row-by-row data lives in the raw `.txt` logs.
- The row-owned kernels dominate P9. `shared_mem` is the current best
  direction, with `warp` close behind on larger feature counts.
