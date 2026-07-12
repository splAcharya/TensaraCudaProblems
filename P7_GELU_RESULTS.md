# `P7_GELU.cu` Results

## At a Glance

- Recommendation: `float4`.
- Correctness: all 18 current CPU-backed rows pass.
- Performance: wins all 108 comparable benchmark configurations.
- Status: current raw logs; benchmark rows remain unverified.

Current skip-CPU evidence has 6 exact checked rows and 210 benchmark-only rows.
`float4` wins all five published Tensara rows.

Benchmark-only rows are timing evidence, not correctness evidence. See the
[repository index](RESULTS_INDEX.md) for cross-problem status.

Updated from regenerated local logs:

- CPU-backed correctness run: [p7_with_cpu.txt](p7_with_cpu.txt)
- Benchmark run: [p7_skip_cpu.txt](p7_skip_cpu.txt)

## Harness Context

- GPU: NVIDIA GeForce RTX 3050 Ti Laptop GPU
- Default launch: `block_x=256`, `grid_x=64`
- Timing: median `kernel_ms` from `5` timed samples after `1` warmup
- Scaling sweep:
  - `block_x in [64, 128, 256, 512]`
  - `grid_x in [8, 16, 32, 64, 128]`

## Kernel Variants

- `basic`: scalar grid-stride GELU kernel
- `float4`: vectorized `float4` GELU kernel with scalar tail handling

## Correctness Summary

From [p7_with_cpu.txt](p7_with_cpu.txt):

- `6 PASS/PASS` rows, `12 REF/PASS` rows
- `0 FAIL` rows
- Both kernels pass all small exact tests.
- Both kernels pass generated medium and large CPU-reference cases.

From [p7_skip_cpu.txt](p7_skip_cpu.txt):

- `6 SKIP/PASS` rows, `210 SKIP/SKIP` rows
- `0 FAIL` rows
- Small exact rows are still checked.
- Tensara and scale rows are benchmark-only.

## Performance Summary

From [p7_skip_cpu.txt](p7_skip_cpu.txt):

- Across all comparable skip-CPU configurations, `float4` wins `108`
  of `108`.
- Across default-launch non-scale rows, `float4` wins `8` of `8`.
- Across launch-sweep rows, `float4` wins `100` of `100`.
- `float4` wins every published Tensara row.

## Tensara Summary

Published-size benchmark rows:

- `tensara_1`: `4096 x 4096`
  - default best: `float4 = 0.761 ms`
  - scale best: `float4 = 0.753 ms` at `(64, 128)`
- `tensara_2`: `6144 x 4096`
  - default best: `float4 = 1.144 ms`
  - scale best: `float4 = 1.121 ms` at `(128, 64)`
- `tensara_3`: `4096 x 7168`
  - default best: `float4 = 1.324 ms`
  - scale best: `float4 = 1.311 ms` at `(512, 16)`
- `tensara_4`: `4096 x 8192`
  - default best: `float4 = 1.511 ms`
  - scale best: `float4 = 1.497 ms` at `(64, 128)`
- `tensara_5`: `8192 x 8192`
  - default best: `float4 = 3.024 ms`
  - scale best: `float4 = 2.979 ms` at `(512, 16)`

Default-launch `kernel_ms` by variant:

- `tensara_1`: `basic 0.815`, `float4 0.761`
- `tensara_2`: `basic 1.219`, `float4 1.144`
- `tensara_3`: `basic 1.421`, `float4 1.324`
- `tensara_4`: `basic 1.619`, `float4 1.511`
- `tensara_5`: `basic 3.260`, `float4 3.024`

Best launch-sweep `kernel_ms` by variant:

- `scale_tensara_1`: `basic 0.793` at `(512, 32)`,
  `float4 0.753` at `(64, 128)`
- `scale_tensara_2`: `basic 1.188` at `(128, 128)`,
  `float4 1.121` at `(128, 64)`
- `scale_tensara_3`: `basic 1.389` at `(512, 32)`,
  `float4 1.311` at `(512, 16)`
- `scale_tensara_4`: `basic 1.580` at `(512, 32)`,
  `float4 1.497` at `(64, 128)`
- `scale_tensara_5`: `basic 3.193` at `(128, 128)`,
  `float4 2.979` at `(512, 16)`

## Notes

- `--skip-cpu` rows are timing evidence, not correctness evidence.
- Complete row-by-row data lives in the raw `.txt` logs.
