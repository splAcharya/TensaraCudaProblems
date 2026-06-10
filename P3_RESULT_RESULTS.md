# `P3_RELU.cu` Results

Updated from regenerated local logs:

- CPU-backed correctness run: [p3_with_cpu.txt](p3_with_cpu.txt)
- Benchmark run: [p3_skip_cpu.txt](p3_skip_cpu.txt)

## Harness Context

- GPU: NVIDIA GeForce RTX 3050 Laptop GPU
- Default launch: `block_x=256`, `grid_x=64`
- Timing: median `kernel_ms` from `5` timed samples after `1` warmup
- Scaling sweep:
  - `block_x in [64, 128, 256, 512]`
  - `grid_x in [8, 16, 32, 64, 128]`

## Kernel Variants

- `basic`: scalar grid-stride ReLU kernel
- `float4`: vectorized `float4` kernel with scalar tail handling

## Correctness Summary

From [p3_with_cpu.txt](p3_with_cpu.txt):

- `8 PASS/PASS` rows, `94 REF/PASS` rows
- `0 FAIL` rows
- Both kernels pass all small exact tests.
- Both kernels pass medium and larger CPU-reference cases.
- Both kernels pass CPU-backed launch sweeps over tail and rectangular cases.

From [p3_skip_cpu.txt](p3_skip_cpu.txt):

- `8 SKIP/PASS` rows, `358 SKIP/SKIP` rows
- `0 FAIL` rows
- Small exact rows are still checked.
- Tensara, shape, tail, and scale rows are benchmark-only.

## Performance Summary

From [p3_skip_cpu.txt](p3_skip_cpu.txt):

- Across all comparable skip-CPU configurations, `float4` wins `178`
  of `183`; `basic` wins `2`; `3` are ties.
- Across default-launch non-scale rows, `float4` wins `18` of `23`;
  `basic` wins `2`; `3` are ties.
- `float4` is the better default path for the published shapes.

## Tensara Summary

Published-size benchmark rows:

- `tensara_1`: `4096 x 4096`
  - default best: `float4 = 0.752 ms`
  - scale best: `float4 = 0.750 ms` at `(128, 128)`
- `tensara_2`: `6144 x 4096`
  - default best: `float4 = 1.125 ms`
  - scale best: `float4 = 1.119 ms` at `(512, 128)`
- `tensara_3`: `4096 x 7168`
  - default best: `float4 = 1.314 ms`
  - scale best: `float4 = 1.310 ms` at `(64, 128)`
- `tensara_4`: `4096 x 8192`
  - default best: `float4 = 1.502 ms`
  - scale best: `float4 = 1.496 ms` at `(512, 128)`
- `tensara_5`: `8192 x 8192`
  - default best: `float4 = 3.012 ms`
  - scale best: `float4 = 2.982 ms` at `(64, 128)`

Default-launch `kernel_ms` by variant:

- `tensara_1`: `basic 0.794`, `float4 0.752`
- `tensara_2`: `basic 1.189`, `float4 1.125`
- `tensara_3`: `basic 1.385`, `float4 1.314`
- `tensara_4`: `basic 1.578`, `float4 1.502`
- `tensara_5`: `basic 3.182`, `float4 3.012`

## Notes

- `--skip-cpu` rows are timing evidence, not correctness evidence.
- Complete row-by-row data lives in the raw `.txt` logs.
