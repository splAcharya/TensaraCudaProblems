# `P3_RELU.cu` Results

## At a Glance

- Recommendation: `float4`.
- Correctness: all 102 current CPU-backed rows pass.
- Performance: wins 50 of 56 comparable current benchmark table rows.
- Status: current raw logs; benchmark rows remain unverified.

Current skip-CPU evidence has 8 exact checked rows and 358 benchmark-only
rows. `float4` wins all five published Tensara rows.

Benchmark-only rows are timing evidence, not correctness evidence. See the
[repository index](RESULTS_INDEX.md) for cross-problem status.

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

- Across comparable table rows, `float4` wins `50` of `56`; `basic` wins `2`;
  `4` are ties.
- Across published Tensara rows, `float4` wins `5` of `5`.
- `float4` is the better default path for the published shapes.

## Tensara Summary

Published-size benchmark rows:

- `tensara_1`: `4096 x 4096`
  - default best: `float4 = 0.753 ms`
  - scale best: `float4 = 0.752 ms` at `(64, 128)`
- `tensara_2`: `6144 x 4096`
  - default best: `float4 = 1.123 ms`
  - scale best: `float4 = 1.119 ms` at `(512, 128)`
- `tensara_3`: `4096 x 7168`
  - default best: `float4 = 1.313 ms`
  - scale best: `float4 = 1.310 ms` at `(128, 128)`
- `tensara_4`: `4096 x 8192`
  - default best: `float4 = 1.499 ms`
  - scale best: `float4 = 1.497 ms` at `(512, 128)`
- `tensara_5`: `8192 x 8192`
  - default best: `float4 = 2.989 ms`
  - scale best: `float4 = 2.980 ms` at `(128, 64)`

Default-launch `kernel_ms` by variant:

- `tensara_1`: `basic 0.796`, `float4 0.753`
- `tensara_2`: `basic 1.186`, `float4 1.123`
- `tensara_3`: `basic 1.384`, `float4 1.313`
- `tensara_4`: `basic 1.577`, `float4 1.499`
- `tensara_5`: `basic 3.178`, `float4 2.989`

## Notes

- `--skip-cpu` rows are timing evidence, not correctness evidence.
- Complete row-by-row data lives in the raw `.txt` logs.
