# `P5_LEAKY_RELU.cu` Results

Updated summary based on the current two-variant Leaky ReLU harness with
in-harness median timing.

- CPU-backed correctness run: [p5_with_cpu.txt](p5_with_cpu.txt)
- Heavier benchmark run: [p5_skip_cpu.txt](p5_skip_cpu.txt)

## Kernel Variants

- `basic`: scalar grid-stride Leaky ReLU kernel
- `float4`: vectorized `float4` kernel with scalar tail handling

## Harness Context

- GPU: NVIDIA GeForce RTX 3050 Laptop GPU
- Default launch: `block_x=256`, `grid_x=64`
- Timing: each row reports median `kernel_ms` from `5` timed samples after
  `1` warmup launch.
- Scaling sweep:
  - `block_x in [64, 128, 256, 512]`
  - `grid_x in [8, 16, 32, 64, 128]`
- Default runs verify:
  - small exact tests
  - generated medium cases
  - selected larger cases
  - CPU-backed launch sweeps
- `--skip-cpu` adds:
  - Tensara-size rows
  - shape and tail variants
  - larger launch-configuration sweeps

## Correctness Summary

From [p5_with_cpu.txt](p5_with_cpu.txt):

- `basic` passes all current small, medium, large, and scale rows.
- `float4` passes all current small, medium, large, and scale rows.
- The refreshed CPU-backed log has `6 PASS/PASS`, `94 REF/PASS`,
  `0 FAIL`, and `0 SKIP` result rows.
- The CPU-backed launch sweeps pass for both kernels:
  - `scale_tail: 257 x 258`
  - `scale_rect: 513 x 1025`

From [p5_skip_cpu.txt](p5_skip_cpu.txt):

- The skip-CPU log has `6 SKIP/PASS`, `164 SKIP/SKIP`, and `0 FAIL`
  result rows.
- Small exact rows are still checked in skip-CPU mode.
- Larger Tensara, shape, tail, and scale rows are benchmark-only in this log.

## Performance Summary

From [p5_skip_cpu.txt](p5_skip_cpu.txt):

- `float4` wins all 8 default-launch Tensara rows.
- `float4` wins all 3 best-of-sweep scale rows.
- Across default-launch Tensara, shape, and tail rows, `float4` wins 11 of
  15 comparable rows.
- `basic` wins 4 tail-heavy default-launch rows, where the scalar tail path
  is a larger fraction of total work.

The broad reading is that `float4` is the better default kernel for the
published Tensara shapes. The scalar `basic` kernel is still useful as a
simple baseline and remains competitive on tail-heavy shapes.

## Tensara Tests

From [p5_skip_cpu.txt](p5_skip_cpu.txt), default launch, `kernel_ms`:

```text
name       shape       alpha  basic  float4  winner
tensara_1 4096x4096   0.01   0.798  0.752   float4
tensara_2 4096x4096   0.05   0.800  0.751   float4
tensara_3 4096x4096   0.10   0.795  0.752   float4
tensara_4 4096x4096   0.20   0.798  0.752   float4
tensara_5 6144x4096   0.01   1.194  1.125   float4
tensara_6 6144x4096   0.05   1.194  1.124   float4
tensara_7 6144x4096   0.10   1.197  1.123   float4
tensara_8 6144x4096   0.20   1.194  1.125   float4
```

Best observed launch shapes for matching Tensara-sized scale rows:

```text
shape       scale row     kernel  best block/grid  best ms
4096x4096   scale_sq      basic   128 x 128        0.783
4096x4096   scale_sq      float4  256 x 64         0.750
6144x4096   scale_rect_1  basic   128 x 128        1.171
6144x4096   scale_rect_1  float4  512 x 128        1.121
```

The launch sweep covers one alpha per Tensara shape, so treat the best
block/grid pair as shape-level evidence rather than alpha-specific proof.

## Best Scaling Results

From the heatmaps in [p5_skip_cpu.txt](p5_skip_cpu.txt):

- `scale_sq / basic`: best `0.783 ms` at `(128, 128)`
- `scale_sq / float4`: best `0.750 ms` at `(256, 64)`
- `scale_rect_1 / basic`: best `1.171 ms` at `(128, 128)`
- `scale_rect_1 / float4`: best `1.121 ms` at `(512, 128)`
- `scale_rect_2 / basic`: best `1.565 ms` at `(128, 128)`
- `scale_rect_2 / float4`: best `1.494 ms` at `(512, 128)`

## Notes

- Complete row-by-row data lives in the raw `.txt` logs.
- `float4` is now correct on scalar-tail cases and launch sweeps.
- Local timings are directional; Tensara leaderboard timings may differ.
