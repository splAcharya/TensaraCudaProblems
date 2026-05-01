# `P3_RELU.cu` Results

Updated summary based on the current two-variant ReLU harness:

- CPU-backed correctness run: [p3_with_cpu.txt](/mnt/d/gitrepo/TensaraCudaProblems/p3_with_cpu.txt)
- Heavier benchmark run: [p3_skip_cpu.txt](/mnt/d/gitrepo/TensaraCudaProblems/p3_skip_cpu.txt)

## Kernel Variants

- `basic`: scalar grid-stride ReLU kernel
- `float4`: vectorized `float4` kernel with scalar tail handling

## Harness Context

- GPU: NVIDIA GeForce RTX 3050 Laptop GPU
- Default launch: `block_x=256`, `grid_x=64`
- Scaling sweep:
  - `block_x in [64, 128, 256, 512]`
  - `grid_x in [8, 16, 32, 64, 128]`
- Default runs now verify:
  - small exact tests
  - medium generated tests
  - selected larger odd-shape cases
  - `float4` remainder-2 tail coverage
  - CPU-backed launch sweeps for tail and rectangular shapes
- `--skip-cpu` adds:
  - Tensara-size rows
  - shape variants
  - dedicated odd tail cases
  - scaling sweeps

## Correctness Summary

From [p3_with_cpu.txt](/mnt/d/gitrepo/TensaraCudaProblems/p3_with_cpu.txt):

- Status legend:
  - `cpu=PASS`: CPU output matched a hard-coded expected answer.
  - `cpu=REF`: CPU output was generated and used as the reference for GPU verification.
  - `cpu=SKIP`: CPU reference generation was skipped.
  - `gpu=PASS`: GPU output matched either the hard-coded expected answer or the CPU reference.
  - `gpu=SKIP`: GPU verification was skipped, usually because the run used `--skip-cpu`.
- `basic` passes all current small tests.
- `float4` also passes all current small tests.
- Both kernels pass the medium CPU-reference cases, including odd matrix sizes:
  - `255 x 257`
  - `513 x 1025`
  - `257 x 258`, which covers `total % 4 == 2`
- Both kernels also pass the selected larger odd-shape verification cases:
  - `1023 x 2049`
  - `1537 x 2049`
- The verified launch sweeps pass for both kernels:
  - `scale_tail2: 257 x 258`
  - `scale_rect: 513 x 1025`
- The regenerated CPU-backed log has `8 PASS/PASS`, `94 REF/PASS`, `0 FAIL`, and `0 SKIP` result rows.

That is the main result of the recent harness work: the `float4` path is now checked on nontrivial odd shapes, every scalar tail remainder class, and multiple launch shapes.

## Performance Summary

From [p3_skip_cpu.txt](/mnt/d/gitrepo/TensaraCudaProblems/p3_skip_cpu.txt):

- On most default-launch rows, `float4` is modestly faster than `basic`.
- The new `tail_*` rows also favor `float4`, which is a useful sign that the scalar remainder path is not obviously hurting the kernel.
- The scaling sweeps consistently show a small but repeatable `float4` advantage.

Representative default-launch rows:

- `tensara_1 (4096 x 4096)`:
  - `basic 0.794 ms`
  - `float4 0.757 ms`
- `tensara_2 (6144 x 4096)`:
  - `basic 1.193 ms`
  - `float4 1.128 ms`
- `tensara_3 (4096 x 7168)`:
  - `basic 1.384 ms`
  - `float4 1.314 ms`
- `tensara_4 (4096 x 8192)`:
  - `basic 1.578 ms`
  - `float4 1.502 ms`
- `tensara_5 (8192 x 8192)`:
  - `basic 3.174 ms`
  - `float4 2.990 ms`
- `tail_1 (2049 x 2049)`:
  - `basic 0.204 ms`
  - `float4 0.195 ms`
- `tail_2 (3073 x 4097)`:
  - `basic 0.596 ms`
  - `float4 0.571 ms`
- `tail_3 (4097 x 8193)`:
  - `basic 1.575 ms`
  - `float4 1.514 ms`

## Rerun Note

The current skip-CPU run is clean:

- `tensara_5 (8192 x 8192)` reports `float4 = 2.990 ms`
- `shape_4 (8192 x 8192)` reports `float4 = 3.022 ms`

The earlier `shape_4 / float4` outlier is not present in this rerun. Because `tensara_5` and `shape_4` use the same `8192 x 8192` shape and default launch, the representative `8192 x 8192` `float4` time is still about `3.0 ms`.

The safer reading is:

- `float4` looks correct
- `float4` is slightly faster than `basic` on the verified medium, large, tail, and Tensara-sized rows
- the previous outlier does not reproduce in the retained skip-CPU log

## Best Scaling Results

From the heatmaps in [p3_skip_cpu.txt](/mnt/d/gitrepo/TensaraCudaProblems/p3_skip_cpu.txt):

- `scale_sq / basic`: best `0.784 ms` at `(512, 32)`
- `scale_sq / float4`: best `0.752 ms` at `(64, 128)`
- `scale_rect_1 / basic`: best `1.170 ms` at `(256, 64)`
- `scale_rect_1 / float4`: best `1.121 ms` at `(512, 128)`
- `scale_rect_2 / basic`: best `1.553 ms` at `(256, 64)`
- `scale_rect_2 / float4`: best `1.497 ms` at `(64, 128)`

The broad pattern is:

- weak launches such as low-grid, low-block combinations are clearly underfilled
- once the launch is strong enough, the `float4` kernel tends to shave a small amount off the scalar baseline
- the best region is fairly broad rather than razor-thin

## Notes

- The markdown summary is intentionally shorter than the raw console dumps.
- For complete row-by-row data, use:
  - [p3_with_cpu.txt](/mnt/d/gitrepo/TensaraCudaProblems/p3_with_cpu.txt)
  - [p3_skip_cpu.txt](/mnt/d/gitrepo/TensaraCudaProblems/p3_skip_cpu.txt)
