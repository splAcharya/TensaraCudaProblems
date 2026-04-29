# `P1_1D_CONVOLUTIONS.cu` Results

Updated summary based on the current local harness structure:

- CPU-backed correctness run: [p1_with_cpu.txt](/mnt/d/gitrepo/TensaraCudaProblems/p1_with_cpu.txt)
- Heavier benchmark run: [p1_skip_cpu.txt](/mnt/d/gitrepo/TensaraCudaProblems/p1_skip_cpu.txt)

## Kernel Variants

- `basic`: direct global-memory implementation
- `basic_c`: direct implementation with `B` in constant memory
- `tiled`: shared-memory tiled implementation with halo loads
- `tiled_c`: tiled implementation with `B` in constant memory
- `bstride`: tiled implementation with block-stride loading
- `bstride_c`: block-stride tiled implementation with `B` in constant memory

## Harness Context

- GPU: NVIDIA GeForce RTX 3050 Laptop GPU
- Default launch: `block_x=256`, `grid_x=32`
- Scaling sweep:
  - `block_x in [32, 64, 128, 256, 512]`
  - `grid_x in [8, 16, 32, 64]`
- Default runs now verify:
  - small exact tests
  - medium generated tests
  - selected larger odd-`N` tests
- `--skip-cpu` adds:
  - heavier `large`
  - `tile`
  - `web`
  - `odd`
  - scaling sweeps

## Correctness Summary

From [p1_with_cpu.txt](/mnt/d/gitrepo/TensaraCudaProblems/p1_with_cpu.txt):

- Status legend:
  - `cpu=PASS`: CPU output matched a hard-coded expected answer.
  - `cpu=REF`: CPU output was generated and used as the reference for GPU verification.
  - `cpu=SKIP`: CPU reference generation was skipped.
  - `gpu=PASS`: GPU output matched either the hard-coded expected answer or the CPU reference.
  - `gpu=SKIP`: GPU verification was skipped, usually because the run used `--skip-cpu`.
- All six kernels pass the current small exact tests.
- All six kernels pass the new medium CPU-reference cases, including odd sizes like `N=4097, K=63` and `N=8193, K=95`.
- All six kernels also pass the selected larger odd-size verification cases:
  - `large_1: N=32769, K=127`
  - `large_2: N=65537, K=191`

That gives reasonable confidence that the current kernels handle both ordinary and odd-length signals correctly under the refactored harness.

## Performance Summary

From [p1_skip_cpu.txt](/mnt/d/gitrepo/TensaraCudaProblems/p1_skip_cpu.txt):

- `bstride_c` remains the strongest overall kernel on the heavy runs.
- Constant memory still helps most when the filter is large and heavily reused.
- The odd-size benchmark rows behave consistently with the even-size rows; the new odd cases did not expose a correctness-looking performance collapse.

Representative default-launch rows:

- `web_1 (N=32768, K=8191)`:
  - `basic 1.504 ms`
  - `basic_c 1.408 ms`
  - `tiled 1.901 ms`
  - `tiled_c 1.151 ms`
  - `bstride 1.449 ms`
  - `bstride_c 0.734 ms`
- `web_2 (N=65536, K=8191)`:
  - `basic 3.167 ms`
  - `basic_c 2.806 ms`
  - `tiled 3.832 ms`
  - `tiled_c 2.339 ms`
  - `bstride 2.889 ms`
  - `bstride_c 1.459 ms`
- `odd_3 (N=262147, K=383)`:
  - `basic 0.601 ms`
  - `basic_c 0.547 ms`
  - `tiled 0.744 ms`
  - `tiled_c 0.479 ms`
  - `bstride 0.571 ms`
  - `bstride_c 0.298 ms`
- `tile_5 (N=2097152, K=511)`:
  - `basic 6.277 ms`
  - `basic_c 5.890 ms`
  - `tiled 8.410 ms`
  - `tiled_c 5.030 ms`
  - `bstride 6.001 ms`
  - `bstride_c 3.145 ms`

## What Seems To Matter

- For small and medium filters, the ranking is not perfectly stable, but the stronger shared-memory variants are usually near the front.
- For larger filters, moving `B` to constant memory is consistently useful.
- `bstride_c` is the most reliable top-tier kernel across the heavier cases.
- `tiled` without constant memory is usually the weakest heavy-case option.
- The new odd-size cases follow the same broad pattern as the even-size cases, which is a useful sanity check for boundary handling.

## Best Scaling Results

From the heatmaps in [p1_skip_cpu.txt](/mnt/d/gitrepo/TensaraCudaProblems/p1_skip_cpu.txt):

- `web_1 / basic`: best `1.068 ms` at `(512, 32)`
- `web_1 / basic_c`: best `0.936 ms` at `(512, 64)`
- `web_1 / tiled`: best `1.277 ms` at `(512, 64)`
- `web_1 / tiled_c`: best `0.751 ms` at `(512, 64)`
- `web_1 / bstride`: best `1.070 ms` at `(512, 64)`
- `web_1 / bstride_c`: best `0.550 ms` at `(256, 32)`
- `web_2 / basic`: best `2.158 ms` at `(512, 64)`
- `web_2 / basic_c`: best `1.984 ms` at `(512, 64)`
- `web_2 / tiled`: best `2.546 ms` at `(512, 64)`
- `web_2 / tiled_c`: best `1.459 ms` at `(512, 64)`
- `web_2 / bstride`: best `2.152 ms` at `(256, 64)`
- `web_2 / bstride_c`: best `1.081 ms` at `(512, 64)`

The useful high-level takeaway is unchanged: larger blocks with moderate-to-high grid counts work well, and `bstride_c` is still the best tuned path in this local environment.

## Notes

- The markdown summary is intentionally shorter than the raw console dumps.
- For complete row-by-row data, use:
  - [p1_with_cpu.txt](/mnt/d/gitrepo/TensaraCudaProblems/p1_with_cpu.txt)
  - [p1_skip_cpu.txt](/mnt/d/gitrepo/TensaraCudaProblems/p1_skip_cpu.txt)
