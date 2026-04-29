# TensaraCudaProblems

Local CUDA workbench for solving and benchmarking Tensara problems before
submitting to the platform. The goal of this repo is to iterate on kernel
correctness, compare alternative implementations, sweep launch configurations,
and document what performs well locally before pushing toward Tensara
leaderboards.

> Note: This repository was developed with assistance from Codex agent
> ChatGPT 5.4. It generated the test suite and authored all repository code
> outside the kernel implementations and their underlying logic.

## Tensara

Tensara describes itself as a platform for GPU programming challenges: write
efficient GPU kernels, benchmark them, and compare against other developers on
standardized hardware. Its homepage centers the loop:

- optimize
- benchmark
- repeat

The site emphasizes:

- real hardware benchmarking
- per-problem leaderboards
- competitive GPU optimization workflows
- community discussion and iteration

Problem catalog context from the public problems page:

- The public problems page currently shows `84` problems.
- The catalog spans practical GPU tasks such as convolution, pooling,
  reduction, normalization, activation functions, matrix multiplication,
  graphics, cryptography, sorting, and quantization.
- `1D Convolution` appears in the public catalog as an easy convolution task.

What this repo is trying to do:

- build local CUDA solutions for Tensara-style problems
- test correctness against CPU references
- compare multiple kernel strategies for the same problem
- profile launch-shape choices such as block size and grid size
- keep concise notes on which approaches are worth submitting

## Problems

### `P1_1D_CONVOLUTIONS.cu`

Implements 1D same-padding convolution / cross-correlation with three kernels:

- `basic`: direct global-memory implementation
- `tiled`: shared-memory tiled version with halo loads
- `bstride`: shared-memory tiled version with block-stride loading

Short summary:

- GPU: NVIDIA GeForce RTX 3050 Laptop GPU, 4 GB VRAM
- Default launch used by the main benchmark rows: `block_x=256`, `grid_x=32`
- All kernels pass correctness checks on the current small, large, tile, and
  `K=8191` web-style cases.
- `tiled` is correct, but it is usually slower than `basic` or `bstride` for
  larger filters.
- For this problem on the local RTX 3050, `bstride` is the best shared-memory
  design so far.
- `basic` remains very competitive, especially at large `K`.
- Larger blocks (`256` or `512`) with moderate-to-high grid counts perform best
  for the `K=8191` scaling cases.
- Default runs focus on small, medium, and selected large correctness cases with CPU checking.
- Odd-size cases are included to exercise boundary and non-multiple launch behavior.
- `--skip-cpu` enables the heavier large/tile/web/odd/scaling benchmark sweep.
- Full benchmark dump, scaling heatmaps, and best-launch notes:
  [P1_1D_CONVOLUTIONS_RESULTS.md](/mnt/d/gitrepo/TensaraCudaProblems/P1_1D_CONVOLUTIONS_RESULTS.md)

### `P3_RELU.cu`

Local ReLU harness for the Tensara problem:

- Matches the Tensara signature
  `extern "C" void solution(const float* input, float* output, size_t n, size_t m)`
- Treats the input/output as row-major `m x n` matrices and applies
  `C[i][j] = max(0, A[i][j])`
- Includes a CPU reference and a baseline GPU kernel implementation
- Default runs focus on small, medium, and selected large correctness cases with CPU checking
- Odd-shape cases are included to exercise tail paths such as `float4` remainder handling
- `--skip-cpu` enables the heavier Tensara-size/shape/tail/scaling benchmark sweep
