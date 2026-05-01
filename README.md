# TensaraCudaProblems

Local CUDA workbench for developing, testing, and benchmarking Tensara-style GPU
kernel solutions before submitting them to the platform.

The repo is intentionally organized around a simple loop:

- implement one or more CUDA kernels for a problem
- expose a Tensara-compatible `extern "C" solution(...)` entry point
- verify correctness against small expected cases and generated CPU references
- benchmark representative input sizes and launch configurations
- summarize the useful findings in a per-problem results file

## Scope

This is not a general CUDA library. Each problem file is a self-contained local
harness for one Tensara problem. The harness code exists to make iteration fast:
it can launch different kernel variants behind the same exported `solution`
routine, run CPU-backed verification, and collect local timing data.

Current problem files:

- `P1_1D_CONVOLUTIONS.cu`: 1D same-padding convolution / cross-correlation.
- `P3_RELU.cu`: elementwise ReLU over a row-major matrix.
- `P4_MVM.cu`: matrix-vector multiplication over a row-major matrix.

Detailed correctness and benchmark notes live next to each problem:

- [P1_1D_CONVOLUTIONS_RESULTS.md](/mnt/d/gitrepo/TensaraCudaProblems/P1_1D_CONVOLUTIONS_RESULTS.md)
- [P3_RESULT_RESULTS.md](/mnt/d/gitrepo/TensaraCudaProblems/P3_RESULT_RESULTS.md)

## Harness Pattern

Each problem follows the same broad structure:

- CPU reference implementation for correctness checks.
- One or more CUDA kernel implementations.
- A Tensara-facing `extern "C"` launcher that receives device pointers.
- A local host-side runner that handles allocation, copies, timing, and checks.
- A default correctness-oriented run.
- A heavier `--skip-cpu` benchmark run for larger sizes and launch sweeps.

The exported `solution(...)` function should stay close to what Tensara expects:
it should launch device work using the provided device pointers, not own the full
host allocation or verification flow. Local-only testing belongs in the harness
around it.

## Result Files

Raw run logs are kept as `.txt` files:

- `*_with_cpu.txt`: CPU-backed correctness-oriented runs.
- `*_skip_cpu.txt`: larger benchmark-oriented runs where expensive CPU checks are
  skipped.

The result tables use these status labels:

- `cpu=PASS`: CPU output matched a hard-coded expected answer.
- `cpu=REF`: CPU output was generated and used as the GPU verification reference.
- `cpu=SKIP`: CPU reference generation was skipped.
- `gpu=PASS`: GPU output matched the expected output or CPU reference.
- `gpu=SKIP`: GPU verification was skipped.

The markdown result files summarize the raw logs instead of duplicating every
row. They are the place to record which variants are correct, which launch
shapes are promising, and which benchmark rows look noisy or suspicious.

## Local Benchmarking Notes

Local timings are useful for iteration, but they are not a substitute for
Tensara leaderboard measurements. Treat them as directional data:

- compare kernel variants under the same harness and input set
- check odd sizes and tail cases, especially for vectorized kernels
- rerun suspicious rows before drawing conclusions
- prefer correctness evidence from CPU-backed runs before trusting benchmark-only
  results

The current local benchmark environment used for the saved result files is an
NVIDIA GeForce RTX 3050 Laptop GPU.

## Development Notes

The repository has been developed with Codex assistance for harness structure,
test generation, benchmark organization, and documentation. Kernel strategy and
implementation details should still be reviewed against the CUDA code and the
raw result logs before submission.
