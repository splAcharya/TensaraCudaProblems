# TensaraCudaProblems

Local CUDA workbench for developing, testing, and benchmarking Tensara-style
GPU kernel solutions before submitting them to the platform.

The repo is intentionally organized around a simple loop:

- implement one or more CUDA kernels for a problem
- expose a Tensara-compatible `extern "C" solution(...)` entry point
- verify correctness against small expected cases and generated CPU references
- benchmark representative input sizes and launch configurations
- summarize the useful findings in a per-problem results file

## Scope

This is not a general CUDA library. Each problem file is a self-contained
local harness for one Tensara problem. The harness code exists to make
iteration fast: it can launch different kernel variants behind the same
exported `solution` routine, run CPU-backed verification, and collect local
timing data.

Current problem files:

- `P1_1D_CONVOLUTIONS.cu`: 1D same-padding convolution / cross-correlation.
- `P3_RELU.cu`: elementwise ReLU over a row-major matrix.
- `P4_MVM.cu`: matrix-vector multiplication over a row-major matrix.
- `P5_LEAKY_RELU.cu`: elementwise Leaky ReLU over a row-major matrix.
- `P6_AVG_POOL_1D.cu`: 1D average pooling over a vector.
- `P7_GELU.cu`: elementwise GELU over a row-major matrix.
- `P8_SIGMOID.cu`: elementwise Sigmoid over a row-major matrix.
- `P9_RMS_NORM.cu`: RMS normalization over a row-major matrix.

Detailed correctness and benchmark notes live next to each problem:

- [Repository-wide results index](RESULTS_INDEX.md)

- [P1_1D_CONVOLUTIONS_RESULTS.md](P1_1D_CONVOLUTIONS_RESULTS.md)
- [P3_RESULT_RESULTS.md](P3_RESULT_RESULTS.md)
- [P4_MVM_RESULTS.md](P4_MVM_RESULTS.md)
- [P5_LEAKY_RELU_RESULTS.md](P5_LEAKY_RELU_RESULTS.md)
- [P6_AVG_POOL_1D_RESULTS.md](P6_AVG_POOL_1D_RESULTS.md)
- [P7_GELU_RESULTS.md](P7_GELU_RESULTS.md)
- [P8_SIGMOID_RESULTS.md](P8_SIGMOID_RESULTS.md)
- [P9_RMS_NORM_RESULTS.md](P9_RMS_NORM_RESULTS.md)

## Harness Pattern

Each problem follows the same broad structure:

- CPU reference implementation for correctness checks.
- One or more CUDA kernel implementations.
- A Tensara-facing `extern "C"` launcher that receives device pointers.
- A local host-side runner that handles allocation, copies, timing, and checks.
- A default correctness-oriented run.
- A heavier `--skip-cpu` benchmark run for larger sizes and launch sweeps.

The exported `solution(...)` function should stay close to what Tensara
expects: it should launch device work using the provided device pointers, not
own the full host allocation or verification flow. Local-only testing belongs
in the harness around it.

## Result Files

Raw run logs are kept as `.txt` files:

- `*_with_cpu.txt`: CPU-backed correctness-oriented runs.
- `*_skip_cpu.txt`: larger benchmark-oriented runs where expensive CPU checks
  are skipped.

The result tables use these status labels:

- `cpu=PASS`: CPU output matched a hard-coded expected answer.
- `cpu=REF`: CPU output was generated and used as the GPU verification
  reference.
- `cpu=SKIP`: CPU reference generation was skipped.
- `gpu=PASS`: GPU output matched the expected output or CPU reference.
- `gpu=SKIP`: GPU verification was skipped.

The markdown result files summarize the raw logs instead of duplicating every
row. They are the place to record which variants are correct, which launch
shapes are promising, and which benchmark rows look noisy or suspicious.

## Current Snapshot

The saved logs cover CPU-backed and skip-CPU runs for P1 and P3 through P9.
P9 is marked stale/incomplete in the results index because its current source
contains a kernel variant absent from the saved logs.

- `P1_1D_CONVOLUTIONS.cu`
  - `bstride_c` is the strongest current heavy-run kernel.
  - It wins 60 of 66 comparable skip-CPU configurations.
  - Best Tensara sweep row: `tensara_1` uses `bstride_c = 0.483 ms`.
- `P3_RELU.cu`
  - `float4` is correct on odd shapes and scalar tail cases.
  - It wins 178 of 183 comparable skip-CPU configurations.
  - Best Tensara sweep row: `4096 x 4096` uses `float4 = 0.750 ms`.
- `P4_MVM.cu`
  - `warp_per_row` is the strongest current matrix-vector kernel.
  - It wins 132 of 180 comparable skip-CPU configurations.
  - Best Tensara sweep row: `4096 x 4096` uses `warp_per_row = 0.362 ms`.
- `P5_LEAKY_RELU.cu`
  - `float4` and `basic` pass the expanded CPU-backed checks.
  - `float4` wins 240 of 245 comparable skip-CPU configurations.
  - Best Tensara sweep row: `4096 x 4096` uses `float4 = 0.751 ms`.
- `P6_AVG_POOL_1D.cu`
  - `basic_ldg` is the strongest current average-pooling kernel overall.
  - It wins 104 of 140 comparable skip-CPU configurations.
  - Best Tensara sweep row: `tensara_1` uses `basic_ldg = 0.062 ms`.
- `P7_GELU.cu`
  - `float4` and `basic` pass the CPU-backed checks.
  - `float4` wins 108 of 108 comparable skip-CPU configurations.
  - Best Tensara sweep row: `4096 x 4096` uses `float4 = 0.752 ms`.
- `P8_SIGMOID.cu`
  - `float4` and `basic` pass the CPU-backed checks.
  - `float4` wins 5 of 5 published Tensara benchmark rows.
  - Best Tensara default row: `4096 x 4096` uses `float4 = 0.757 ms`.

## Local Benchmarking Notes

Local timings are useful for iteration, but they are not a substitute for
Tensara leaderboard measurements. Treat them as directional data:

- every problem supports `--profile --kernel=<variant>`
- profile mode uses `5` warmup launches and `50` kernel-only timed launches
- profile output reports average `avg_kernel_ms` with verification disabled
- every problem supports `--help` and `--list-kernels`

- compare kernel variants under the same harness and input set
- check odd sizes and tail cases, especially for vectorized kernels
- rerun suspicious rows before drawing conclusions
- prefer correctness evidence from CPU-backed runs before trusting
  benchmark-only results

The current local benchmark environment used for the saved result files is an
NVIDIA GeForce RTX 3050 Ti Laptop GPU.

## Development Notes

Repository-wide build, evidence, and reporting commands are documented in
[tools/README.md](tools/README.md). Automation tiers and their evidence limits
are documented in [docs/AUTOMATION.md](docs/AUTOMATION.md).

The repository has been developed with Codex assistance for harness structure,
test generation, benchmark organization, and documentation. Kernel strategy and
implementation details should still be reviewed against the CUDA code and the
raw result logs before submission.
