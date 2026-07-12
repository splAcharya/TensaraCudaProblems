# Results Index

The imported raw logs retain partial provenance. Counts below are derived from
the currently available CPU-backed and skip-CPU logs. See
[the reporting standard](docs/RESULTS_FORMAT.md) for claim semantics.

| Problem | Kernels | Verified rows | Benchmark-only | Status | Report |
|---|---:|---:|---:|---|---|
| P1 Conv | 6/6 | 204 | 378 | CURRENT | [P1](P1_1D_CONVOLUTIONS_RESULTS.md) |
| P3 ReLU | 2/2 | 110 | 358 | CURRENT | [P3](P3_RESULT_RESULTS.md) |
| P4 MVM | 6/6 | 210 | 1056 | CURRENT | [P4](P4_MVM_RESULTS.md) |
| P5 Leaky ReLU | 2/2 | 106 | 484 | CURRENT | [P5](P5_LEAKY_RELU_RESULTS.md) |
| P6 Avg Pool | 3/3 | 306 | 396 | CURRENT | [P6](P6_AVG_POOL_1D_RESULTS.md) |
| P7 GELU | 2/2 | 24 | 210 | CURRENT | [P7](P7_GELU_RESULTS.md) |
| P8 Sigmoid | 2/2 | 20 | 10 | CURRENT | [P8](P8_SIGMOID_RESULTS.md) |
| P9 RMS Norm | 4/4 | 27 | 252 | FAIL | [P9](P9_RMS_NORM_RESULTS.md) |

`Verified rows` means an exact-fixture or CPU-reference row reported a GPU
pass. It does not constitute proof beyond the named cases. `Benchmark-only`
rows provide timing evidence only. P9 has seven current CPU-backed `float4`
failures and no `float4` benchmark rows.
