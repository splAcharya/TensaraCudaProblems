# Automation Tiers

The policy workflow runs without a GPU and checks protected implementations,
tooling tests, and patch whitespace. It makes no CUDA correctness claim.

The manually dispatched CUDA workflow uses explicitly labeled self-hosted
runners:

- `compile`: requires a CUDA toolkit but not a usable GPU.
- `correctness`: requires a compatible NVIDIA GPU and runs CPU-backed tests.
- `sanitizer`: runs `compute-sanitizer` memory diagnostics.

Benchmarks remain manual and nonblocking until dedicated hardware, thermal
controls, compatible toolchain keys, and regression thresholds are qualified.
Sanitizer success is diagnostic evidence, not mathematical correctness.

All problem executables share the following discovery and timing interface:

```text
--help
--list-kernels
--kernel=<variant>
--skip-cpu
--timing=median|best
--timing-repeats=<count>
--profile
```

Profile mode requires an explicit kernel and uses five warmups followed by 50
kernel-only launches. Normal timing remains a median of five samples after one
warmup unless overridden.
