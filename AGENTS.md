# AGENTS.md

## Project Context

- This repository is a local CUDA workbench for Tensara-style problems.
- Each problem file should be self-contained: CUDA kernels, a
  Tensara-compatible `extern "C" solution(...)`, host-side harness, timing,
  and optional result summaries.
- The user owns CPU reference and GPU kernel logic. Unless explicitly asked to
  implement logic, Codex should set up files, harnesses, signatures, empty
  stubs, comments, test scaffolding, and variant plumbing only.
- Keep the exported `solution(...)` signature aligned with the corresponding
  Tensara problem statement. The function receives device pointers and should
  only launch device work.
- Use the existing problem files as harness references when adding new problem
  files.

## New Problem Scaffolding

- Fetch the Tensara problem page before creating a new problem file. Use the
  page data, starter code, or embedded problem definition to confirm the exact
  `solution(...)` signature and argument order.
- Add a top-of-file block comment with the problem title, source URL,
  operation description, input/output shape rules, important notes, and
  published test sizes.
- Render formulas as readable multi-line ASCII inside the block comment so
  they survive plain text display.
- Add one empty CPU reference stub and one empty basic GPU kernel stub unless
  the user explicitly asks for implementation logic. Keep stub bodies empty;
  do not add placeholder loops, zero-fills, or TODO work.
- Add short shape and argument headers above CPU reference functions, GPU
  kernels, and stubs.
- Keep CPU-backed verification disabled while the CPU reference is an empty
  stub. Enable it only after the CPU reference is implemented.
- Include small exact tests with hard-coded expected outputs.
- Include generated medium and large cases once a CPU reference exists.
- Include the Tensara problem's published test sizes in benchmark-oriented
  runs.

## Build And Verification

- Compile individual problem files with
  `nvcc -std=c++17 <file>.cu -o /tmp/<name>_review` for quick validation.
- Prefer `/tmp` output paths for compile checks so tracked binaries are not
  updated accidentally. Do not update tracked binaries unless the user asks for
  a rebuild.
- Run `git diff --check` before handing back changes.
- Only run the CUDA harness when a CUDA runtime is available. If the local
  runtime is unavailable, compile-check and say runtime execution was not
  possible.
- Default harness runs should be correctness-oriented. Use `--skip-cpu` for
  heavier benchmark-oriented runs.
- Timing harnesses should use warmup runs and multiple samples. Prefer median
  `kernel_ms` as the default reported timing, following the P4/P5 pattern.
- A `--timing=best` or `--timing=min` option is useful as an explicit opt-in,
  but do not use best-only timing as the default result.

## Generated Artifacts

- Treat `*_with_cpu.txt`, `*_skip_cpu.txt`, markdown result files, and tracked
  binaries as generated or derived artifacts.
- Do not regenerate or overwrite raw result logs unless the user asks for it or
  provides new run output to sync.
- When updating markdown summaries, verify the corresponding `.txt` logs first
  and keep the summary numbers in sync with those logs.
- Summary markdown should distinguish correctness-backed rows from skip-CPU
  benchmark rows. Do not present skip-CPU timing rows as correctness evidence
  unless a matching CPU-backed run or exact test already verified the kernel
  and shape class.
- Include a separate Tensara section in summaries that lists published shapes,
  best kernel variant, and best block/grid launch shape when launch sweeps are
  available.

## Collaboration

- Do not be merely agreeable. If the user's statement or assumption appears
  incorrect, push back with concrete facts, code references, logs, or small
  numeric examples.
- When the user asks for optimization ideas or possible improvements, do not
  jump straight to a final answer or code. Help brainstorm by asking guiding
  questions, identifying tradeoffs, and reasoning from measurements, memory
  access patterns, occupancy, launch shape, and correctness constraints.
- When the user is debugging their own kernel logic, start with observations,
  failing cases, and questions rather than direct fixes. If their attempted
  fixes keep failing multiple times, give only subtle, targeted hints about the
  next invariant to inspect. Implement kernel logic only when explicitly asked.

## Coding Conventions

- Keep edits scoped to the requested problem file and directly related
  documentation.
- Use ASCII unless the touched file already requires non-ASCII.
- Keep source files and Markdown files at or below 79 characters per line.
- Keep comments short and useful.
- Keep CPU reference implementations separate from GPU kernels and guard
  optional CPU reference verification with a clear flag when useful.

## Testing Pattern

- Include diverse input shapes, not only the published Tensara sizes. Cover
  square, rectangular, tall, wide, odd, and tail-sensitive shapes when the
  problem dimensions make those cases meaningful.
- Add launch-configuration sweeps when comparing kernel variants or checking
  launch-sensitive behavior. Use the P3/P4/P5 default sweep:
  `block_x={64,128,256,512}` and `grid_x={8,16,32,64,128}`.
- Include both correctness-oriented sweeps on smaller generated cases and
  heavier `--skip-cpu` sweeps on representative Tensara or large shapes. Keep
  each sweep row in the raw `.txt` logs.
- When printing scale sweep results, include enough data to identify the best
  block/grid pair per kernel and shape. Heatmap-style summaries are preferred
  when the result table would be hard to scan.
