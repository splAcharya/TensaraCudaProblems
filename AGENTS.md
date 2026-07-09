# AGENTS.md

## Scope

- These are local instructions for this repository.
- This repo is a CUDA workbench for small, self-contained problem files.
- Follow existing file patterns before introducing a new structure.
- Keep edits scoped to the file or documentation the user asked about.
- Do not clean up unrelated files or revert user changes unless asked.

## Collaboration

- Be direct when code, logs, or measurements contradict an assumption.
- When the user asks for brainstorming or debugging, start with observations,
  failing cases, and tradeoffs before changing code.
- When the user asks for an empty stub, keep the body empty. Do not add loops,
  placeholder writes, zero fills, TODO logic, or partial implementations.
- Implement CPU reference logic or GPU kernel logic only when explicitly asked.
- If a request is ambiguous, prefer the smallest reversible scaffolding change.

## CUDA Problem Files

- Keep each problem file self-contained when practical: kernels,
  `extern "C" solution(...)`, host harness, timing, and local tests.
- Keep exported `solution(...)` signatures aligned with the relevant problem
  statement. `solution(...)` receives device pointers and should only launch
  device work.
- Keep CPU reference code separate from GPU kernels.
- Guard optional CPU-backed verification when the CPU reference is absent or
  intentionally empty.
- Preserve existing kernel variants when adding new variants or stubs.
- Add new variant plumbing only when the user asks for it.

## Build And Test

- Compile individual CUDA files with:

  ```text
  nvcc -std=c++17 <file>.cu -o /tmp/<name>_review
  ```

- Prefer `/tmp` output paths for compile checks so tracked binaries are not
  changed accidentally.
- Run `git diff --check` before handing back code changes.
- Run CUDA harnesses only when the local CUDA runtime is available.
- Default test runs should be correctness-oriented.
- Use `--skip-cpu` for heavier benchmark-oriented runs.
- Timing harnesses should use warmup runs and multiple samples. Prefer median
  `kernel_ms` as the default timing.

## Generated Artifacts

- Treat `*_with_cpu.txt`, `*_skip_cpu.txt`, result markdown, and binaries as
  generated or derived artifacts.
- Do not regenerate or overwrite raw result logs unless the user asks.
- When updating summaries, verify the matching raw logs first.
- Do not present skip-CPU timing rows as correctness evidence unless a matching
  CPU-backed run or exact test already verifies the kernel and shape class.

## Style

- Use ASCII unless the touched file already needs non-ASCII.
- Keep source and Markdown lines at or below 79 characters.
- Keep comments short and useful.
- Prefer `rg` for searching.
- Use `apply_patch` for manual file edits.
