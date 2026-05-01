# AGENTS.md

## Project Context

- This repository is a local CUDA workbench for Tensara-style problems.
- Each problem file is intended to be self-contained: CUDA kernels, a Tensara-compatible `extern "C" solution(...)`, host-side test harness, timing, and optional result summaries.
- Keep the exported `solution(...)` signature aligned with the corresponding Tensara problem statement. The function receives device pointers and should only launch device work.
- Use the existing problem files as harness references when adding new problem files.

## Build And Verification

- Compile individual problem files with `nvcc -std=c++17 <file>.cu -o /tmp/<name>_review` for quick validation.
- Prefer `/tmp` output paths for compile checks so tracked binaries are not updated accidentally. Do not update tracked binaries unless the user asks for a rebuild.
- Run `git diff --check` before handing back changes.
- Only run the CUDA harness when a CUDA runtime is available. If the local runtime is unavailable, compile-check and say runtime execution was not possible.
- Default harness runs should be correctness-oriented. Use `--skip-cpu` for heavier benchmark-oriented runs.

## Generated Artifacts

- Treat `*_with_cpu.txt`, `*_skip_cpu.txt`, markdown result files, and tracked binaries as generated or derived artifacts.
- Do not regenerate or overwrite raw result logs unless the user asks for it or provides new run output to sync.
- When updating markdown summaries, verify the corresponding `.txt` logs first and keep the summary numbers in sync with those logs.

## Coding Conventions

- Keep edits scoped to the requested problem file and directly related documentation.
- Use ASCII unless the touched file already requires non-ASCII.
- Keep comments short and useful. Prefer shape and argument headers above CPU reference functions, GPU kernels, and stubs.
- When generating empty CPU reference or GPU kernel stubs, keep their function bodies empty because the user will implement them. Do not add placeholder logic such as zero-fills or TODO loops. Add header comments above those empty stubs that describe each input, output, matrix/vector shape, and size parameter.
- Keep CPU-backed verification disabled while the CPU reference is an empty stub. Enable it only after the CPU reference is implemented.
- Keep CPU reference implementations separate from GPU kernels and guard optional CPU reference verification with a clear flag when useful.

## Testing Pattern

- Include small exact tests with hard-coded expected outputs.
- Include generated medium and large cases once a CPU reference exists.
- Include the Tensara problem's published test sizes in benchmark-oriented runs.
- Add launch-configuration sweeps when comparing kernel variants or checking launch-sensitive behavior.
