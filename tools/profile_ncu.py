#!/usr/bin/env python3
"""Build a CUDA problem and collect an interactive Nsight Compute profile."""

from __future__ import annotations

import argparse
import datetime
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


SECTIONS = (
    "LaunchStats",
    "Occupancy",
    "SpeedOfLight",
    "MemoryWorkloadAnalysis",
)


def command_text(command: list[str]) -> str:
    return shlex.join(command)


def source_kernel_names(source: str) -> list[str]:
    pattern = re.compile(
        r"__global__\s+(?:__launch_bounds__\s*\([^)]*\)\s+)?"
        r"(?:[\w:<>,*&]+\s+)+([A-Za-z_]\w*)\s*\("
    )
    return list(dict.fromkeys(pattern.findall(source)))


def source_kernel_aliases(source: str) -> dict[str, str]:
    alias_matches = re.findall(
        r"case\s+([A-Za-z_]\w*)::(\w+):\s*"
        r"return\s+\"([^\"]+)\";",
        source,
    )
    aliases = {
        (enum, variant): alias
        for enum, variant, alias in alias_matches
    }
    mapping: dict[str, str] = {}
    dispatch_pattern = re.compile(
        r"case\s+([A-Za-z_]\w*)::(\w+):(?P<body>.*?)"
        r"(?=\n\s*case\s+[A-Za-z_]\w*::|\n\s*})",
        re.DOTALL,
    )
    for match in dispatch_pattern.finditer(source):
        variant = (match.group(1), match.group(2))
        kernel_match = re.search(
            r"\b([A-Za-z_]\w*)\s*<<<", match.group("body")
        )
        if variant in aliases and kernel_match is not None:
            mapping[kernel_match.group(1)] = aliases[variant]
    return mapping


def supports_option(binary: Path, option: str) -> bool:
    result = subprocess.run(
        [str(binary), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    return option in result.stdout


def choose_kernel(
    kernels: list[str], aliases: dict[str, str], requested: str | None
) -> str:
    alias_to_kernel = {alias: kernel for kernel, alias in aliases.items()}
    if requested is not None:
        if requested in kernels:
            return requested
        if requested in alias_to_kernel:
            return alias_to_kernel[requested]
        choices = ", ".join(kernels)
        if aliases:
            choices += " (or harness aliases: "
            choices += ", ".join(aliases.values()) + ")"
        raise SystemExit(
            f"Unknown kernel '{requested}'. Choose from: " + choices
        )

    print("Available CUDA kernels:")
    for index, kernel in enumerate(kernels, start=1):
        alias = aliases.get(kernel)
        suffix = f" (harness alias: {alias})" if alias else ""
        print(f"  {index}. {kernel}{suffix}")

    while True:
        answer = input("Select a kernel by number or name: ").strip()
        if answer in kernels:
            return answer
        if answer in alias_to_kernel:
            return alias_to_kernel[answer]
        if answer.isdigit():
            index = int(answer) - 1
            if 0 <= index < len(kernels):
                return kernels[index]
        print("Please enter one of the listed numbers or names.")


def stream_to_terminal_and_file(command: list[str], output_path: Path) -> int:
    print(f"Running: {command_text(command)}")
    print(f"Saving Nsight Compute output to: {output_path}")
    with output_path.open("w", encoding="utf-8") as output:
        output.write(f"Command: {command_text(command)}\n\n")
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            output.write(line)
        return process.wait()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a .cu file and profile one kernel with ncu."
    )
    parser.add_argument("source", type=Path, help="CUDA source file")
    parser.add_argument(
        "--kernel",
        help="Select a kernel without prompting",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/tmp/ncu_profiles"),
        help="Directory for text output (default: /tmp/ncu_profiles)",
    )
    parser.add_argument(
        "--launch-skip",
        type=int,
        default=5,
        help="Kernel launches to skip before profiling (default: 5)",
    )
    parser.add_argument(
        "--launch-count",
        type=int,
        default=1,
        help="Number of launches to profile (default: 1)",
    )
    parser.add_argument(
        "--extra-arg",
        action="append",
        default=[],
        help="Extra argument passed to the CUDA executable; repeatable",
    )
    args = parser.parse_args()

    source = args.source.resolve()
    if not source.is_file() or source.suffix != ".cu":
        parser.error(f"source is not a .cu file: {source}")
    if shutil.which("nvcc") is None:
        raise SystemExit("nvcc was not found in PATH")
    if shutil.which("ncu") is None:
        raise SystemExit("ncu was not found in PATH")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix=f"{source.stem}_ncu_") as build:
        binary = Path(build) / source.stem
        compile_command = [
            "nvcc",
            "-O3",
            "-std=c++17",
            "-lineinfo",
            str(source),
            "-o",
            str(binary),
        ]
        print(f"Building: {command_text(compile_command)}")
        compiled = subprocess.run(compile_command, check=False)
        if compiled.returncode != 0:
            return compiled.returncode

        source_text = source.read_text(encoding="utf-8")
        kernels = source_kernel_names(source_text)
        if not kernels:
            raise SystemExit(
                "Could not find __global__ kernel names in the source file"
            )
        aliases = source_kernel_aliases(source_text)
        profile_mode = supports_option(binary, "--profile")

        kernel = choose_kernel(kernels, aliases, args.kernel)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = args.output_dir / (
            f"{source.stem}_{kernel}_{timestamp}.txt"
        )

        ncu_command = ["ncu"]
        for section in SECTIONS:
            ncu_command.extend(["--section", section])
        ncu_command.extend(
            [
                "--launch-skip",
                str(args.launch_skip),
                "--launch-count",
                str(args.launch_count),
            ]
        )
        ncu_command.extend(["--kernel-name", kernel])
        ncu_command.append(str(binary))
        app_kernel = aliases.get(kernel)
        if app_kernel is not None:
            if profile_mode:
                ncu_command.append("--profile")
            ncu_command.append(f"--kernel={app_kernel}")
        ncu_command.extend(args.extra_arg)

        return stream_to_terminal_and_file(ncu_command, output_path)


if __name__ == "__main__":
    raise SystemExit(main())
