#!/usr/bin/env python3
"""Profile every kernel variant in a CUDA problem with Nsight Compute."""

from __future__ import annotations

import argparse
import datetime
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

try:
    from profile_ncu import (
        SECTIONS,
        command_text,
        source_kernel_aliases,
        source_kernel_names,
        stream_to_terminal_and_file,
        supports_option,
    )
except ModuleNotFoundError:
    from tools.profile_ncu import (
        SECTIONS,
        command_text,
        source_kernel_aliases,
        source_kernel_names,
        stream_to_terminal_and_file,
        supports_option,
    )


def build_command(source: Path, binary: Path) -> list[str]:
    return [
        "nvcc",
        "-O3",
        "-std=c++17",
        "-lineinfo",
        str(source),
        "-o",
        str(binary),
    ]


def ncu_command(
    binary: Path,
    kernel: str,
    profile_mode: bool,
    app_kernel: str | None,
    launch_skip: int,
    launch_count: int,
    extra_args: list[str],
) -> list[str]:
    command = ["ncu"]
    for section in SECTIONS:
        command.extend(["--section", section])
    command.extend(
        [
            "--launch-skip",
            str(launch_skip),
            "--launch-count",
            str(launch_count),
        ]
    )
    command.extend(["--kernel-name", kernel])
    command.append(str(binary))
    if app_kernel is not None:
        if profile_mode:
            command.append("--profile")
        command.append(f"--kernel={app_kernel}")
    command.extend(extra_args)
    return command


def metric_value(
    text: str, section_prefix: str, label: str, unit: str | None = None
) -> str | None:
    section = ""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("Section: "):
            section = stripped.removeprefix("Section: ")
        if not section.startswith(section_prefix):
            continue
        if not stripped.startswith(label):
            continue
        tokens = stripped.split()
        if unit is not None:
            for index, token in enumerate(tokens[:-1]):
                if token == unit:
                    return tokens[index + 1]
            return None
        return tokens[-1] if tokens else None
    return None


METRIC_UNITS = {
    "Ghz",
    "cycle",
    "%",
    "ms",
    "us",
    "ns",
    "Mbyte/s",
    "Kbyte",
    "Kbyte/block",
    "byte/block",
    "block",
    "SM",
    "thread",
    "warp",
    "register/thread",
}


def parse_section_metrics(
    text: str,
) -> dict[str, dict[str, tuple[str, str | None]]]:
    sections: dict[str, dict[str, tuple[str, str | None]]] = {}
    current: str | None = None
    in_metric_table = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("Section: "):
            current = stripped.removeprefix("Section: ")
            sections.setdefault(current, {})
            in_metric_table = False
            continue
        if current is None:
            continue
        if stripped.startswith("Metric Name"):
            in_metric_table = True
            continue
        if stripped.startswith("INF ") or stripped.startswith("OPT "):
            in_metric_table = False
            continue
        if not in_metric_table:
            continue
        if (
            not stripped
            or stripped.startswith("Metric Unit")
            or stripped.startswith("-")
        ):
            continue
        tokens = stripped.split()
        if len(tokens) < 2:
            continue
        unit_index = next(
            (index for index, token in enumerate(tokens[:-1])
             if token in METRIC_UNITS),
            None,
        )
        if unit_index is None:
            label = " ".join(tokens[:-1])
            value = tokens[-1]
            unit = None
        else:
            label = " ".join(tokens[:unit_index])
            unit = tokens[unit_index]
            value = tokens[-1]
        if label:
            sections[current][label] = (value, unit)
    return sections


def parse_report(path: Path) -> dict[str, object]:
    text = path.read_text(encoding="utf-8")
    avg_match = re.search(r"avg_kernel_ms=([0-9.+-]+)", text)
    sections = parse_section_metrics(text)
    duration = sections.get("GPU Speed Of Light Throughput", {}).get(
        "Duration"
    )
    duration_ms = None
    if duration is not None:
        value, unit = duration
        try:
            duration_number = float(value)
            if unit == "us":
                duration_number /= 1000.0
            elif unit == "ns":
                duration_number /= 1000000.0
            duration_ms = f"{duration_number:.6g}"
        except ValueError:
            duration_ms = None
    metrics = {
        "duration_ms": duration_ms,
        "avg_kernel_ms": avg_match.group(1) if avg_match else None,
        "achieved_occupancy": metric_value(
            text, "Occupancy", "Achieved Occupancy", "%"
        ),
        "theoretical_occupancy": metric_value(
            text, "Occupancy", "Theoretical Occupancy", "%"
        ),
        "compute_throughput": metric_value(
            text,
            "GPU Speed Of Light Throughput",
            "Compute (SM) Throughput",
            "%",
        ),
        "memory_throughput": metric_value(
            text,
            "GPU Speed Of Light Throughput",
            "Memory Throughput",
            "%",
        ),
        "l1_hit_rate": metric_value(
            text, "Memory Workload Analysis", "L1/TEX Hit Rate", "%"
        ),
        "l2_hit_rate": metric_value(
            text, "Memory Workload Analysis", "L2 Hit Rate", "%"
        ),
        "registers": metric_value(
            text,
            "Launch Statistics",
            "Registers Per Thread",
            "register/thread",
        ),
        "block_size": metric_value(
            text, "Launch Statistics", "Block Size"
        ),
        "grid_size": metric_value(
            text, "Launch Statistics", "Grid Size"
        ),
        "waves_per_sm": metric_value(
            text, "Launch Statistics", "Waves Per SM"
        ),
    }
    metrics["sections"] = sections
    return metrics


def numeric(value: str | None) -> float:
    if value is None:
        return float("inf")
    try:
        return float(value)
    except ValueError:
        return float("inf")


def write_markdown(
    path: Path,
    source: Path,
    batch_stamp: str,
    launch_skip: int,
    launch_count: int,
    records: list[dict[str, object]],
) -> None:
    ordered = sorted(
        records,
        key=lambda record: numeric(record["metrics"]["duration_ms"]),
    )
    with path.open("w", encoding="utf-8") as report:
        report.write(f"# Nsight Compute comparison: {source.stem}\n\n")
        report.write(f"- Source: `{source}`\n")
        report.write(f"- Batch: `{batch_stamp}`\n")
        report.write(f"- Launch skip: `{launch_skip}`\n")
        report.write(f"- Launch count: `{launch_count}`\n")
        report.write(
            "- Sections: LaunchStats, Occupancy, SpeedOfLight, "
            "MemoryWorkloadAnalysis\n\n"
        )
        report.write(
            "Rows are sorted by Nsight Compute duration, lower is better. "
            "Microsecond durations are normalized to milliseconds.\n\n"
        )
        report.write("## Overview\n\n")
        report.write(
            "| Kernel | Alias | NCU ms | Status |\n"
        )
        report.write("|---|---|---:|---|\n")
        for record in ordered:
            metrics = record["metrics"]
            row = [
                record["kernel"],
                record["alias"] or "-",
                metrics["duration_ms"] or "-",
                "PASS" if record["return_code"] == 0 else "FAIL",
            ]
            report.write("| " + " | ".join(str(value) for value in row))
            report.write(" |\n")

        section_order = [
            "GPU Speed Of Light Throughput",
            "Memory Workload Analysis",
            "Launch Statistics",
            "Occupancy",
        ]
        for section_name in section_order:
            report.write(f"\n## {section_name}\n\n")
            report.write(
                "Each column is one kernel; metric values retain the NCU "
                "reported units.\n\n"
            )
            headers = [
                record["alias"] or record["kernel"] for record in ordered
            ]
            report.write("| Metric | " + " | ".join(headers) + " |\n")
            report.write(
                "|---|" + "---:|" * len(headers) + "\n"
            )
            metric_names: list[str] = []
            for record in ordered:
                section_metrics = record["metrics"]["sections"].get(
                    section_name, {}
                )
                for metric_name in section_metrics:
                    if metric_name not in metric_names:
                        metric_names.append(metric_name)
            for metric_name in metric_names:
                values = []
                for record in ordered:
                    section_metrics = record["metrics"]["sections"].get(
                        section_name, {}
                    )
                    value, unit = section_metrics.get(
                        metric_name, ("-", None)
                    )
                    values.append(
                        f"{value} {unit}" if unit is not None else value
                    )
                report.write(
                    "| " + metric_name + " | "
                    + " | ".join(values) + " |\n"
                )

        successful = [
            record
            for record in records
            if record["return_code"] == 0
            and record["metrics"]["duration_ms"] is not None
        ]
        if successful:
            fastest = min(
                successful,
                key=lambda record: numeric(
                    record["metrics"]["duration_ms"]
                ),
            )
            report.write("\n## Quick takeaways\n\n")
            report.write(
                f"- Fastest profiled duration: `{fastest['kernel']}` "
                f"at `{fastest['metrics']['duration_ms']} ms`.\n"
            )
            highest_occupancy = max(
                successful,
                key=lambda record: numeric(
                    record["metrics"]["achieved_occupancy"]
                ),
            )
            report.write(
                f"- Highest achieved occupancy: "
                f"`{highest_occupancy['kernel']}` "
                f"at `"
                f"{highest_occupancy['metrics']['achieved_occupancy']}%`.\n"
            )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Profile all kernel variants in a .cu file with ncu."
    )
    parser.add_argument("source", type=Path, help="CUDA source file")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/tmp/ncu_profiles"),
        help="Directory for individual and combined logs",
    )
    parser.add_argument(
        "--launch-skip",
        type=int,
        default=5,
        help="Kernel launches to skip before profiling",
    )
    parser.add_argument(
        "--launch-count",
        type=int,
        default=1,
        help="Number of launches to profile per kernel",
    )
    parser.add_argument(
        "--extra-arg",
        action="append",
        default=[],
        help="Extra argument passed to each CUDA executable run",
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
    batch_stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    combined_path = args.output_dir / (
        f"{source.stem}_all_{batch_stamp}.txt"
    )
    markdown_path = args.output_dir / (
        f"{source.stem}_all_{batch_stamp}.md"
    )

    failed = False
    records: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(
        prefix=f"{source.stem}_ncu_all_"
    ) as build:
        binary = Path(build) / source.stem
        compile_args = build_command(source, binary)
        print(f"Building once: {command_text(compile_args)}")
        compiled = subprocess.run(compile_args, check=False)
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
        if profile_mode and len(aliases) != len(kernels):
            print(
                "Note: some kernels have no detected harness alias; "
                "use --extra-arg to configure their launch path."
            )

        print(f"Kernels to profile ({len(kernels)}):")
        for kernel in kernels:
            print(f"  {kernel}")
        print(f"Combined log: {combined_path}")

        with combined_path.open("w", encoding="utf-8") as combined:
            combined.write(f"Source: {source}\n")
            combined.write(f"Kernels: {', '.join(kernels)}\n")
            combined.write(f"Batch: {batch_stamp}\n\n")

            for index, kernel in enumerate(kernels, start=1):
                print(f"\n[{index}/{len(kernels)}] Profiling {kernel}")
                command = ncu_command(
                    binary,
                    kernel,
                    profile_mode,
                    aliases.get(kernel),
                    args.launch_skip,
                    args.launch_count,
                    args.extra_arg,
                )
                individual_path = args.output_dir / (
                    f"{source.stem}_{kernel}_{batch_stamp}.txt"
                )
                return_code = stream_to_terminal_and_file(
                    command, individual_path
                )
                if return_code != 0:
                    failed = True
                records.append(
                    {
                        "kernel": kernel,
                        "alias": aliases.get(kernel),
                        "return_code": return_code,
                        "raw_path": individual_path,
                        "metrics": parse_report(individual_path),
                    }
                )

                combined.write("\n" + "=" * 79 + "\n")
                combined.write(f"KERNEL: {kernel}\n")
                combined.write("=" * 79 + "\n")
                combined.write(individual_path.read_text(encoding="utf-8"))
                combined.flush()

    write_markdown(
        markdown_path,
        source,
        batch_stamp,
        args.launch_skip,
        args.launch_count,
        records,
    )
    print(f"\nCombined log written to: {combined_path}")
    print(f"Markdown comparison written to: {markdown_path}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
