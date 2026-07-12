#!/usr/bin/env python3
"""Read-only-by-default build, evidence, and reporting tools."""

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
MANIFEST = TOOLS / "problems.json"
BASELINE = TOOLS / "protected_baseline.json"
WARNING_BASELINE = TOOLS / "warning_baseline.json"
EVIDENCE_KINDS = {
    "compile", "exact_fixture", "cpu_reference", "metamorphic",
    "sanitizer", "benchmark_unverified", "legacy_partial",
}


def load_json(path):
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def sha(data):
    if isinstance(data, str):
        data = data.encode()
    return hashlib.sha256(data).hexdigest()


def skip_space_and_comments(text, pos):
    while pos < len(text):
        if text[pos].isspace():
            pos += 1
        elif text.startswith("//", pos):
            end = text.find("\n", pos)
            pos = len(text) if end < 0 else end + 1
        elif text.startswith("/*", pos):
            end = text.find("*/", pos + 2)
            if end < 0:
                raise ValueError("unterminated comment")
            pos = end + 2
        else:
            break
    return pos


def matching(text, start, opening, closing):
    depth = 0
    i = start
    state = "code"
    while i < len(text):
        c = text[i]
        n = text[i:i + 2]
        if state == "code":
            if n == "//": state = "line"; i += 2; continue
            if n == "/*": state = "block"; i += 2; continue
            if c == '"': state = "string"; i += 1; continue
            if c == "'": state = "char"; i += 1; continue
            if c == opening: depth += 1
            elif c == closing:
                depth -= 1
                if depth == 0: return i
        elif state == "line" and c == "\n": state = "code"
        elif state == "block" and n == "*/": state = "code"; i += 2; continue
        elif state in ("string", "char"):
            if c == "\\": i += 2; continue
            if (state == "string" and c == '"') or (state == "char" and c == "'"):
                state = "code"
        i += 1
    raise ValueError(f"unmatched {opening} at byte {start}")


def extract_function(text, symbol):
    matches = list(re.finditer(r"\b" + re.escape(symbol) + r"\s*\(", text))
    definitions = []
    for match in matches:
        paren = text.find("(", match.start())
        end_paren = matching(text, paren, "(", ")")
        pos = skip_space_and_comments(text, end_paren + 1)
        while text.startswith(("const", "noexcept", "override", "final"), pos):
            word = re.match(r"\w+", text[pos:]).group(0)
            pos = skip_space_and_comments(text, pos + len(word))
        if pos < len(text) and text[pos] == "{":
            definitions.append((match.start(), matching(text, pos, "{", "}") + 1))
    if len(definitions) != 1:
        raise ValueError(
            f"{symbol}: expected one definition, found {len(definitions)}")
    name_start, end = definitions[0]
    line_start = text.rfind("\n", 0, name_start) + 1
    # Include attributes and return type back to the previous statement boundary.
    start = line_start
    while start > 0:
        prev_start = text.rfind("\n", 0, start - 1) + 1
        prev = text[prev_start:start].strip()
        if not prev or prev.endswith((";", "}")) or prev.startswith("//"):
            break
        start = prev_start
    return text[start:end]


def extract_declaration(text, symbol):
    matches = list(re.finditer(r"\b" + re.escape(symbol) + r"\b", text))
    declarations = []
    for match in matches:
        start = text.rfind("\n", 0, match.start()) + 1
        end = text.find(";", match.end())
        candidate = text[start:end + 1] if end >= 0 else ""
        declarator = re.search(
            r"\b" + re.escape(symbol) + r"\s*(?:=|\[)", candidate)
        if (end >= 0 and "\n" not in text[match.end():end]
                and declarator
                and ("constexpr" in candidate or "__constant__" in candidate)):
            declarations.append(candidate)
    unique = list(dict.fromkeys(declarations))
    if len(unique) != 1:
        raise ValueError(
            f"{symbol}: expected one declaration, found {len(unique)}")
    return unique[0]


def protected_slices(problem):
    text = (ROOT / problem["source"]).read_text(encoding="utf-8")
    symbols = problem["cpu"] + problem["kernels"] + problem["helpers"]
    symbols.append("solution")
    slices = {symbol: extract_function(text, symbol) for symbol in symbols}
    slices.update({symbol: extract_declaration(text, symbol)
                   for symbol in problem.get("constants", [])})
    return slices


def make_baseline():
    manifest = load_json(MANIFEST)
    result = {"schema_version": 1, "algorithm": "exact-source-sha256-v1",
              "problems": {}}
    for problem in manifest["problems"]:
        result["problems"][problem["id"]] = {
            symbol: sha(body)
            for symbol, body in protected_slices(problem).items()
        }
    return result


def check_protected():
    expected = load_json(BASELINE)
    current = make_baseline()
    failures = []
    for pid, symbols in expected["problems"].items():
        actual = current["problems"].get(pid, {})
        for symbol, digest in symbols.items():
            if actual.get(symbol) != digest:
                failures.append(f"{pid}:{symbol}")
    extras = set(current["problems"]) - set(expected["problems"])
    failures.extend(f"unexpected problem {x}" for x in sorted(extras))
    if failures:
        raise SystemExit("protected implementation changed: " + ", ".join(failures))
    print(f"protected implementation: PASS ({sum(map(len, expected['problems'].values()))} symbols)")


def run_capture(argv, cwd=ROOT):
    return subprocess.run(argv, cwd=cwd, text=True, capture_output=True)


def build(args):
    manifest = load_json(MANIFEST)
    warning_baseline = load_json(WARNING_BASELINE)["counts"]
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    failed = False
    for problem in manifest["problems"]:
        target = out / problem["id"].lower()
        cmd = [args.nvcc, "-std=c++17", "-Xcompiler=-Wall,-Wextra",
               problem["source"], "-o", str(target)]
        result = run_capture(cmd)
        print(" ".join(cmd))
        if result.stdout: print(result.stdout, end="")
        if result.stderr: print(result.stderr, end="", file=sys.stderr)
        warning_count = result.stderr.count("warning:")
        new_warnings = warning_count > warning_baseline.get(problem["id"], 0)
        status = "PASS" if result.returncode == 0 and not new_warnings else "FAIL"
        print(f"{problem['id']}: {status}; warnings={warning_count}; "
              f"baseline={warning_baseline.get(problem['id'], 0)}")
        failed |= result.returncode != 0 or new_warnings
    return 1 if failed else 0


def parse_timing_header(line):
    match = re.search(r"mode=(\w+) repeats=(\d+) warmup=(\d+) metric=(\w+)", line)
    return match.groupdict() if match else {}


def split_table_row(line):
    return line.split()


def import_log(path, problem):
    raw = path.read_bytes()
    lines = raw.decode("utf-8", errors="replace").splitlines()
    timing = parse_timing_header(lines[0]) if lines else {}
    records = []
    header = None
    for number, line in enumerate(lines, 1):
        fields = split_table_row(line)
        if fields and fields[0] == "group" and "kernel" in fields:
            header = fields
            continue
        if not header or not fields or set(line.strip()) <= {"-"}:
            continue
        if len(fields) != len(header):
            continue
        row = dict(zip(header, fields))
        if row.get("cpu") not in {"PASS", "REF", "SKIP", "FAIL"}:
            continue
        cpu, gpu = row["cpu"], row.get("gpu", "SKIP")
        if cpu == "PASS": kind = "exact_fixture"
        elif cpu == "REF": kind = "cpu_reference"
        elif gpu == "PASS": kind = "exact_fixture"
        else: kind = "benchmark_unverified"
        case = row.get("name", f"line-{number}")
        semantic = ":".join((problem["id"], case, row["kernel"],
                             row.get("block_x", ""), row.get("grid_x", "")))
        records.append({
            "schema_version": 1,
            "evidence_id": "legacy:" + sha(f"{sha(raw)}:{number}")[:20],
            "comparison_key": sha(semantic)[:20],
            "problem_id": problem["id"], "case_id": case,
            "kernel_id": row["kernel"], "evidence_kind": kind,
            "verification_status": gpu.lower(),
            "cpu_status": cpu.lower(), "provenance_quality": "legacy_partial",
            "dimensions": {k: int(v) for k, v in row.items()
                           if k in {"N", "K", "rows", "cols", "H",
                                    "kernel_size", "stride", "padding"}},
            "launch": {"block_x": int(row.get("block_x", 0)),
                       "grid_x": int(row.get("grid_x", 0))},
            "timing": {"kernel_ms": float(row["kernel_ms"]),
                       "reported_total_ms": float(row["total_ms"]), **timing},
            "legacy_source": {"path": path.name, "sha256": sha(raw),
                              "line": number},
        })
    return records


def import_legacy(args):
    manifest = load_json(MANIFEST)
    all_records = []
    for problem in manifest["problems"]:
        for suffix in ("with_cpu", "skip_cpu"):
            path = ROOT / f"{problem['log_prefix']}_{suffix}.txt"
            if path.exists(): all_records.extend(import_log(path, problem))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(x, sort_keys=True) + "\n"
                              for x in all_records), encoding="utf-8")
    print(f"imported {len(all_records)} rows to {output}")


def load_evidence(path):
    seen = set()
    records = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        record = json.loads(line)
        required = {"schema_version", "evidence_id", "problem_id",
                    "evidence_kind", "verification_status"}
        missing = required - record.keys()
        if missing: raise ValueError(f"line {number}: missing {sorted(missing)}")
        if record["evidence_kind"] not in EVIDENCE_KINDS:
            raise ValueError(f"line {number}: invalid evidence_kind")
        if record["evidence_id"] in seen:
            raise ValueError(f"line {number}: duplicate evidence_id")
        seen.add(record["evidence_id"]); records.append(record)
    return records


def validate(args):
    records = load_evidence(Path(args.evidence))
    print(f"evidence schema: PASS ({len(records)} records)")


def generate(args):
    records = load_evidence(Path(args.evidence))
    manifest = load_json(MANIFEST)
    by_problem = {}
    for record in records: by_problem.setdefault(record["problem_id"], []).append(record)
    lines = ["# Results Index", "",
             "Generated from validated evidence. Imported logs retain partial "
             "provenance.", "",
             "| Problem | Kernels | Verified rows | Benchmark-only | Status | Report |",
             "|---|---:|---:|---:|---|---|"]
    for problem in manifest["problems"]:
        rows = by_problem.get(problem["id"], [])
        kernels = len({x["kernel_id"] for x in rows})
        verified = sum(x["evidence_kind"] in {"exact_fixture", "cpu_reference"}
                       and x["verification_status"] == "pass" for x in rows)
        bench = sum(x["evidence_kind"] == "benchmark_unverified" for x in rows)
        failures = sum(x["verification_status"] == "fail" for x in rows)
        expected = len(problem["kernels"])
        if failures:
            status = "FAIL" if kernels == expected else "FAIL/INCOMPLETE"
        elif kernels < expected or bench == 0:
            status = "INCOMPLETE"
        else:
            status = "CURRENT"
        lines.append(f"| {problem['id']} {problem['name']} | {kernels}/{expected} | "
                     f"{verified} | {bench} | {status} | [{problem['report']}]({problem['report']}) |")
    output = Path(args.output)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"generated {output}")


def command_version(argv):
    result = run_capture(argv)
    if result.returncode:
        return None
    return (result.stdout or result.stderr).strip()


def capture_run(args):
    executable = Path(args.executable).resolve()
    if not executable.exists():
        raise SystemExit(f"executable not found: {executable}")
    now = dt.datetime.now(dt.timezone.utc)
    run_id = now.strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:12]
    root = Path(args.output_root) / run_id
    root.mkdir(parents=True, exist_ok=False)
    git_commit = command_version(["git", "rev-parse", "HEAD"])
    dirty = bool(run_capture(["git", "status", "--porcelain"]).stdout)
    arguments = args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments
    command = [str(executable), *arguments]
    result = run_capture(command)
    (root / "stdout.txt").write_text(result.stdout, encoding="utf-8")
    (root / "stderr.txt").write_text(result.stderr, encoding="utf-8")
    unavailable = "CUDA runtime unavailable" in result.stderr
    status = "skipped" if unavailable else (
        "complete" if result.returncode == 0 else "failed")
    source = Path(args.source).resolve() if args.source else None
    manifest = {
        "schema_version": 1, "run_id": run_id,
        "status": status,
        "timestamp_utc": now.isoformat(), "command": command,
        "exit_status": result.returncode, "git_commit": git_commit,
        "git_dirty": dirty, "binary_sha256": sha(executable.read_bytes()),
        "source_path": str(source) if source else None,
        "source_sha256": sha(source.read_bytes()) if source else None,
        "nvcc_version": command_version(["nvcc", "--version"]),
        "gpu": command_version([
            "nvidia-smi", "--query-gpu=index,uuid,name,compute_cap,driver_version",
            "--format=csv,noheader"]),
        "artifacts": {
            "stdout_sha256": sha(result.stdout),
            "stderr_sha256": sha(result.stderr),
        },
    }
    temp = root / "manifest.json.tmp"
    temp.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")
    temp.replace(root / "manifest.json")
    print(f"captured run {run_id} in {root}")
    return 0 if unavailable else result.returncode


def check(args):
    check_protected()
    result = run_capture(["git", "diff", "--check"])
    if result.returncode:
        print(result.stdout + result.stderr, file=sys.stderr); return 1
    print("git diff --check: PASS")
    if args.evidence:
        load_evidence(Path(args.evidence)); print("evidence schema: PASS")
    return 0


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("print-baseline")
    sub.add_parser("protect")
    p = sub.add_parser("build"); p.add_argument("--nvcc", default="nvcc"); p.add_argument("--output-dir", default="/tmp/tensara-build")
    p = sub.add_parser("import-legacy"); p.add_argument("--output", default="/tmp/tensara-legacy.jsonl")
    p = sub.add_parser("validate"); p.add_argument("evidence")
    p = sub.add_parser("generate-index"); p.add_argument("evidence"); p.add_argument("--output", default="/tmp/RESULTS_INDEX.md")
    p = sub.add_parser("capture-run")
    p.add_argument("--source")
    p.add_argument("executable")
    p.add_argument("arguments", nargs=argparse.REMAINDER)
    p.add_argument("--output-root", default="/tmp/tensara-runs")
    p = sub.add_parser("check"); p.add_argument("--evidence")
    args = parser.parse_args()
    if args.command == "print-baseline": print(json.dumps(make_baseline(), indent=2, sort_keys=True)); return 0
    if args.command == "protect": check_protected(); return 0
    if args.command == "build": return build(args)
    if args.command == "import-legacy": import_legacy(args); return 0
    if args.command == "validate": validate(args); return 0
    if args.command == "generate-index": generate(args); return 0
    if args.command == "capture-run": return capture_run(args)
    if args.command == "check": return check(args)


if __name__ == "__main__": sys.exit(main())
