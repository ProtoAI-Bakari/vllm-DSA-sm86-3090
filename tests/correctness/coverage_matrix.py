#!/usr/bin/env python3
# --ProtoAI-Bakari--
# METRICS_OK gate coverage tooling; not a perf bench
# Story W3-NEW: kernel × test-tier coverage matrix.
#
# Walks the repo and reports which sm_86 kernels have:
#   - a CPU pytorch reference                  (tests/correctness/cpu_reference.py REFERENCES)
#   - an L1 fixture file                       (tests/unit/fixtures/<name>.pt)
#   - an L1 unit test                          (tests/unit/test_kernels_sm86.py contains "test_<name>")
#   - an L2 forward-slice test                 (tests/integration/test_l2_forward_slice.py contains "test_<name>")
#
# Helps CC3/CC4/CC5 spot gaps before merging — without this, a kernel could ship
# without an L1 negative-test, slipping a regression past the gate.
#
# Run:
#   python3 coverage_matrix.py [--repo <path>] [--report <md>]
#   exit 0 if all kernels covered at L0 (ref) + L1 (test) + L2 (test)
#   exit 1 if any gap
import argparse, importlib.util, os, re, sys, time
from pathlib import Path


def load_references(cpu_ref_path):
    spec = importlib.util.spec_from_file_location("cpu_reference", cpu_ref_path)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as e:
        print(f"  WARN: could not import {cpu_ref_path}: {e}", file=sys.stderr)
        return {}
    return getattr(mod, "REFERENCES", {})


def kernels_with_test(test_path):
    if not test_path.exists():
        return set()
    text = test_path.read_text()
    return set(re.findall(r"def test_([A-Za-z_]\w*)\s*\(", text))


def fixture_present(fixtures_dir, name):
    return (fixtures_dir / f"{name}.pt").exists()


def check(repo):
    cpu_ref = repo / "tests" / "correctness" / "cpu_reference.py"
    fixtures_dir = repo / "tests" / "unit" / "fixtures"
    l1_test = repo / "tests" / "unit" / "test_kernels_sm86.py"
    l2_test = repo / "tests" / "integration" / "test_l2_forward_slice.py"

    refs = load_references(cpu_ref) if cpu_ref.exists() else {}
    l1_tests = kernels_with_test(l1_test)
    l2_tests = kernels_with_test(l2_test)
    rows = []
    gaps = 0
    for name in sorted(refs):
        l0 = True
        l1_fix = fixture_present(fixtures_dir, name)
        l1_t = name in l1_tests or any(t.endswith(name) for t in l1_tests)
        l2_t = (
            f"{name}_in_chain" in l2_tests
            or name in l2_tests
            or any(t.endswith(name) for t in l2_tests)
            or any(t.endswith(f"{name}_in_chain") for t in l2_tests)
        )
        rows.append({
            "kernel": name,
            "L0_cpu_ref": l0,
            "L1_fixture": l1_fix,
            "L1_test": l1_t,
            "L2_test": l2_t,
        })
        if not (l1_fix and l1_t and l2_t):
            gaps += 1
    return rows, gaps


def render(rows, gaps):
    L = []
    L.append(f"# CC6 Kernel Coverage Matrix — {time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    L.append("")
    L.append(f"**Authored by ProtoAI-Bakari, with Assistance by Claude Opus 4.7 (claude-opus-4-7) [agent: CC6]**")
    L.append("")
    if not rows:
        L.append("(No kernels found in `tests/correctness/cpu_reference.py REFERENCES`.)")
        return "\n".join(L) + "\n"
    L.append(f"Status: **{len(rows) - gaps} / {len(rows)} kernels fully covered** (L0+L1+L2). Gaps: **{gaps}**.")
    L.append("")
    L.append("| Kernel | L0 CPU ref | L1 fixture | L1 test | L2 test |")
    L.append("|---|---|---|---|---|")
    for r in rows:
        def cell(b):
            return "✓" if b else "—"
        L.append(f"| {r['kernel']} | {cell(r['L0_cpu_ref'])} | {cell(r['L1_fixture'])} | {cell(r['L1_test'])} | {cell(r['L2_test'])} |")
    L.append("")
    if gaps:
        L.append("## Action items")
        for r in rows:
            missing = []
            if not r["L1_fixture"]:
                missing.append("regenerate L1 fixture (run `python3 tests/unit/gen_fixtures.py --only " + r["kernel"] + "`)")
            if not r["L1_test"]:
                missing.append(f"add `def test_{r['kernel']}(...)` in tests/unit/test_kernels_sm86.py")
            if not r["L2_test"]:
                missing.append(f"add `def test_{r['kernel']}_in_chain(...)` in tests/integration/test_l2_forward_slice.py")
            if missing:
                L.append(f"- **{r['kernel']}**: " + "; ".join(missing))
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=str(Path(__file__).resolve().parents[2]))
    ap.add_argument("--report", default=None)
    args = ap.parse_args()
    repo = Path(args.repo)
    rows, gaps = check(repo)
    md = render(rows, gaps)
    out = args.report or os.path.expanduser("~/AGENT/comms/CC6_COVERAGE_MATRIX.md")
    with open(out, "w") as f:
        f.write(md)
    sys.stdout.write(md)
    print(f"\n[coverage_matrix] kernels={len(rows)} gaps={gaps} → {out}", file=sys.stderr)
    sys.exit(0 if gaps == 0 else 1)


if __name__ == "__main__":
    main()
