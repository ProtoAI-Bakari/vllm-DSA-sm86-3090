#!/usr/bin/env python3
# --ProtoAI-Bakari--
# test_optimize_loop.py — smoke test for bench/optimize_loop.py.

import json, os, subprocess, sys, tempfile, unittest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCRIPT = os.path.join(ROOT, "bench", "optimize_loop.py")


def write_tier(path, agg_tier, conc1_tier):
    with open(path, "w") as f:
        json.dump({
            "aggregate": {"tier": agg_tier, "value": 0, "threshold_met": 0,
                          "next_tier": None, "next_threshold": None, "gap_to_next": None},
            "conc1": {"tier": conc1_tier, "value": 0, "threshold_met": 0,
                      "next_tier": None, "next_threshold": None, "gap_to_next": None},
            "by_conc": {},
        }, f)


class OptimizeLoop(unittest.TestCase):
    def test_below_bronze_proposes_int4_first(self):
        with tempfile.TemporaryDirectory() as td:
            tp = os.path.join(td, "tier.json")
            write_tier(tp, "below_bronze", "below_bronze")
            out = subprocess.check_output([sys.executable, SCRIPT, "--tier-classify", tp]).decode()
            data = json.loads(out)
            names = [c["name"] for c in data["candidates"]]
            self.assertIn("int4_awq_marlin_sm86", names)
            self.assertEqual(data["candidates"][0]["name"], "int4_awq_marlin_sm86")

    def test_diamond_proposes_eagle3(self):
        with tempfile.TemporaryDirectory() as td:
            tp = os.path.join(td, "tier.json")
            write_tier(tp, "diamond", "diamond")
            out = subprocess.check_output([sys.executable, SCRIPT, "--tier-classify", tp]).decode()
            data = json.loads(out)
            names = [c["name"] for c in data["candidates"]]
            self.assertIn("eagle3_speculative_decode", names)

    def test_silver_includes_eplb(self):
        with tempfile.TemporaryDirectory() as td:
            tp = os.path.join(td, "tier.json")
            write_tier(tp, "silver", "silver")
            out = subprocess.check_output([sys.executable, SCRIPT, "--tier-classify", tp]).decode()
            data = json.loads(out)
            names = [c["name"] for c in data["candidates"]]
            self.assertIn("eplb_load_balancer", names)


if __name__ == "__main__":
    unittest.main()
