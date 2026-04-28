#!/usr/bin/env python3
# --ProtoAI-Bakari--
# test_tier_classify.py — smoke test for bench/runners/tier_classify.py.
# Pure unit (no endpoint), runs in <1s. Verifies tier ladder logic against
# known thresholds from corpus §8.1.

import json, os, subprocess, sys, tempfile, unittest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCRIPT = os.path.join(ROOT, "bench", "runners", "tier_classify.py")


def run_classify(*args) -> dict:
    out = subprocess.check_output([sys.executable, SCRIPT, *args]).decode()
    return json.loads(out)


class TierLadder(unittest.TestCase):
    def test_aggregate_below_bronze(self):
        r = run_classify("--agg-tps", "50", "--conc1-tps", "10")
        self.assertEqual(r["aggregate"]["tier"], "below_bronze")
        self.assertEqual(r["conc1"]["tier"], "below_bronze")

    def test_aggregate_bronze_lower_bound(self):
        r = run_classify("--agg-tps", "100", "--conc1-tps", "25")
        self.assertEqual(r["aggregate"]["tier"], "bronze")
        self.assertEqual(r["conc1"]["tier"], "bronze")

    def test_aggregate_silver(self):
        r = run_classify("--agg-tps", "350", "--conc1-tps", "40")
        self.assertEqual(r["aggregate"]["tier"], "silver")
        self.assertEqual(r["conc1"]["tier"], "silver")

    def test_aggregate_gold(self):
        r = run_classify("--agg-tps", "550", "--conc1-tps", "60")
        self.assertEqual(r["aggregate"]["tier"], "gold")
        self.assertEqual(r["conc1"]["tier"], "gold")

    def test_aggregate_diamond_conc1_diamond(self):
        r = run_classify("--agg-tps", "950", "--conc1-tps", "80")
        self.assertEqual(r["aggregate"]["tier"], "diamond")
        self.assertEqual(r["conc1"]["tier"], "diamond")

    def test_aggregate_cosmic(self):
        r = run_classify("--agg-tps", "1100", "--conc1-tps", "75")
        self.assertEqual(r["aggregate"]["tier"], "cosmic")

    def test_gap_to_next(self):
        r = run_classify("--agg-tps", "150", "--conc1-tps", "30")
        agg = r["aggregate"]
        self.assertEqual(agg["tier"], "bronze")
        self.assertEqual(agg["next_tier"], "silver")
        self.assertEqual(agg["gap_to_next"], 150)


class ConcSweepParse(unittest.TestCase):
    def test_synthetic_sweep(self):
        with tempfile.TemporaryDirectory() as td:
            raw_path = os.path.join(td, "raw_conc1.jsonl")
            with open(raw_path, "w") as f:
                for i in range(4):
                    f.write(json.dumps({
                        "id": i, "ok": True, "wall_s": 2.4,
                        "completion_tokens": 60, "prompt_tokens": 12,
                        "tps_per_request": 25.0,
                    }) + "\n")
            summary = os.path.join(td, "sweep.jsonl")
            with open(summary, "w") as f:
                f.write(json.dumps({
                    "ts": "2026-04-28T10:00:00Z",
                    "profile_endpoint": "http://cuda1:8000",
                    "model": "glm51-iq2xxs",
                    "conc": 1, "step_wall_s": 2.4, "raw": raw_path,
                }) + "\n")
            r = run_classify("--conc-sweep", summary)
            self.assertIn("by_conc", r)
            self.assertIn("1", r["by_conc"])
            self.assertAlmostEqual(r["by_conc"]["1"]["agg_tps"], 100.0, places=1)
            self.assertEqual(r["aggregate"]["tier"], "bronze")
            self.assertEqual(r["conc1"]["tier"], "bronze")


if __name__ == "__main__":
    unittest.main()
