#!/usr/bin/env python3
# --ProtoAI-Bakari--
# Story 15 (W3): long-context regression prompt generator.
#
# Emits deterministic "needle-in-haystack" prompts at three context lengths:
#   - 8K   : 50 prompts
#   - 16K  : 30 prompts
#   - 32K  : 20 prompts
#
# Pattern: random filler text padded to ~target tokens (rough char approx;
# ~4 chars/tok), with a unique "needle" (a specific fact + its key) embedded at
# a deterministic but varied position. The prompt ends with the question
# "What is the value associated with key XYZ?" — the model must recall the
# needle. Greedy temp=0 means deterministic baseline.
#
# Output JSONL schema matches gen_prompts.py so verify_gate.py grades unchanged.
#
# Run:
#   python3 gen_long_ctx_prompts.py --out long_ctx_prompts.jsonl
#   python3 gen_long_ctx_prompts.py --out long_ctx_prompts.jsonl --tiers 8K,16K
import argparse, json, os, random
from pathlib import Path

# ~4 chars/token approximation
CHARS_PER_TOK = 4

# Filler paragraph pool — deterministic, license-clean (paraphrased public-domain themes)
FILLER = [
    "The river flows slowly through the valley, carrying sediment from the highlands to the sea, depositing nutrients along the way. ",
    "Stars in the night sky have guided travelers for millennia, their patterns charted by ancient civilizations across every continent. ",
    "A scholar once observed that knowledge expands not by accumulation alone but by the careful pruning of error from prior assumptions. ",
    "Bees pollinate the flowering plants that produce a third of the food humans eat, making them quietly essential to civilization. ",
    "Languages evolve continuously, borrowing words from neighbors and inventing new terms for ideas that did not exist a generation earlier. ",
    "The library kept a catalog of every book in alphabetical order, organized by subject and cross-referenced by author and date. ",
    "Mountains rise as tectonic plates collide, then erode under wind and water over geological epochs that dwarf human history. ",
    "An apprentice baker learns first to weigh ingredients precisely, then to feel dough by hand, and only later to trust their senses without scales. ",
    "Ocean currents redistribute heat from the equator toward the poles, moderating climate in regions that would otherwise be uninhabitable. ",
    "Chess masters speak of position and tempo, the silent forces that shape every move long before the final tactical sequence appears. ",
    "Forests recycle carbon, water, and nutrients in cycles that took millions of years to balance and that humans now disturb in decades. ",
    "A lighthouse keeper recorded the daily weather in a leather-bound journal kept beside the brass instruments she trusted above all others. ",
]

# Needles: (key_fragment, value) — the model must produce VALUE when asked about KEY.
NEEDLES = [
    ("MAGENTA-19",   "the ledger weighed forty-two pounds"),
    ("CRIMSON-04",   "she carried seven brass keys"),
    ("AZURE-77",     "the ship returned on the eleventh day"),
    ("INDIGO-22",    "the message was sealed with green wax"),
    ("AMBER-58",     "a single black feather marked the trail"),
    ("OLIVE-31",     "the monastery bell rang at dawn and dusk"),
    ("VIOLET-66",    "three silver coins lay beneath the loose tile"),
    ("CORAL-13",     "the painter signed his work with two interlocking circles"),
    ("UMBER-99",     "the manuscript was bound with a red ribbon"),
    ("SCARLET-46",   "the cellar held barrels of cider, vinegar, and wine"),
]

TIERS = {
    "8K":  (8 * 1024, 50),
    "16K": (16 * 1024, 30),
    "32K": (32 * 1024, 20),
}


def make_prompt(rng, target_tokens, needle_key, needle_value, idx_within_tier):
    target_chars = target_tokens * CHARS_PER_TOK
    head = (
        "Read the following document carefully. Some sentences encode key/value pairs "
        "where a key is an uppercase color word followed by a hyphen and a number "
        "(for example INDIGO-22). After the document, you will be asked the value "
        "associated with one specific key.\n\n"
    )
    needle_sentence = f"NOTE: the key {needle_key} corresponds to: {needle_value}.\n"
    tail = (
        "\n\nQuestion: What is the value associated with key "
        f"{needle_key}? Answer in a single short sentence.\nAnswer:"
    )

    overhead = len(head) + len(needle_sentence) + len(tail)
    body_chars_needed = max(0, target_chars - overhead)
    pieces = []
    while sum(len(p) for p in pieces) < body_chars_needed:
        pieces.append(rng.choice(FILLER))
    body = "".join(pieces)
    body = body[:body_chars_needed]

    # Insert needle at a deterministic but varied position (10%-90% range)
    pos_frac = 0.10 + (idx_within_tier * 0.04) % 0.80
    cut = int(len(body) * pos_frac)
    body = body[:cut] + "\n" + needle_sentence + body[cut:]
    return head + body + tail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).parent / "long_ctx_prompts.jsonl"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tiers", default="8K,16K,32K", help="comma list of tiers to emit")
    args = ap.parse_args()
    rng = random.Random(args.seed)
    selected = [t.strip() for t in args.tiers.split(",") if t.strip()]
    out = []
    for tier in selected:
        if tier not in TIERS:
            print(f"  WARN: unknown tier {tier}, skipping")
            continue
        target_tokens, count = TIERS[tier]
        for i in range(count):
            needle = NEEDLES[(i + hash(tier)) % len(NEEDLES)]
            p = make_prompt(rng, target_tokens, needle[0], needle[1], i)
            out.append({
                "id": f"longctx_{tier}_{i:03d}",
                "category": "long_context",
                "tier": tier,
                "needle_key": needle[0],
                "needle_value_expected": needle[1],
                "approx_token_target": target_tokens,
                "approx_chars": len(p),
                "prompt": p,
                "max_tokens": 96,
                "stop": ["\n\n"],
            })
    with open(args.out, "w") as f:
        for o in out:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
    print(f"wrote {len(out)} long-context prompts to {args.out}")
    by = {}
    for o in out:
        by[o["tier"]] = by.get(o["tier"], 0) + 1
    for t, c in by.items():
        print(f"  {t}: {c}")


if __name__ == "__main__":
    main()
