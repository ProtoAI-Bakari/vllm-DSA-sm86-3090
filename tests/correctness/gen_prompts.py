#!/usr/bin/env python3
# --ProtoAI-Bakari--
# Deterministic 1500-prompt fixed-seed correctness test set generator.
# Run: python3 gen_prompts.py [--out prompts.jsonl] [--seed 42]
# Output: JSONL, one prompt per line, schema:
#   {"id": "cat_NNNN", "category": str, "prompt": str, "max_tokens": int, "stop": [str]|null}
# Categories + counts (total 1500):
#   factual       200   short answer (16 tok)
#   mmlu_style    300   4-option multiple choice (placeholder; real MMLU corpus loaded if MMLU_DIR set)
#   reasoning     200   multi-step word problem (96 tok)
#   code          200   short python/bash completion (160 tok)
#   summarization 150   paragraph -> summary (96 tok)
#   creative      150   1-2 sentence in style (64 tok)
#   multilingual  100   translation (48 tok)
#   repetition    100   attention-sink adversarial (80 tok)
#   edge          100   empty/1-tok/special-char/very-long (64 tok)
#
# AO1 audit (2026-04-27): 1500 prompts is statistical floor for 1% regression detection at 95% CI.
# Gate (LEAD corpus PART2 §8.4): top-1 >=98% AND logit cosine >=0.97 AND perplexity <=2% on MMLU subset.
import argparse, json, os, random, sys
from pathlib import Path

# -------------------- bank: factual --------------------
FACTUAL_TEMPLATES = [
    ("The capital of {country} is", ["France","Germany","Japan","Brazil","Egypt","Australia","Canada","India","Mexico","South Korea","Italy","Spain","Argentina","Turkey","Vietnam","Poland","Sweden","Norway","Finland","Greece"]),
    ("The chemical symbol for {element} is", ["gold","silver","iron","copper","sodium","potassium","mercury","tungsten","oxygen","hydrogen","helium","neon","argon","krypton","xenon","carbon","nitrogen","sulfur","chlorine","calcium"]),
    ("The author of '{book}' is", ["1984","Pride and Prejudice","Moby Dick","War and Peace","The Great Gatsby","Crime and Punishment","Don Quixote","Ulysses","One Hundred Years of Solitude","To Kill a Mockingbird","The Brothers Karamazov","Anna Karenina","Brave New World","Catch-22","The Catcher in the Rye","Frankenstein","Dracula","The Hobbit","Beloved","Lolita"]),
    ("In what year did {event} occur?", ["the moon landing","the fall of the Berlin Wall","the French Revolution","the signing of the Magna Carta","the start of World War I","the end of World War II","the discovery of penicillin","the publication of Origin of Species","the invention of the telephone","the launch of Sputnik"]),
    ("The largest planet in our solar system is", [""]),
    ("The speed of light in vacuum is approximately", [""]),
    ("The deepest part of the Earth's oceans is the", [""]),
    ("The longest river in {region} is", ["Africa","South America","Asia","Europe","North America","Australia"]),
    ("The smallest country in the world by area is", [""]),
    ("The currency of {country} is", ["Japan","United Kingdom","Switzerland","India","Brazil","Russia","China","South Africa","Mexico","Egypt"]),
]

# -------------------- bank: mmlu_style placeholders --------------------
# Real MMLU corpus loaded from MMLU_DIR env (CSVs from cais/mmlu); else deterministic placeholders.
MMLU_PLACEHOLDER_SUBJECTS = ["high_school_mathematics","college_physics","professional_law","clinical_knowledge","high_school_biology","computer_science","electrical_engineering","econometrics","formal_logic","abstract_algebra"]

# -------------------- bank: reasoning --------------------
REASONING_TEMPLATES = [
    "If a train leaves city A at {h1}:00 going {s1} mph and another leaves city B at {h2}:00 going {s2} mph toward each other, and the cities are {d} miles apart, when do they meet? Show your work.",
    "A bag has {r} red balls, {b} blue balls, {g} green balls. If you draw 2 without replacement, what is the probability both are red? Show your work.",
    "Sarah is {sa} years old. Her brother is {ba} years younger. In {y} years, the sum of their ages will be {target}. Find the current ages of both. Show your work.",
    "A rectangle has perimeter {p} and area {a}. Find its dimensions. Show your work.",
    "If {x}^2 + {y}x + {z} = 0, solve for x using the quadratic formula. Show your work.",
    "A store offers a {d}% discount, then adds {t}% sales tax. If the original price is ${p}, what is the final price? Show your work.",
    "Three friends split a bill of ${total} so that A pays twice as much as B, and C pays the same as A. How much does each pay? Show your work.",
    "A tank is filled by pipe X in {x} hours and pipe Y in {y} hours. How long to fill if both run together? Show your work.",
    "A right triangle has legs of {a} and {b}. Find the hypotenuse. Show your work.",
    "If 3x - 7 = {target}, find x and verify. Show your work.",
]

# -------------------- bank: code --------------------
CODE_TEMPLATES = [
    "# Write a Python function that returns the {nth} Fibonacci number using memoization.\ndef fib(n):\n",
    "# Implement quicksort in Python.\ndef quicksort(arr):\n",
    "# Implement binary search returning the index, or -1 if not found.\ndef binary_search(arr, target):\n",
    "# Reverse a singly linked list in Python.\nclass ListNode:\n    def __init__(self, val=0, next=None):\n        self.val = val\n        self.next = next\n\ndef reverse_list(head):\n",
    "# Check if a string is a palindrome ignoring case and non-alphanumeric.\ndef is_palindrome(s):\n",
    "# Bash one-liner: find all .log files in /var/log modified in last {n} hours, print sizes sorted desc.\n",
    "# Python: parse a CSV file 'data.csv' with columns 'name,age,city', return list of dicts.\ndef parse_csv(path):\n",
    "# Implement a thread-safe LRU cache of size {n} in Python.\nfrom collections import OrderedDict\nimport threading\nclass LRU:\n",
    "# Find the longest common subsequence of two strings.\ndef lcs(a, b):\n",
    "# Implement Dijkstra's shortest path on an adjacency dict.\nimport heapq\ndef dijkstra(graph, start):\n",
    "# Bash: tail -F a log file and prefix each line with the local timestamp in ISO 8601.\n",
    "# Python: given an integer n, return the nth row of Pascal's triangle.\ndef pascal_row(n):\n",
    "# Python: validate an IPv4 address string strictly (no leading zeros, octets 0-255).\ndef is_valid_ipv4(s):\n",
    "# Implement merge sort recursively.\ndef merge_sort(arr):\n",
    "# Find all permutations of a list (no duplicates assumed) without itertools.\ndef permutations(arr):\n",
]

# -------------------- bank: summarization --------------------
SUMM_TEXTS = [
    "Photosynthesis is a biological process used by plants, algae, and certain bacteria to convert light energy into chemical energy stored in glucose. The process occurs primarily in the chloroplasts of plant cells, specifically within structures called thylakoids. The overall reaction combines carbon dioxide and water in the presence of light to produce glucose and oxygen as a byproduct. Photosynthesis underpins nearly all life on Earth by providing the foundational energy source for food chains and by producing the atmospheric oxygen that aerobic organisms require.",
    "The Roman Empire reached its greatest territorial extent under the emperor Trajan in the early second century, encompassing lands from Britain in the north to Egypt in the south and from Spain in the west to Mesopotamia in the east. Its decline over the following centuries is attributed to a combination of economic strain, military overextension, political instability, and waves of barbarian incursions. The Western half formally fell in 476 CE, while the Eastern half, known as the Byzantine Empire, persisted for nearly another thousand years until the fall of Constantinople in 1453.",
    "CRISPR-Cas9 is a gene-editing technology adapted from a bacterial immune system that uses a guide RNA to direct the Cas9 enzyme to a specific DNA sequence, where it makes a cut. Cells repair the cut either by joining the ends back together (often introducing small insertions or deletions that disable a gene) or by inserting a new sequence provided by the experimenter. CRISPR has revolutionized molecular biology by making genome modification fast, cheap, and precise enough for use in basic research, agriculture, and clinical therapeutics.",
    "Quantum entanglement is a phenomenon in which two or more particles become correlated such that the quantum state of each cannot be described independently of the others, even when separated by large distances. Measurement of one entangled particle instantaneously fixes the corresponding property of its partner, a behavior Einstein famously called 'spooky action at a distance.' Entanglement is a key resource in quantum information science, underlying applications such as quantum cryptography, dense coding, and quantum teleportation.",
    "Plate tectonics is the unifying theory of geology that describes the Earth's outer shell as composed of large lithospheric plates that move slowly over the underlying mantle. Plates can pull apart at divergent boundaries, collide at convergent boundaries, or slide past one another at transform boundaries. These interactions generate earthquakes, volcanic activity, mountain ranges, and ocean trenches, and over geological time they have produced and broken up supercontinents in cycles lasting hundreds of millions of years.",
    "The Industrial Revolution, beginning in mid-eighteenth-century Britain and spreading through Europe and North America in the nineteenth century, marked the transition from agrarian, hand-production economies to mechanized factory systems powered first by water and then by steam. It produced unprecedented gains in productivity, urbanization, and material wealth, but also harsh labor conditions, pollution, and social upheaval. Its long-term legacy includes modern capitalism, the rise of industrial cities, and the technological foundations of the twentieth century.",
    "Black holes are regions of spacetime where gravity is so strong that not even light can escape once it crosses the event horizon. They form when massive stars collapse at the end of their lives or by the merger of dense remnants. Outside the event horizon, matter falling in can heat to enormous temperatures and emit X-rays detectable by observatories, and gravitational waves from merging black holes have been directly observed since 2015.",
    "Vaccination works by exposing the immune system to a harmless component of a pathogen so that the body develops memory cells capable of mounting a rapid response if it later encounters the real pathogen. Modern vaccines include attenuated live viruses, inactivated viruses, subunit proteins, and mRNA-encoded antigens. Widespread vaccination has eradicated smallpox, eliminated polio from most of the world, and dramatically reduced mortality from diseases such as measles, diphtheria, and pertussis.",
    "Climate change refers to long-term shifts in temperature and weather patterns, predominantly driven since the mid-twentieth century by human emissions of greenhouse gases such as carbon dioxide and methane. Consequences include rising sea levels, more frequent extreme weather, ocean acidification, and shifting ranges of species. Mitigation strategies focus on decarbonizing energy, transportation, and industry, while adaptation strategies aim to strengthen resilience of communities, agriculture, and infrastructure.",
    "The human brain contains roughly 86 billion neurons, each forming thousands of synaptic connections with others, organized into specialized regions that handle perception, motor control, memory, language, and decision-making. Neuroscience studies the brain at scales from individual ion channels to large-scale networks, using techniques ranging from patch-clamp electrophysiology to functional magnetic resonance imaging. Understanding how the activity of billions of neurons gives rise to subjective experience remains one of the deepest open problems in science.",
]

# -------------------- bank: creative --------------------
CREATIVE_TEMPLATES = [
    "Write a single sentence describing a sunset over {place} in the style of Hemingway.",
    "Write two sentences of dialogue between a {char1} and a {char2} who have just met on a train.",
    "Write a haiku (5-7-5 syllables) about {topic}.",
    "Write a one-paragraph product description for {product} aimed at {audience}.",
    "Write the opening sentence of a noir detective novel set in {place}.",
    "In one sentence, describe {object} from the perspective of someone who has never seen one before.",
    "Write a four-line limerick about {topic}.",
    "Compose a short tweet (<= 280 chars) explaining {concept} to a curious 12-year-old.",
    "Write an opening line for a fantasy novel that involves {element1} and {element2}.",
    "Write a single sentence describing the smell of {scent} using only sound-related metaphors.",
]

# -------------------- bank: multilingual --------------------
MULTI_TEMPLATES = [
    'Translate to French: "{text}"',
    'Translate to Spanish: "{text}"',
    'Translate to German: "{text}"',
    'Translate to Japanese: "{text}"',
    'Translate to Mandarin Chinese (simplified): "{text}"',
    'Translate to Russian: "{text}"',
    'Translate to Arabic: "{text}"',
    'Translate to Portuguese: "{text}"',
    'Translate to Hindi: "{text}"',
    'Translate to Korean: "{text}"',
]
MULTI_SENTENCES = [
    "The quick brown fox jumps over the lazy dog.",
    "Knowledge is power.",
    "Time waits for no one.",
    "Better late than never.",
    "Honesty is the best policy.",
    "Actions speak louder than words.",
    "Where there is smoke, there is fire.",
    "Practice makes perfect.",
    "A picture is worth a thousand words.",
    "The early bird catches the worm.",
]

# -------------------- bank: repetition stress --------------------
REPETITION_TEMPLATES = [
    "Repeat the word '{word}' exactly {n} times separated by spaces.",
    "Continue the sequence: {seq}",
    "Count from {start} to {end} as a comma-separated list.",
    "List the first {n} prime numbers separated by commas.",
    "Write the alphabet from {a} to {b} in lowercase, separated by '-'.",
    "Repeat back exactly: '{phrase}'",
    "Echo this number {n} times on separate lines: {num}",
    "Generate a sequence of {n} alternating 'cat' and 'dog' separated by commas.",
    "Output the integers from {a} to {b}, each on its own line.",
    "Write the multiplication table for {n} (n*1 through n*12), one per line.",
]

# -------------------- bank: edge cases --------------------
EDGE_CASES = [
    {"prompt": "", "max_tokens": 8, "tag": "empty"},
    {"prompt": " ", "max_tokens": 8, "tag": "single_space"},
    {"prompt": "a", "max_tokens": 16, "tag": "single_char"},
    {"prompt": "?", "max_tokens": 16, "tag": "single_punct"},
    {"prompt": "Tell me." * 50, "max_tokens": 32, "tag": "long_repeat"},
    {"prompt": "🚀 " * 20, "max_tokens": 32, "tag": "emoji"},
    {"prompt": "<|im_start|>user\nHello<|im_end|>", "max_tokens": 32, "tag": "chat_tokens"},
    {"prompt": "```python\n", "max_tokens": 64, "tag": "code_open"},
    {"prompt": "<<<", "max_tokens": 16, "tag": "special_chars"},
    {"prompt": "\\n\\t\\r", "max_tokens": 16, "tag": "escape_chars"},
]

# -------------------- generators --------------------
def gen_factual(rng, n=200):
    out = []
    for i in range(n):
        tmpl, opts = rng.choice(FACTUAL_TEMPLATES)
        if "{country}" in tmpl: p = tmpl.format(country=rng.choice(opts))
        elif "{element}" in tmpl: p = tmpl.format(element=rng.choice(opts))
        elif "{book}" in tmpl: p = tmpl.format(book=rng.choice(opts))
        elif "{event}" in tmpl: p = tmpl.format(event=rng.choice(opts))
        elif "{region}" in tmpl: p = tmpl.format(region=rng.choice(opts))
        else: p = tmpl
        out.append({"id": f"factual_{i:04d}", "category": "factual", "prompt": p, "max_tokens": 16, "stop": ["\n\n"]})
    return out

def load_real_mmlu(mmlu_dir):
    """Try loading real MMLU corpus from CSVs in mmlu_dir/test/. Returns list of dicts or None."""
    import csv, glob
    if not mmlu_dir or not os.path.isdir(mmlu_dir):
        return None
    files = sorted(glob.glob(os.path.join(mmlu_dir, "test", "*_test.csv")))
    if not files:
        files = sorted(glob.glob(os.path.join(mmlu_dir, "*_test.csv")))
    if not files:
        return None
    items = []
    for f in files:
        subj = os.path.basename(f).replace("_test.csv", "")
        with open(f, newline="") as fh:
            rd = csv.reader(fh)
            for row in rd:
                if len(row) >= 6:
                    q, a, b, c, d, ans = row[0], row[1], row[2], row[3], row[4], row[5]
                    items.append({"subject": subj, "q": q, "A": a, "B": b, "C": c, "D": d, "ans": ans})
    return items

def gen_mmlu(rng, n=300):
    out = []
    real = load_real_mmlu(os.environ.get("MMLU_DIR"))
    if real:
        sampled = rng.sample(real, min(n, len(real)))
        for i, item in enumerate(sampled):
            p = f"The following is a multiple choice question about {item['subject'].replace('_',' ')}.\n\nQuestion: {item['q']}\nA. {item['A']}\nB. {item['B']}\nC. {item['C']}\nD. {item['D']}\nAnswer:"
            out.append({"id": f"mmlu_{i:04d}", "category": "mmlu_real", "subject": item["subject"], "prompt": p, "max_tokens": 4, "stop": ["\n"], "expected_letter": item["ans"]})
        # If real corpus smaller than n, pad with placeholders
        if len(out) < n:
            out.extend(_mmlu_placeholder(rng, n - len(out), start=len(out)))
        return out
    return _mmlu_placeholder(rng, n)

def _mmlu_placeholder(rng, n, start=0):
    out = []
    for i in range(n):
        subj = rng.choice(MMLU_PLACEHOLDER_SUBJECTS)
        a, b, c, d = rng.sample(["alpha","beta","gamma","delta","omega","sigma","theta","phi","psi","rho","mu","nu"], 4)
        p = f"The following is a multiple choice question about {subj.replace('_',' ')}.\n\nQuestion: Which of the following is most closely associated with concept #{rng.randint(1,9999)} in {subj.replace('_',' ')}?\nA. {a}\nB. {b}\nC. {c}\nD. {d}\nAnswer:"
        out.append({"id": f"mmlu_{start+i:04d}", "category": "mmlu_placeholder", "subject": subj, "prompt": p, "max_tokens": 4, "stop": ["\n"]})
    return out

def gen_reasoning(rng, n=200):
    out = []
    for i in range(n):
        tmpl = rng.choice(REASONING_TEMPLATES)
        p = tmpl.format(
            h1=rng.randint(6,11), h2=rng.randint(6,11), s1=rng.randint(40,90), s2=rng.randint(40,90),
            d=rng.randint(100,500),
            r=rng.randint(2,8), b=rng.randint(2,8), g=rng.randint(2,8),
            sa=rng.randint(20,40), ba=rng.randint(2,10), y=rng.randint(5,15), target=rng.randint(60,120),
            p=rng.choice([20,24,28,32,36,40]), a=rng.choice([24,32,48,60,80]),
            x=rng.randint(2,5), y_=rng.randint(-10,10), z=rng.randint(-20,20),
            d_=rng.randint(10,40), t=rng.randint(5,15),
            total=rng.randint(60,300),
            a_=rng.randint(3,12), b_=rng.randint(4,16),
        ) if "{p}" in tmpl else tmpl
        # Re-render with all keys present (simpler: keep substitution best-effort)
        try:
            p = tmpl.format(
                h1=rng.randint(6,11), h2=rng.randint(6,11), s1=rng.randint(40,90), s2=rng.randint(40,90),
                d=rng.randint(100,500), r=rng.randint(2,8), b=rng.randint(2,8), g=rng.randint(2,8),
                sa=rng.randint(20,40), ba=rng.randint(2,10), y=rng.randint(5,15), target=rng.randint(60,120),
                p=rng.choice([20,24,28,32,36,40]), a=rng.choice([24,32,48,60,80]),
                x=rng.randint(2,5), z=rng.randint(-20,20),
                t=rng.randint(5,15), total=rng.randint(60,300),
            )
        except KeyError:
            p = tmpl
        out.append({"id": f"reasoning_{i:04d}", "category": "reasoning", "prompt": p, "max_tokens": 96, "stop": ["\n\n\n"]})
    return out

def gen_code(rng, n=200):
    out = []
    for i in range(n):
        tmpl = rng.choice(CODE_TEMPLATES)
        try:
            p = tmpl.format(nth=rng.choice(["10th","20th","50th","100th"]), n=rng.randint(3,16))
        except KeyError:
            p = tmpl
        out.append({"id": f"code_{i:04d}", "category": "code", "prompt": p, "max_tokens": 160, "stop": ["\n\n\n"]})
    return out

def gen_summ(rng, n=150):
    out = []
    for i in range(n):
        text = rng.choice(SUMM_TEXTS)
        p = f"Summarize the following passage in two sentences:\n\n{text}\n\nSummary:"
        out.append({"id": f"summ_{i:04d}", "category": "summarization", "prompt": p, "max_tokens": 96, "stop": ["\n\n"]})
    return out

def gen_creative(rng, n=150):
    out = []
    fillers = {
        "place": ["Lisbon","a small island","the Pacific","a desert canyon","Manhattan","an Alaskan glacier"],
        "char1": ["physicist","retired chef","astronaut","linguist","child"],
        "char2": ["spy","monk","conductor","poet","traveling salesman"],
        "topic": ["autumn leaves","insomnia","first snowfall","an old harbor","quantum spin","a quiet library"],
        "product": ["a noise-cancelling pillow","a self-watering plant pot","a solar-powered backpack","a fold-flat kayak"],
        "audience": ["college students","apartment dwellers","parents of toddlers","minimalist travelers"],
        "object": ["a paperclip","a vinyl record","a typewriter","a ceiling fan","an umbrella"],
        "concept": ["entropy","compound interest","DNA replication","plate tectonics","catalysts"],
        "element1": ["a lost sword","a singing river","a cursed mirror"],
        "element2": ["seven moons","an exiled prince","a forgotten god"],
        "scent": ["fresh bread","wet stone","old paper","jasmine"],
    }
    for i in range(n):
        tmpl = rng.choice(CREATIVE_TEMPLATES)
        kwargs = {k: rng.choice(v) for k, v in fillers.items()}
        try:
            p = tmpl.format(**kwargs)
        except KeyError:
            p = tmpl
        out.append({"id": f"creative_{i:04d}", "category": "creative", "prompt": p, "max_tokens": 64, "stop": ["\n\n"]})
    return out

def gen_multi(rng, n=100):
    out = []
    for i in range(n):
        tmpl = rng.choice(MULTI_TEMPLATES)
        p = tmpl.format(text=rng.choice(MULTI_SENTENCES))
        out.append({"id": f"multi_{i:04d}", "category": "multilingual", "prompt": p, "max_tokens": 48, "stop": ["\n\n"]})
    return out

def gen_repetition(rng, n=100):
    out = []
    for i in range(n):
        tmpl = rng.choice(REPETITION_TEMPLATES)
        try:
            p = tmpl.format(
                word=rng.choice(["echo","banana","yes","ok","ping"]),
                n=rng.randint(5,15),
                seq=", ".join(str(x) for x in [2,4,8,16,32]),
                start=rng.randint(1,10), end=rng.randint(20,40),
                a=rng.choice(["a","c","e","g","m"]), b=rng.choice(["k","p","s","v","z"]),
                phrase=rng.choice(["the rain in Spain","mary had a little lamb","row row row your boat"]),
                num=rng.randint(100,9999),
            )
        except KeyError:
            p = tmpl
        out.append({"id": f"rep_{i:04d}", "category": "repetition", "prompt": p, "max_tokens": 80, "stop": ["\n\n\n"]})
    return out

def gen_edge(rng, n=100):
    out = []
    for i in range(n):
        c = EDGE_CASES[i % len(EDGE_CASES)]
        out.append({"id": f"edge_{i:04d}_{c['tag']}", "category": "edge", "prompt": c["prompt"], "max_tokens": c["max_tokens"], "stop": None})
    return out

# -------------------- main --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).parent / "prompts.jsonl"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    parts = (
        gen_factual(rng, 200)
        + gen_mmlu(rng, 300)
        + gen_reasoning(rng, 200)
        + gen_code(rng, 200)
        + gen_summ(rng, 150)
        + gen_creative(rng, 150)
        + gen_multi(rng, 100)
        + gen_repetition(rng, 100)
        + gen_edge(rng, 100)
    )
    assert len(parts) == 1500, f"expected 1500, got {len(parts)}"
    with open(args.out, "w") as f:
        for p in parts:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    cats = {}
    for p in parts:
        cats[p["category"]] = cats.get(p["category"], 0) + 1
    print(f"wrote {len(parts)} prompts to {args.out}")
    print("breakdown:")
    for c in sorted(cats):
        print(f"  {c:24s} {cats[c]:4d}")

if __name__ == "__main__":
    main()
