"""How trustworthy is the postings.csv join?

123,849 source rows collapse to 103,372 unique (title, desc-prefix) keys. Where
several rows share a key, do they AGREE on the label? A key whose rows disagree
cannot be used -- picking one would import silent noise, which is the exact
failure mode we are trying not to repeat.
"""
import csv, re, collections

csv.field_size_limit(10**7)
POSTINGS = r"C:\Users\Mokshitha Sree\Downloads\postings.csv\postings.csv"
TRAIN    = r"C:\Users\Mokshitha Sree\Downloads\joblet\joblet-job-classifier\data\combined_training.csv"

def ct(t):
    t = re.sub(r"\s*\((?:[\d,\s#/-]+)\)\s*$", "", str(t or ""))
    t = re.sub(r"\s*[-#]\s*\d{3,}\s*$", "", t)
    return re.sub(r"\s+", " ", t).strip()

def cd(d):
    d = re.sub(r"<[^>]+>", " ", str(d or ""))
    d = re.sub(r"&[a-z#0-9]+;", " ", d)
    return re.sub(r"\s+", " ", d).strip()

def key(t, d):
    return (ct(t).casefold(), cd(d)[:180].casefold())

# key -> set of observed values, per field
obs = {f: collections.defaultdict(set) for f in ("wt", "exp", "rem")}
rows_per_key = collections.Counter()

with open(POSTINGS, encoding="utf-8", newline="") as f:
    for row in csv.DictReader(f):
        k = key(row.get("title"), row.get("description"))
        rows_per_key[k] += 1
        for field, col in (("wt", "formatted_work_type"),
                           ("exp", "formatted_experience_level"),
                           ("rem", "remote_allowed")):
            v = (row.get(col) or "").strip()
            if v:
                obs[field][k].add(v)

multi = sum(1 for k, c in rows_per_key.items() if c > 1)
print(f"unique keys           : {len(rows_per_key):,}")
print(f"keys with >1 row      : {multi:,} ({100*multi/len(rows_per_key):.1f}%)\n")

names = {"wt": "formatted_work_type",
         "exp": "formatted_experience_level",
         "rem": "remote_allowed"}
clean_keys = {}
for field in obs:
    d = obs[field]
    agree    = {k for k, vs in d.items() if len(vs) == 1}
    conflict = {k for k, vs in d.items() if len(vs) > 1}
    clean_keys[field] = {k: next(iter(vs)) for k, vs in d.items() if len(vs) == 1}
    tot = len(d)
    print(f"{names[field]}")
    print(f"  keys with a value   : {tot:,}")
    print(f"  unanimous           : {len(agree):,} ({100*len(agree)/tot:.1f}%)")
    print(f"  CONFLICTING         : {len(conflict):,} ({100*len(conflict)/tot:.1f}%)  <- dropped")
    print()

# now re-join the training set using ONLY unanimous keys
rows = list(csv.DictReader(open(TRAIN, encoding="utf-8", newline="")))
kept = {f: collections.Counter() for f in obs}
for r in rows:
    k = key(r["title"], r["description"])
    for field in obs:
        v = clean_keys[field].get(k)
        if v:
            kept[field][v] += 1

print("=" * 66)
print("CONFLICT-FREE LABELS RECOVERED FOR combined_training.csv")
print(f"(training rows: {len(rows):,})")
print("=" * 66)
for field in ("wt", "exp", "rem"):
    tot = sum(kept[field].values())
    print(f"\n  {names[field]}: {tot:,}  ({100*tot/len(rows):.1f}% of training set)")
    for v, c in kept[field].most_common():
        print(f"      {c:7,}  {100*c/max(tot,1):5.1f}%  {v}")
