"""Can we recover real job_type / remote / experience labels for the existing
53,823-row training set by joining it back to its LinkedIn source file?

build_combined.py cleaned the title and truncated the description to 500 chars
before writing combined_training.csv, so we rebuild the same keys here and join
on them.
"""
import csv, re, collections, os

csv.field_size_limit(10**7)
POSTINGS = r"C:\Users\Mokshitha Sree\Downloads\postings.csv\postings.csv"
TRAIN    = r"C:\Users\Mokshitha Sree\Downloads\joblet\joblet-job-classifier\data\combined_training.csv"

# --- key construction, mirroring build_combined.py's ct()/cd() -----------------
def ct(t):
    t = re.sub(r"\s*\((?:[\d,\s#/-]+)\)\s*$", "", str(t or ""))
    t = re.sub(r"\s*[-#]\s*\d{3,}\s*$", "", t)
    return re.sub(r"\s+", " ", t).strip()

def cd(d):
    d = re.sub(r"<[^>]+>", " ", str(d or ""))
    d = re.sub(r"&[a-z#0-9]+;", " ", d)
    return re.sub(r"\s+", " ", d).strip()

def key(title, desc):
    return (ct(title).casefold(), cd(desc)[:180].casefold())

# --- 1. index the source ------------------------------------------------------
print("scanning postings.csv ...", flush=True)
src = {}
wt   = collections.Counter()
exp  = collections.Counter()
rem  = collections.Counter()
n = 0
with open(POSTINGS, encoding="utf-8", newline="") as f:
    for row in csv.DictReader(f):
        n += 1
        w = (row.get("formatted_work_type") or "").strip()
        e = (row.get("formatted_experience_level") or "").strip()
        r = (row.get("remote_allowed") or "").strip()
        wt[w or "(empty)"] += 1
        exp[e or "(empty)"] += 1
        rem[r or "(empty)"] += 1
        src[key(row.get("title"), row.get("description"))] = (w, e, r)

print(f"postings rows: {n:,}   unique keys: {len(src):,}\n")

def show(name, c, total):
    print(f"  {name}")
    filled = total - c.get("(empty)", 0)
    print(f"    filled: {filled:,} ({100*filled/total:.1f}%)")
    for k, v in c.most_common(12):
        print(f"      {v:8,}  {100*v/total:5.1f}%  {k}")
    print()

print("=" * 70)
print("SOURCE COLUMN COVERAGE (all of postings.csv)")
print("=" * 70)
show("formatted_work_type", wt, n)
show("formatted_experience_level", exp, n)
show("remote_allowed", rem, n)

# --- 2. join the training set back -------------------------------------------
print("=" * 70)
print("JOIN AGAINST combined_training.csv")
print("=" * 70)
rows = list(csv.DictReader(open(TRAIN, encoding="utf-8", newline="")))
hit = 0
got = collections.Counter()
jt, je, jr = collections.Counter(), collections.Counter(), collections.Counter()
for r in rows:
    k = key(r["title"], r["description"])
    v = src.get(k)
    if v is None:
        continue
    hit += 1
    w, e, rm = v
    if w:  jt[w] += 1
    if e:  je[e] += 1
    if rm: jr[rm] += 1
    got["work_type" if w else "-"] += 0

print(f"training rows          : {len(rows):,}")
print(f"joined to postings.csv : {hit:,} ({100*hit/len(rows):.1f}%)\n")

def show2(name, c):
    tot = sum(c.values())
    print(f"  {name}: {tot:,} labelled ({100*tot/len(rows):.1f}% of training set)")
    for k, v in c.most_common(12):
        print(f"      {v:7,}  {100*v/max(tot,1):5.1f}%  {k}")
    print()

show2("formatted_work_type", jt)
show2("formatted_experience_level", je)
show2("remote_allowed", jr)
