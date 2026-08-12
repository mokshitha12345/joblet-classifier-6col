"""Measure how much of the 4 new fields the existing rules can actually label
on the joblet-job-classifier training corpus.

The corpus has NO structured DB columns (title/description/industry/role only),
so every rule falls through to its text branch. That is the honest ceiling for
rule-labelling this dataset.

Rule logic is copied verbatim from scripts/geo/production/classification.py
(text-fallback branches only) so we measure the real rules, not a paraphrase.
"""
import csv, re, sys, collections, json

csv.field_size_limit(10**7)

PATH = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\Mokshitha Sree\Downloads\joblet\joblet-job-classifier\data\combined_training.csv"

# ---- verbatim from classification.py ----------------------------------------
_BLUE_RE = re.compile(
    r"\b(driver|warehouse|forklift|welder|plumber|electrician|mechanic|laborer|labourer|"
    r"construction worker|carpenter|roofer|janitor|cleaner|security guard|assembler)\b",
    re.I,
)
_CLINICAL_RE = re.compile(
    r"\b(nurse|physician|surgeon|therapist|pharmacist|medical assistant|dental hygienist|"
    r"caregiver|clinical technician|patient care)\b",
    re.I,
)

def unique_signal(values):
    """CONFIRMED if exactly one distinct value, CONFLICT if >1, UNKNOWN if none."""
    unique = sorted({v for v, _ in values})
    if len(unique) == 1:
        return unique[0]
    if len(unique) > 1:
        return "__CONFLICT__"
    return "__UNKNOWN__"

def rule_remote(title, desc):
    text = f"{title}\n{desc}"
    hits = []
    for value, pattern in (
        ("Remote", r"\b(fully remote|100% remote|work from home|remote position)\b"),
        ("Hybrid", r"\bhybrid\b"),
        ("On-site", r"\b(on[- ]site|onsite|in office|office[- ]based)\b"),
    ):
        if re.search(pattern, text, re.I):
            hits.append((value, pattern))
    return unique_signal(hits)

def rule_job_type(title, desc):
    text = f"{title}\n{desc}"
    hits = []
    for value, pattern in (
        ("Full-time", r"\bfull[- ]time\b"),
        ("Part-time", r"\bpart[- ]time\b"),
        ("Contract", r"\b(contract|contractor)\b"),
        ("Internship", r"\b(intern|internship)\b"),
        ("Temporary", r"\b(temporary|temp role)\b"),
    ):
        if re.search(pattern, text, re.I):
            hits.append((value, pattern))
    return unique_signal(hits)

def rule_experience(title, desc):
    # no DB experience_level column here -> straight to the years/title branches
    text = f"{title}\n{desc}"
    stated_years = []
    for match in re.finditer(r"(?i)\b(\d{1,2})\s*(?:\+|to\s*\d{1,2})?\s*years?\b", text):
        prefix = text[max(0, match.start() - 35): match.start()].casefold()
        if not any(t in prefix for t in ("founded", "in business", "serving clients", "company history")):
            stated_years.append(int(match.group(1)))
    if stated_years:
        m = min(stated_years)
        return "Entry" if m <= 1 else "Mid-Senior" if m <= 4 else "Senior"
    hits = []
    if re.search(r"(?i)\b(junior|jr\.?|entry[- ]level)\b", title):
        hits.append(("Entry", "t"))
    if re.search(r"(?i)\b(senior|sr\.?|lead|principal|director)\b", title):
        hits.append(("Senior", "t"))
    return unique_signal(hits)

def rule_collar(title):
    """Only the text branches: clinical veto, then trade veto. No DB column,
    and no governed role field, so everything else is UNKNOWN."""
    if _CLINICAL_RE.search(title):
        return "Blue"
    if _BLUE_RE.search(title):
        return "Blue"
    return "__UNKNOWN__"

# ---- source heuristic --------------------------------------------------------
# ESCO/O*NET rows are canonical occupation definitions: short, lowercase titles,
# third-person generic descriptions. LinkedIn/feed rows are real ads.
def looks_like_ad(title, desc):
    if len(desc) > 900:
        return True
    if re.search(r"\b(we are|you will|apply|benefits|401k|salary|equal opportunity|join our)\b", desc, re.I):
        return True
    return False

rows = list(csv.DictReader(open(PATH, encoding="utf-8", newline="")))
print(f"file: {PATH}")
print(f"rows: {len(rows):,}\n")

fields = {
    "remote_type": rule_remote,
    "job_type": rule_job_type,
    "experience_band": rule_experience,
    "collar": rule_collar,
}

dist = {k: collections.Counter() for k in fields}
dist_ad = {k: collections.Counter() for k in fields}
dist_esco = {k: collections.Counter() for k in fields}
n_ad = 0

for r in rows:
    t, d = r.get("title", ""), r.get("description", "")
    is_ad = looks_like_ad(t, d)
    n_ad += is_ad
    for name, fn in fields.items():
        v = fn(t) if name == "collar" else fn(t, d)
        dist[name][v] += 1
        (dist_ad if is_ad else dist_esco)[name][v] += 1

print(f"looks like a real job ad : {n_ad:,} ({100*n_ad/len(rows):.1f}%)")
print(f"looks like ESCO/O*NET def: {len(rows)-n_ad:,} ({100*(len(rows)-n_ad)/len(rows):.1f}%)\n")

def report(title, counter_map, total):
    print("=" * 74)
    print(title, f"(n={total:,})")
    print("=" * 74)
    for name in fields:
        c = counter_map[name]
        unk = c["__UNKNOWN__"]
        conf = c["__CONFLICT__"]
        labelled = total - unk - conf
        print(f"\n  {name}")
        print(f"    usable label : {labelled:6,}  ({100*labelled/total:5.1f}%)")
        print(f"    UNKNOWN      : {unk:6,}  ({100*unk/total:5.1f}%)")
        print(f"    CONFLICT     : {conf:6,}  ({100*conf/total:5.1f}%)")
        real = {k: v for k, v in c.items() if not k.startswith("__")}
        if real:
            parts = ", ".join(f"{k}={v:,} ({100*v/max(labelled,1):.0f}%)"
                              for k, v in sorted(real.items(), key=lambda x: -x[1]))
            print(f"    classes      : {parts}")

report("ALL ROWS", dist, len(rows))
report("REAL-AD ROWS ONLY", dist_ad, max(n_ad, 1))
report("ESCO / O*NET ROWS ONLY", dist_esco, max(len(rows) - n_ad, 1))
