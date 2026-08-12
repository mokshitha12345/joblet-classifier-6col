# Recover the four extra label columns for the existing training set.
#
# THE POINT: combined_training.csv was built from LinkedIn's postings.csv, but
# build_combined.py kept only (title, description, industry, role) and discarded
# the structured employment columns that were sitting right there. This joins
# them back on.
#
# Why these labels and not regex labels: formatted_work_type and
# formatted_experience_level are filled in by the employer on LinkedIn's own
# posting form. They are completely independent of the regexes in
# scripts/geo/production/classification.py, so a model measured against them is
# not being graded on an answer key it helped write. Regex-derived labels only
# ever measure "can an SVM reproduce a regex".
#
# Row order is preserved EXACTLY so the frozen e5 cache
# (embeddings_cache_e5large.npz) stays aligned -- train_v4.py indexes it by
# position. Do not sort, filter or reorder rows in this file.
#
#   python build_training_v4.py        ->  data/training_v4.csv
import csv, re, os, collections

csv.field_size_limit(10**7)
HERE = os.path.dirname(os.path.abspath(__file__))
POSTINGS = r"C:\Users\Mokshitha Sree\Downloads\postings.csv\postings.csv"
TRAIN = os.path.join(HERE, "data", "combined_training.csv")
OUT = os.path.join(HERE, "data", "training_v4.csv")

# --- join key: mirrors build_combined.py's ct()/cd() so the cleaned training
# --- titles/descriptions line up with the raw source rows again.
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

# --- collar is NOT learned -------------------------------------------------
# Under the product policy in scripts/geo/production/classification.py:182-202,
# collar is already a deterministic function of role: a confirmed role is White,
# except clinical/patient-care and trades which are vetoed to Blue. Training a
# classifier to reproduce a lookup table is the same distillation trap in a new
# costume, so we derive it instead and get 100% coverage for free.
# "Others" stays blank on purpose: an unknown role cannot imply a collar.
COLLAR_FROM_ROLE = {
    "Healthcare Professional":    "Blue",   # clinical veto
    "Driver & Logistics":         "Blue",   # trades veto
    "Construction & Real Estate": "Blue",   # trades veto
}
WHITE_ROLES = {
    "Software Developer", "Data Analyst", "Data Scientist", "Data Engineer",
    "Machine Learning Engineer", "DevOps Engineer", "IT Specialist", "QA Engineer",
    "Cybersecurity Specialist", "Product Manager", "Designer", "Marketing Specialist",
    "Sales Representative", "HR Specialist", "Accountant & Finance", "Consultant",
    "Project Manager", "Teacher & Educator", "Executive",
    "Customer Service Representative", "Legal Professional",
}

def collar_for(role):
    if role in COLLAR_FROM_ROLE:
        return COLLAR_FROM_ROLE[role]
    if role in WHITE_ROLES:
        return "White"
    return ""            # "Others" -> unknown, never guessed

# --- 1. index postings.csv, keeping only keys whose rows AGREE -------------
# 7.6% of keys cover more than one source row. Where those rows disagree on a
# label the key is dropped rather than arbitrarily resolved -- silently picking
# one is how label noise gets imported.
print("indexing postings.csv ...", flush=True)
obs = {f: collections.defaultdict(set) for f in ("wt", "exp", "rem")}
COLS = {"wt": "formatted_work_type",
        "exp": "formatted_experience_level",
        "rem": "remote_allowed"}
with open(POSTINGS, encoding="utf-8", newline="") as f:
    for row in csv.DictReader(f):
        k = key(row.get("title"), row.get("description"))
        for field, col in COLS.items():
            v = (row.get(col) or "").strip()
            if v:
                obs[field][k].add(v)

lookup, dropped = {}, {}
for field, d in obs.items():
    lookup[field] = {k: next(iter(vs)) for k, vs in d.items() if len(vs) == 1}
    dropped[field] = sum(1 for vs in d.values() if len(vs) > 1)
    print(f"  {COLS[field]}: {len(lookup[field]):,} unanimous keys, "
          f"{dropped[field]:,} conflicting dropped", flush=True)

# --- 2. write the augmented training file ---------------------------------
# LinkedIn's NATIVE vocabularies are kept (6 experience levels, 7 work types)
# rather than pre-collapsed to the 3-band / 5-type product enums. Mapping down
# at serving time is a config change; mapping down here would bake one
# judgement call into the weights and force a retrain to revise it. It also
# keeps Associate -- which genuinely sits between Entry and Mid-Senior -- from
# being silently assigned.
REMOTE_FROM_FLAG = {"1.0": "Remote", "1": "Remote"}

rows = list(csv.DictReader(open(TRAIN, encoding="utf-8", newline="")))
cov = collections.Counter()
dist = {c: collections.Counter() for c in ("job_type", "experience_band", "remote_type", "collar")}

with open(OUT, "w", encoding="utf-8", newline="") as f:
    w = csv.writer(f)
    w.writerow(["title", "description", "industry", "role",
                "job_type", "experience_band", "remote_type", "collar"])
    for r in rows:
        k = key(r["title"], r["description"])
        job_type = lookup["wt"].get(k, "")
        experience = lookup["exp"].get(k, "")
        # remote_allowed is 1-or-absent. Absent means "not flagged", NOT
        # "on-site" -- 9.5% of unflagged rows in the live table say "fully
        # remote" in their own text. Writing On-site here would teach the model
        # that one in ten genuine remote jobs is on-site.
        remote = REMOTE_FROM_FLAG.get(lookup["rem"].get(k, ""), "")
        collar = collar_for(r["role"])

        for name, v in (("job_type", job_type), ("experience_band", experience),
                        ("remote_type", remote), ("collar", collar)):
            if v:
                cov[name] += 1
                dist[name][v] += 1
        w.writerow([r["title"], r["description"], r["industry"], r["role"],
                    job_type, experience, remote, collar])

n = len(rows)
print(f"\nwrote {OUT}  ({n:,} rows, order preserved for the e5 cache)\n")
print(f"{'column':<18}{'labelled':>10}{'coverage':>11}")
print("-" * 39)
for c in ("industry", "role", "job_type", "experience_band", "remote_type", "collar"):
    got = n if c in ("industry", "role") else cov[c]
    print(f"{c:<18}{got:>10,}{100*got/n:>10.1f}%")
for c in ("job_type", "experience_band", "remote_type", "collar"):
    print(f"\n  {c}")
    tot = max(cov[c], 1)
    for v, k2 in dist[c].most_common():
        print(f"    {k2:>7,}  {100*k2/tot:5.1f}%  {v}")
