# Label provenance

Where every label in `data/training_v4.csv` came from, and the evidence behind
each decision. Reproduce any of it with the scripts in `audit/`.

This document exists because **the previous accuracy numbers for these four
fields were unfalsifiable**, and the reason was always provenance rather than
modelling. It should be the first thing anyone reads before quoting a figure.

---

## Summary

| Column | Source | Labels | Coverage | Independent of the rules? |
|---|---|---|---|---|
| `industry` | curated (existing) | 53,823 | 100% | n/a |
| `role` | curated (existing) | 53,823 | 100% | n/a |
| `job_type` | LinkedIn `formatted_work_type` | 32,714 | 60.8% | **yes** |
| `experience_band` | LinkedIn `formatted_experience_level` | 24,727 | 45.9% | **yes** |
| `remote_type` | LinkedIn `remote_allowed` | 5,175 | 9.6% | **yes** |
| `collar` | derived from `role` | 49,323 | 91.6% | deterministic |

"Independent of the rules" means the label was **not** produced by
`scripts/geo/production/classification.py`. This is the property that makes the
measurement meaningful.

---

## Why the obvious approach was rejected

The default plan was to label the existing 53,823 rows with the regexes in
`classification.py`. Two things kill it, both measured rather than assumed.

### 1. Circularity

Labelling rows with a regex and then training a model to reproduce those labels
measures *"can an SVM reproduce a regex"*. It cannot detect a wrong rule, only
extend a rule's keyword reach. Any accuracy figure produced this way is an
agreement score, not an accuracy score, and must never be reported as the
latter. `scripts/classifier-eval/README.md` documents this same failure for
industry and role.

### 2. The rules cannot label this corpus anyway

`audit/measure_label_coverage.py` applies the real `classification.py` text
branches to all 53,823 rows:

| Field | Rows with a usable label | Class distribution |
|---|---|---|
| `remote_type` | **6.3%** | On-site 47%, Hybrid 39%, Remote 14% |
| `job_type` | **11.0%** | Full-time 41%, Contract 31% |
| `experience_band` | **16.8%** | **Senior 87%**, Entry 6%, Mid-Senior 7% |
| `collar` | **9.2%** | **Blue 100%** |

Three independent failures:

- **`collar` returns a single class.** The text-only path in `_collar` is two
  Blue vetoes (clinical, trades). White is only assigned from a database column
  or a confirmed governed role, neither of which exists in this corpus. A
  one-class dataset cannot train a classifier.
- **`experience_band` is 87% Senior.** The title branch fires on
  senior/lead/principal/director versus junior. Training on it produces a model
  that detects the word "senior".
- **73% of the corpus is ESCO/O\*NET occupation definitions, not job ads.**
  "Sprinkler system fitter" is a dictionary entry. When such a definition
  contains the word "contract", that is not a statement about anyone's
  employment terms, so many of the few labels obtained are actively wrong.

---

## What was used instead

`experiments/build_combined.py:12` shows the training set was built from
`postings.csv`. That file has 31 columns; the build kept four. Among the
discarded 27:

| Column | Fill rate in source | What it is |
|---|---|---|
| `formatted_work_type` | 100% | Full-time / Contract / Part-time / Temporary / Internship / Volunteer / Other |
| `formatted_experience_level` | 76.3% | Internship / Entry level / Associate / Mid-Senior level / Director / Executive |
| `remote_allowed` | 12.3% | `1.0` or absent |

These are filled in by the employer on LinkedIn's posting form. They have no
relationship to Joblet's regexes.

### The join

`audit/probe_postings_join.py` rebuilds `build_combined.py`'s `ct()`/`cd()`
cleaning to reconstruct the key, then joins on `(cleaned title, description
prefix)`.

**32,797 of 53,823 rows (60.9%) join.** The 39% that do not are the ESCO/O\*NET
and other-feed rows, which have no LinkedIn source row to join to.

### Was the join trustworthy?

`postings.csv` has 123,849 rows collapsing to 103,372 unique keys, so 7.6% of
keys cover more than one source row. If those rows disagreed on a label,
picking one would import silent noise — exactly the failure mode being avoided.
`audit/check_join_conflicts.py` measures it:

| Field | Keys with a value | Unanimous | Conflicting (dropped) |
|---|---|---|---|
| `formatted_work_type` | 103,372 | **99.8%** | 226 |
| `formatted_experience_level` | 77,940 | **99.9%** | 96 |
| `remote_allowed` | 12,911 | **100.0%** | 0 |

Conflicting keys are dropped, not resolved. Final conflict-free recovery:
**32,714 job_type**, **24,727 experience**, **5,175 remote**.

---

## Per-field notes

### `job_type` — usable, skewed

```
26,209  80.1%  Full-time      288  0.9%  Internship
 3,666  11.2%  Contract       283  0.9%  Temporary
 1,991   6.1%  Part-time      166  0.5%  Other
                              111  0.3%  Volunteer
```

Internship went from **0 examples under regex labelling to 288**. Still thin.
`Other` and `Volunteer` have no product bucket and map to blank at serving.

### `experience_band` — usable, and the skew is gone

```
12,015  48.6%  Mid-Senior level    1,065  4.3%  Director
 8,147  32.9%  Entry level           549  2.2%  Executive
 2,560  10.4%  Associate             391  1.6%  Internship
```

Compare the regex distribution: **87% Senior**. This is the clearest single
demonstration of why the label source mattered more than the model.

### `remote_type` — positive-only, and that is a hard limit

`remote_allowed` is `1.0` or absent. There is no explicit negative, and no
Hybrid value at all — LinkedIn's schema has no such concept.

The negative class is therefore constructed only from rows that **did** join to
`postings.csv`, where a blank means "the employer did not tick the box" rather
than "unknown". Rows that never joined are excluded entirely.

**This negative is weaker than a true negative and must not be reported as
"On-site".** Measured on `jobs_joveo_partner_v2`, of 21,868 sampled rows with
`remote = false`, **2,077 (9.5%) say "fully remote" or "work from home" in their
own description**. The positive direction is clean: only 5 of 694 `remote =
true` rows contradict. So the flag is trustworthy in one direction only.

### `collar` — derived, never predicted

No source exists. `jobs_joveo_partner_v2` has no collar column at all, and
`work_bucket` is a product bucket (ai / software / experts / others, 84%
"others"), not collar.

But collar does not need a source, because under `classification.py:182-202` it
is already a deterministic function of role:

```
Healthcare Professional     -> Blue    (clinical veto)
Driver & Logistics          -> Blue    (trades veto)
Construction & Real Estate  -> Blue    (trades veto)
21 other roles              -> White
Others                      -> blank   (unknown role cannot imply a collar)
```

Result: **91.6% coverage, White 72.6% / Blue 27.4%**, zero training. The 8.4%
blank is exactly the rows whose role is `Others`.

---

## What the live database contains, and why it was not used

Audited before choosing the LinkedIn route.

| Column | State |
|---|---|
| `experienceLevel` | **empty across the entire table** — 0 non-empty rows |
| `jobType` | 9.1% filled, and **not a labelled field** |
| `remote` | boolean, 3% true, `false` means "not flagged" |
| collar | **column does not exist** |

`jobType` is raw partner-feed passthrough:

- **nine spellings of one concept**: `fulltime` / `FullTime` / `Full Time` /
  `full_time` / `Full-Time` / `Full time` / `FULL TIME` / `FULL_TIME` / `Full-time`
- **its single most common value is a menu**: `"fulltime, contract, temporary"`
  — one feed stuffs the whole enum into every row
- **wrong dimension**: `on-site`, `Hybrid`, `Remote`, `No work mode specified`
- **pay basis**: `Hourly`, `Salary`
- **scheduling**: `Per Diem`, `perdiem`, `PRN`
- **hiring arrangement**: `permanent`, `PERM`, `CONTRACTTOHIRE`

After normalization ~65% survives, giving ~5.9% of the table (~106k rows) — a
viable second source, but those rows carry OpenAI 384-dim vectors rather than
e5-large 1024-dim ones, so using them would require a fresh encode and would
mix two embedding spaces.

`sql/add_four_field_classification_columns.sql` puts normalized values in **new**
columns and enforces every vocabulary with a `CHECK` constraint. `jobType`
degenerated precisely because nothing constrained it.

---

## Rules for quoting these numbers

1. **Never report an accuracy measured against regex-generated labels as
   accuracy.** Call it agreement with rules.
2. **Always quote the majority-class baseline beside the accuracy.** 88.8% on a
   79.8%-Full-time field is +9.1, not 88.8.
3. **Quote macro-F1 alongside accuracy** on skewed fields. A 30-point gap means
   the model serves one class.
4. **These are LinkedIn-distribution numbers.** They are not the live-feed
   numbers, and the live-feed evaluation has not been run.

---

# Addendum — the four new heads, final vocabularies

Retrained to the handover document's label sets (`START_HERE.md` §2).

| Column | Vocabulary trained | Label source | Independent? |
|---|---|---|---|
| `job_type` | Full-time · Contract · Temporary · Part-time · Internship | LinkedIn `formatted_work_type` | **yes** |
| `experience_level` | Entry · Mid · Senior · Executive | LinkedIn `formatted_experience_level` | **yes** |
| `collar` | Blue · White | **derived from `role`** | no |
| `remote_mode` | On-site · Remote · Hybrid | **derived from description text** | no |

## `experience_level` — 6 native levels mapped to the handover's 4

LinkedIn distinguishes Director from Executive natively, so all four classes
have genuine examples:

| Handover band | LinkedIn source | Rows |
|---|---|---|
| Entry | Internship + Entry level | 7,227 |
| Mid | Associate + Mid-Senior level | 12,866 |
| Senior | Director | 957 |
| Executive | Executive | 496 |

`Associate → Mid` is empirically justified, not arbitrary: in the 6-class model,
of 2,190 Associate rows the head sent 1,254 to Mid-Senior and got only 313
right (14% recall). The model had already merged them.

## `job_type` — five of six

`Per diem` has **no LinkedIn category**, so this source contains zero examples
and cannot produce that class at any volume. The live DB has ~2,490 per-diem
rows (`Per Diem` / `perdiem` / `PRN`) and the handover has 72 annotated ones;
both are viable sources, neither is this one.

`Volunteer` (111) and `Other` (166) are LinkedIn artefacts with no product
bucket and were dropped rather than forced into a neighbouring class.

## `remote_mode` — text-derived, and why that is a real limitation

`remote_allowed` is 1-or-null: no Hybrid exists in it, and absence is not
On-site (9.5% of `remote = false` rows in the live table say "fully remote" in
their own description). So the flag alone cannot produce this field.

The handover's annotators worked from job text only, so the label was
reconstructed the same way, using deliberately narrow phrases. A bare "remote"
was excluded because it matches "remote monitoring", "remote site" and "remote
control". A row is labelled only when **exactly one** of the three signals
appears; "hybrid schedule with some on-site days" gets no label.

Result: 7,609 labelled (23.6%), 24,669 left Unknown and never imputed.

**This makes the head's 98.2% uninterpretable as accuracy** — the model reads
the same phrases that generated the label. Worse, the 23.6% it covers are the
postings that state their mode outright, which need a regex rather than a model,
while the 76.4% that actually require inference carry no label at all. See
`TRUST.md`.

## `collar` — derived, and a different target

Produced by mapping role → collar per the policy in `classification.py:182-202`
(Healthcare / Driver / Construction → Blue; 21 other roles → White; `Others` →
blank).

Ours is **27.4% Blue**; the handover's annotator-defined labels are **52.9%
Blue**. A gap that size is not sampling noise — the two systems are predicting
different definitions. Ours answers "what collar does this role bucket imply",
theirs answers "what collar is this job". Do not compare the two numbers.

## Rules for quoting these numbers — unchanged and now more load-bearing

1. Never report an accuracy measured against labels the model can read off the
   text as accuracy. `remote_mode` is the live example.
2. Always quote the majority-class baseline beside the accuracy.
3. Quote macro-F1 alongside accuracy on skewed fields.
4. Quote coverage alongside any precision claim.
5. These are LinkedIn-distribution numbers, not live-feed numbers.
