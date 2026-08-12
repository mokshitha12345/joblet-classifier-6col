# Joblet job classifier — six columns

Extends the existing 2-column classifier (industry, role) to **six**, adding
`job_type`, `experience_level`, `collar` and `remote_mode`.

Built on the existing 53,823-row training set and the existing frozen
`multilingual-e5-large` embedding cache. **Nothing was re-encoded and the
encoder was never fine-tuned.**

Label vocabularies match the handover document (`START_HERE.md` §2) so these
heads are directly comparable to the reference implementation.

---

## Read this first

**Four of the six heads are trustworthy. Two are not yet.** The architecture is
sound throughout; what differs is where the labels came from. `TRUST.md` is one
page and explains exactly which numbers you can quote. Read it before putting
any figure in a document.

---

## What predicts what

| Column | Labels | Head |
|---|---|---|
| `industry` | 15 | `model_industry_v4.pkl` |
| `role` | 25 | `model_role_v4.pkl` |
| `job_type` | Full-time · Contract · Temporary · Part-time · Internship | `model_job_type_v5.pkl` |
| `experience_level` | Entry · Mid · Senior · Executive | `model_experience_level_v5.pkl` |
| `collar` | Blue · White | `model_collar_v5.pkl` |
| `remote_mode` | On-site · Remote · Hybrid | `model_remote_mode_v5.pkl` |

**`Per diem` is missing from `job_type`** — five of the handover's six classes.
LinkedIn has no such category, so our label source contains zero examples. Their
72 annotated rows are the only known source.

---

## Results

Leakage-free `GroupKFold` on the description prefix. Thresholds fitted on a
separate group-disjoint calibration split.

| Column | Rows | Cls | Accuracy | Macro-F1 | Kappa | Baseline | Lift |
|---|---|---|---|---|---|---|---|
| `role` | 32,278 | 25 | 88.1% | 87.6% | — | 10.5% | **+77.6** |
| `industry` | 32,278 | 15 | 86.3% | 76.3% | — | 22.9% | **+63.3** |
| `job_type` | 28,486 | 5 | 89.1% | 59.4% | 0.624 | 80.3% | +8.8 |
| `experience_level` | 21,546 | 4 | 79.4% | 70.9% | 0.595 | 59.7% | +19.7 |
| `collar` ⚠ | 28,968 | 2 | 96.7% | 95.7% | 0.915 | 73.4% | +23.3 |
| `remote_mode` ⚠ | 7,609 | 3 | 98.2% | 98.0% | 0.966 | 62.1% | +36.1 |

⚠ = **provisional**, see `TRUST.md`. The two highest scores in this table are
the two you cannot rely on. That is not a coincidence — both were trained on
labels a text model can partly read straight off the page.

**Read the lift column, not the accuracy column.** `job_type` at 89.1% is only
+8.8 above always guessing Full-time. `role` at 88.1% is +77.6 above its
baseline. Accuracy alone ranks these backwards.

---

## Quick start

```bash
pip install numpy scipy scikit-learn sentence-transformers
python code/predict_6col.py "Travel ICU RN" "13-week contract, night shift"
python code/predict_6col.py --csv myjobs.csv     # -> predictions_6col.csv
python code/predict_6col.py                      # demo
```

First run downloads the e5-large encoder (~2.2 GB), once. Everything is offline
after that — no API, no per-row cost.

**Not yet run end to end here**: `sentence_transformers` is not installed in the
environment these models were trained in. The full head stack was validated
against cached vectors, so only the `encode()` call is unexercised.

---

## Retraining

Requires the frozen cache at
`C:\Users\Mokshitha Sree\Downloads\categorization ml model\embeddings_cache_e5large.npz`.

```bash
python code/build_training_v4.py          # rebuild labels (needs postings.csv)
python code/train_v4.py                   # industry + role          ~35 min
python code/train_v5_handover_vocab.py    # job_type, experience, collar  ~40 min
python code/train_remote_mode.py          # remote_mode              ~10 min
```

**None of the training scripts import `sentence_transformers`.** They load the
cache and index it by position. This is why training is minutes rather than the
~7 hours quoted in the original README — that figure is the encode step, skipped
entirely.

---

## Layout

```
joblet-classifier-6col/
├── README.md                this file
├── TRUST.md                 which numbers to quote — read before writing any doc
├── RESULTS.md               full metrics, per-class detail, caveats
├── LABEL_PROVENANCE.md      where every label came from, with evidence
├── code/
│   ├── predict_6col.py           all six columns, one shared encode
│   ├── build_training_v4.py      recovers labels from postings.csv
│   ├── train_v4.py               industry + role
│   ├── train_v5_handover_vocab.py  job_type + experience_level + collar
│   ├── train_remote_mode.py      remote_mode
│   └── guardrails.py             existing rule layer (unchanged)
├── data/training_v4.csv     53,823 rows, cache-aligned — DO NOT REORDER
├── models/                  the six production heads
│   └── superseded/          older heads with pre-handover vocabularies
├── audit/                   re-runnable evidence for every claim
├── sql/                     DB migration — NOT YET APPLIED
└── logs/                    raw output of every training run
```

---

## Design decisions

**One encode, six heads.** The encoder is the only expensive component. It runs
once per job and all six heads read the same vector, which is why columns 3-6
are nearly free. The existing `predict.py` calls `encode()` *inside* the head
loop — with six fields that becomes six transformer passes per job. Fixing that
is the precondition for everything here (handover §3).

**Frozen embedding, trainable heads.** No fine-tuning. Only linear heads are
fitted, which is why the cached vectors stay valid indefinitely.

**Independent heads, not one joint model.** Each trains on its own row subset —
32,278 for role, 7,609 for remote_mode. A joint model would have to impute the
gaps. Adding a seventh column later touches nothing existing.

**Row order in `training_v4.csv` is load-bearing.** The frozen cache is indexed
by position. Every training script asserts `len(rows) == len(Xe)` so a reordered
file dies loudly instead of training on misaligned vectors. Never sort or filter
that file in place.

**Per-column class floors.** `industry`/`role` use a 5-row floor to reproduce
the previous model exactly — a higher floor silently drops "Agriculture &
Primary" (20 rows). The new columns use 30, because 5-fold `GroupKFold` inside
`CalibratedClassifierCV(cv=3)` needs ~15 members before every fold sees a class.
Every dropped class is printed, never silently truncated.

**Unknown is never imputed.** A row with no label for a field is excluded from
that field's training and scoring. It is not a class and not a guess.

---

## Known limitations

1. **`collar` and `remote_mode` are provisional.** Their labels were generated
   from a role mapping and from text patterns respectively, so both heads partly
   learn to reproduce their own labelling rule. See `TRUST.md`.
2. **`Per diem` is untrainable here** — zero examples in the label source.
3. **`job_type` predicts Full-time and little else.** Macro-F1 59.4% against
   89.1% accuracy. Minority recall: Temporary 0.146, Internship 0.407,
   Part-time 0.473.
4. **`experience_level` gates at only ~20% coverage.** A trustworthy seniority
   label exists for about one job in five.
5. **These are LinkedIn-distribution numbers.** The rows skew white-collar and
   professional; the live Joblet feed is heavily gig-driving and healthcare.
   **The live-feed evaluation has not been run.** Expect these to fall.
6. **The migration is unapplied**, and backfilling values afterwards must be
   chunked through pg_cron — direct bulk UPDATEs on that table roll back
   silently.

---

## What would improve this most

Obtaining `data/labelled_jobs.csv` from the handover package. Its 3,073 rows are
annotator-labelled on live-feed postings, which fixes three problems at once:
it supplies the `Per diem` and Hybrid/On-site classes our source lacks, it gives
an **independent answer key** that turns `collar` and `remote_mode` from
provisional into real, and it is the correct distribution for the live-feed
evaluation that limitation 5 describes.
