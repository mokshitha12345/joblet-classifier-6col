# Which numbers you can quote

One page. Read it before putting any figure from this package into a document,
a slide, or a handover.

---

## The short version

| Column | Quote it? | Number | Why |
|---|---|---|---|
| `role` | **yes** | 88.1%, +77.6 lift | curated labels, huge lift, macro-F1 tracks accuracy |
| `industry` | **yes** | 86.3%, +63.3 lift | curated labels |
| `experience_level` | **yes, with the lift** | 79.4%, κ 0.595 | employer-supplied labels |
| `job_type` | **yes, with the lift** | 89.1%, **+8.8 only** | employer-supplied labels |
| `collar` | **no — provisional** | 96.7%, κ 0.915 | labels derived from role |
| `remote_mode` | **no — provisional** | 98.2%, κ 0.966 | labels derived from text patterns |

**The two highest scores are the two you cannot use.** That is the whole point
of this page.

---

## Why `collar` and `remote_mode` are provisional

Both were trained on labels that a text model can partly read straight off the
page, so their scores measure *label reproduction* rather than judgement.

### `remote_mode` — 98.2%, and it means nothing

The labels were made by searching descriptions for phrases like
`fully remote`, `hybrid schedule`, `on-site`. The model's TF-IDF features
include those exact phrases. So it learned to detect the phrase that created the
label.

The tell is the shape of the result: near-perfect precision *and* recall on all
three classes, including Hybrid at 0.995 — which should be the hardest class,
not the easiest.

**The deeper problem is which rows got labelled at all.** Only **23.6%** — the
postings that state their work mode outright. Those need a regex, not a model.
The **24,669 rows left Unknown** are the silent ones, and those are the only
cases where a classifier would add value. The head was never trained or tested
on them.

**Worth raising with the reference implementation.** This 98.2% matches the
handover's headline remote-mode figure to the decimal, reached through a
mechanism known to be circular. Their annotators read full descriptions and
could infer beyond explicit phrases ("must be located in Dallas" → On-site), so
theirs is probably better. But one check settles it: **what share of their 2,328
remote-mode labels came from postings that state the mode explicitly?** If it is
most of them, both numbers are measuring the same easy subset.

### `collar` — 96.7%, partly circular

The labels were produced by mapping `role` → collar (Healthcare / Driver /
Construction → Blue, 21 other roles → White). The head therefore partly learns
to reproduce that mapping rather than to judge collar.

It is also **not the same target** as the reference implementation's collar.
Ours is 27.4% Blue; their annotator-defined labels are 52.9% Blue. That gap is
far too large to be sampling noise, so the two systems are predicting different
definitions of the word. **Do not put 96.7% next to their 94.4%** — they are not
comparable.

---

## How to quote the four that are real

**Always give the majority-class baseline beside the accuracy.** Every one of
these fields is skewed and two are severely so.

> `job_type` reaches 89.1% accuracy against an 80.3% majority-class baseline —
> a **+8.8 point** gain. Macro-F1 is 59.4%, so the head predicts Full-time well
> and the minority classes poorly (Temporary recall 0.146).

Not:

> ~~`job_type` is 89% accurate.~~

**Give macro-F1 too on skewed fields.** A large accuracy/macro-F1 gap means the
head serves one class:

| Column | Gap | Meaning |
|---|---|---|
| `role` | 0.5 pts | works across all classes |
| `industry` | 10.0 pts | weaker on rare industries |
| `experience_level` | 8.5 pts | consistent, weak overall |
| `job_type` | **29.7 pts** | Full-time only |

**Give coverage with any precision claim.** These thresholds were fitted on a
separate group-disjoint calibration split, so they are honest — but coverage
varies enormously:

| Column | 95% precision at | Coverage |
|---|---|---|
| `collar` | 0.200 | 100% |
| `job_type` | 0.820 | 78.9% |
| `experience_level` | 0.910 | **20.0%** |

A trustworthy seniority label exists for about **one job in five**.

---

## The caveat that applies to all six

**Every number here was measured on LinkedIn-distribution rows** — white-collar,
professional, employer-form-filled. The live Joblet feed is heavily gig-driving
and healthcare, and carries far more taxonomy-gap jobs.

This is the same trap `scripts/classifier-eval/README.md` documents for the
existing columns, where cross-validation said 90.9% and hand-judged real jobs
came in around 80%. **Expect these figures to fall on the live feed.** That
evaluation has not been run.

---

## What would make all six quotable

`data/labelled_jobs.csv` from the handover package — 3,073 live-feed postings
labelled by annotators who never saw our role mapping or our text patterns.

Grading these six heads against it would:

1. turn `collar` and `remote_mode` from provisional into measured,
2. supply the `Per diem`, Hybrid and On-site classes our source lacks,
3. give the correct distribution for the live-feed evaluation.

Until then, quote four columns and mark two as provisional.
