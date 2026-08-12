# Results

Every figure comes from the logs in `logs/`. Protocol: 5-fold `GroupKFold`
grouped on the normalised first 300 characters of the description, so no
description group spans train and test. Calibrated
`LinearSVC(C=2.0, class_weight="balanced")` over TF-IDF (≤40k features, 1-2
grams, `min_df=3`) concatenated with frozen 1024-dim `multilingual-e5-large`
vectors. TF-IDF is refit inside every fold, never before the split.

Training pool after the variant-D filter (English only, ≤2 rows per description
group): **32,278 of 53,823**.

---

## Headline

| Column | Rows | Cls | Accuracy | Macro-F1 | Kappa | Baseline | Lift | 95% gate |
|---|---|---|---|---|---|---|---|---|
| `role` | 32,278 | 25 | 88.1% | 87.6% | — | 10.5% | +77.6 | 0.690 @ 76.4% |
| `industry` | 32,278 | 15 | 86.3% | 76.3% | — | 22.9% | +63.3 | 0.755 @ 68.1% |
| `job_type` | 28,486 | 5 | 89.1% | 59.4% | 0.624 | 80.3% | +8.8 | 0.820 @ 78.9% |
| `experience_level` | 21,546 | 4 | 79.4% | 70.9% | 0.595 | 59.7% | +19.7 | 0.910 @ 20.0% |
| `collar` ⚠ | 28,968 | 2 | 96.7% | 95.7% | 0.915 | 73.4% | +23.3 | 0.200 @ 100% |
| `remote_mode` ⚠ | 7,609 | 3 | 98.2% | 98.0% | 0.966 | 62.1% | +36.1 | 0.200 @ 100% |

⚠ provisional — see `TRUST.md`.

`industry` and `role` gates were fitted on the same out-of-fold predictions
their accuracy comes from, so their coverage figures are optimistic. The four
new columns use a separate group-disjoint calibration split and are honest.

---

## Per column

### `role` — the reference for what "working" looks like

```
88.1% accuracy · 87.6% macro-F1 · 10.5% baseline · +77.6 lift
```

Accuracy and macro-F1 differ by 0.5 points. Compare every other head against
this.

### `industry`

```
86.3% accuracy · 76.3% macro-F1 · 22.9% baseline · +63.3 lift
```

Reproduces the previous model (86.1 / 77.3). Ten-point gap: strong on common
industries, weak on rare ones. "Agriculture & Primary" has 20 rows, retained
but unreliable.

### `job_type` — highest-looking, second-weakest

```
89.1% accuracy · 59.4% macro-F1 · 0.624 kappa · 80.3% baseline · +8.8 lift

              precision  recall  f1     support
Full-time         0.903   0.974  0.937   22,876
Contract          0.822   0.623  0.709    3,600
Part-time         0.829   0.473  0.602    1,516
Internship        0.587   0.407  0.481      248
Temporary         0.720   0.146  0.243      246
```

**A 29.7-point accuracy/macro-F1 gap.** Temporary recall is **0.146** — the head
finds one in seven. Part-time 0.473, Internship 0.407. It predicts Full-time and
little else, and Full-time is 80.3% of the data.

Shippable only gated: 95% precision at 78.9% coverage.

### `experience_level`

```
79.4% accuracy · 70.9% macro-F1 · 0.595 kappa · 59.7% baseline · +19.7 lift

              precision  recall  f1     support
Mid               0.804   0.882  0.841   12,866
Entry             0.777   0.688  0.730    7,227
Senior            0.773   0.568  0.655      957
Executive         0.753   0.510  0.608      496
```

Everything drifts toward `Mid`, the 59.7% majority. **Entry 0.688 and Senior
0.568 recall** — the two bands users would actually filter on are the weakest,
and Executive is worse at 0.510.

The 4-class handover vocabulary is better supported by our data than the 3-band
alternative: LinkedIn distinguishes Director from Executive natively, so all
four classes have real examples rather than being merged arbitrarily.

Gate is the binding constraint: 95% precision covers only **20.0%** of rows.

### `collar` ⚠ provisional

```
96.7% accuracy · 95.7% macro-F1 · 0.915 kappa · 73.4% baseline · +23.3 lift
Blue  0.947 / 0.927      White 0.974 / 0.981
```

Labels were derived from role by a mapping, so the head partly reproduces that
mapping. Also a different target from the reference implementation's collar —
ours is 27.4% Blue, theirs is 52.9%. Not comparable to their 94.4%.

### `remote_mode` ⚠ provisional — the number to distrust most

```
98.2% accuracy · 98.0% macro-F1 · 0.966 kappa · 62.1% baseline · +36.1 lift

              precision  recall  f1     support
Hybrid            0.995   0.987  0.991    1,269
On-site           0.972   0.953  0.963    1,616
Remote            0.981   0.990  0.986    4,724
```

Labels came from searching descriptions for phrases like `fully remote` and
`hybrid schedule`; the model's TF-IDF features contain those exact phrases. It
learned to detect the phrase that made the label.

Near-perfect precision *and* recall across all three classes — with Hybrid, the
hardest class, scoring highest — is the signature of that circularity rather
than of a strong model.

**Only 23.6% of rows could be labelled at all.** The 24,669 rows left Unknown
are postings that never state their mode, and those are precisely the cases
where a classifier would be worth having. The head has never seen them.

---

## Confidence gates

Fitted on a separate group-disjoint calibration split for the four new columns
(handover §7 — fitting the threshold on the reported out-of-fold predictions
inflates its stated precision; their measurement of the existing heads found 95%
claimed delivering 86.1% industry / 84.0% role).

| Column | Threshold | Precision | Coverage |
|---|---|---|---|
| `collar` ⚠ | 0.200 | 96.9% | 100% |
| `remote_mode` ⚠ | 0.200 | 97.8% | 100% |
| `job_type` | 0.820 | 95.0% | 78.9% |
| `experience_level` | 0.910 | 95.5% | **20.0%** |

The two provisional heads gate at 100% coverage, which is another symptom of
circular labels rather than a strength.

---

## Not measured

1. **Live-feed accuracy.** All of the above is in-distribution for LinkedIn. The
   Joblet feed is heavily gig-driving and healthcare. Expect these to fall.
2. **`collar` and `remote_mode` against independent labels.** The only thing
   that would make them quotable.
3. **`Per diem`** — zero examples in this label source.
4. **`predict_6col.py` end to end** — `sentence_transformers` is not installed
   in the training environment. The head stack was validated against cached
   vectors; only the `encode()` call is unexercised.
