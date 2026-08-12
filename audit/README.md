# Audit scripts

The evidence behind every claim in `../README.md` and `../LABEL_PROVENANCE.md`.
Each script is standalone and re-runnable, so no assertion in this deliverable
has to be taken on trust.

Run order matches the order the questions were actually asked.

---

### 1. `measure_label_coverage.py` — *could the existing rules label this dataset?*

Applies the real text branches of `scripts/geo/production/classification.py`
(copied verbatim, not paraphrased) to all 53,823 training rows and reports
coverage and class distribution per field. Also splits the corpus into rows that
look like real job ads versus ESCO/O\*NET occupation definitions.

```bash
python measure_label_coverage.py [path/to/combined_training.csv]
```

**Answer: no.** 6.3% / 11.0% / 16.8% / 9.2% coverage, `collar` comes back
100% Blue (single class, untrainable), `experience_band` comes back 87% Senior,
and 73% of the corpus is dictionary definitions rather than postings.

This is the script that killed the obvious plan and forced the search for a
real label source.

---

### 2. `probe_postings_join.py` — *do the labels exist anywhere?*

Scans the LinkedIn source file `postings.csv` for its structured employment
columns, reports their fill rates, then reconstructs `build_combined.py`'s
`ct()`/`cd()` cleaning to join the training set back to it.

```bash
python probe_postings_join.py
```

**Answer: yes.** `formatted_work_type` 100% filled, `formatted_experience_level`
76.3%, `remote_allowed` 12.3%. **32,797 of 53,823 training rows (60.9%) join.**

---

### 3. `check_join_conflicts.py` — *are the recovered labels trustworthy?*

`postings.csv` has 123,849 rows collapsing to 103,372 unique join keys, so 7.6%
of keys cover more than one source row. This measures whether those rows agree,
and drops any key whose rows disagree rather than arbitrarily picking one.

```bash
python check_join_conflicts.py
```

**Answer: yes.** 99.8% / 99.9% / 100.0% unanimous. Only 226, 96 and 0
conflicting keys, all dropped. Final conflict-free recovery: 32,714 job_type,
24,727 experience, 5,175 remote.

This step matters more than its size suggests. Silently resolving a conflicting
key is how label noise gets imported without anyone noticing.

---

### 4. `measure_experience_3band.py` — *what is the real experience number?*

`train_v4.py` reports 72.4% on LinkedIn's native 6 levels, but the product enum
has 3 bands. Collapsing 6→3 merges Associate↔Mid-Senior and Director↔Executive,
which are two of the model's most frequent confusions, so the 3-band figure is
mechanically higher and is the one that belongs in a handover.

Runs both vocabularies through the identical leakage-free `GroupKFold`
protocol and prints accuracy, macro-F1, majority baseline, 95% gate
achievability and a full confusion matrix for each.

```bash
python measure_experience_3band.py        # ~15-20 min, needs the e5 cache
```

Output: `../logs/experience_3band.log`

**Compare lift, not accuracy.** Accuracy always rises when classes merge,
because merging removes errors without the model getting better. If accuracy
jumps but lift does not, the gain is bookkeeping rather than capability.

---

## Not covered here

The live-database audit (`jobType` being nine spellings plus multi-value menus,
`experienceLevel` being empty table-wide, `remote = false` meaning "not
flagged") was run as ad-hoc SQL against `jobs_joveo_partner_v2` rather than as a
committed script. Findings and the exact figures are written up in
`../LABEL_PROVENANCE.md` under "What the live database contains". Re-running
them needs read access to project `tfbjiyknoagcxjzeoqzw`.
