# Six-column classifier. Same process as train_v3.py -- frozen e5-large vectors
# + TF-IDF, calibrated LinearSVC, leakage-free GroupKFold -- extended from 2
# targets to 6.
#
# NOTHING IS RE-ENCODED. embeddings_cache_e5large.npz is loaded once and indexed
# by position; training_v4.csv preserves combined_training.csv's row order so the
# alignment holds. The extra heads are linear models over the same feature
# matrix, which is why 6 columns cost barely more than 2.
#
# Labels come from LinkedIn's own posting form (see build_training_v4.py), NOT
# from the regexes in scripts/geo/production/classification.py. That matters: a
# model graded against regex output only proves an SVM can reproduce a regex.
#
#   python build_training_v4.py && python train_v4.py
import csv, os, re, pickle, collections, sys, numpy as np
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, f1_score

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = r"C:\Users\Mokshitha Sree\Downloads\categorization ml model\embeddings_cache_e5large.npz"
EMB = "intfloat/multilingual-e5-large"
CAP = 2                 # variant D: at most 2 rows per description group

# Minimum rows before a class is worth learning, PER COLUMN.
# industry/role stay at 5 to reproduce train_v3.py exactly -- raising the floor
# here would silently drop Agriculture & Primary (20 rows) and ship an industry
# head weaker than the one already in production. The new columns use 30: with
# 5-fold GroupKFold wrapped in CalibratedClassifierCV(cv=3), a class needs
# roughly 15 members before every fold can even see it, and 30 is the margin.
MIN_CLASS = {"industry": 5, "role": 5,
             "job_type": 30, "experience_band": 30, "remote_type": 30}
DEFAULT_MIN_CLASS = 30

csv.field_size_limit(10**7)
rows = list(csv.DictReader(open(os.path.join(HERE, "data", "training_v4.csv"),
                                encoding="utf-8", newline="")))
Xe = np.load(CACHE, allow_pickle=True)["X"].astype(np.float32)
assert len(rows) == len(Xe), f"cache misaligned: {len(rows)} rows vs {len(Xe)} vectors"

def norm(s): return re.sub(r"\s+", " ", str(s or "")).strip().lower()
def dkey(s): return norm(s)[:300]
NON_EN = re.compile(r"[äöüßÄÖÜéèêàçñÉÈÀ]|\b(und|für|mit|der|die|das|pour|avec|les|des|dans)\b", re.I)

# ---- variant D, identical to train_v3: English only, cap duplicate descriptions
# The cap matters more here than it did for industry/role -- LinkedIn postings
# share large blocks of boilerplate, and job_type/experience correlate with that
# boilerplate far more strongly than the job's actual subject does.
seen = collections.Counter(); keep = []
for i, r in enumerate(rows):
    if NON_EN.search(r["title"] + " " + r["description"][:200]):
        continue
    g = dkey(r["description"])
    if seen[g] >= CAP:
        continue
    seen[g] += 1; keep.append(i)
print(f"variant D: {len(keep):,} of {len(rows):,} rows kept ({100*len(keep)/len(rows):.0f}%)\n", flush=True)

texts  = [(rows[i]["title"] + " . " + rows[i]["description"]).strip() for i in keep]
groups = [dkey(rows[i]["description"]) for i in keep]
Xk = Xe[keep]

def find_thr(P, yi, target=0.95):
    """Lowest confidence cut that reaches `target` precision on >=200 rows.

    Returns (threshold, precision, coverage) or (None, None, None) when NO cut
    reaches the target. That None matters: train_v3 returned a 0.30 fallback
    here, and 0.30 silently reads as "trust nearly everything" downstream --
    which would leave the least reliable head flagging the fewest rows for
    review. A head that cannot be gated must say so, not fake a threshold.
    """
    conf = P.max(1); ok = (P.argmax(1) == yi)
    for t in np.round(np.arange(0.20, 0.999, 0.005), 3):
        k = conf >= t
        if k.sum() < 200: break
        if ok[k].mean() >= target:
            return float(t), ok[k].mean()*100, k.mean()*100
    return None, None, None

# ---- targets --------------------------------------------------------------
# collar is deliberately absent. It is a deterministic function of role under
# the product policy (classification.py:182-202) and build_training_v4.py
# derives it at 91.6% coverage with no model. A head here could only add error
# to a lookup, so predict_v4.py derives it from the role head's output instead.
TARGETS = ["industry", "role", "job_type", "experience_band", "remote_type"]

summary = []
for col in TARGETS:
    # Only rows that actually carry a label for this column. Blank means "no
    # ground truth", never a class -- inventing a negative is how the previous
    # numbers got inflated.
    labelled = [j for j, i in enumerate(keep) if rows[i][col].strip()]

    if col == "remote_type":
        # LinkedIn's remote_allowed is 1-or-absent, so the positive class is
        # trustworthy and there is no explicit negative. We can only form a
        # negative from rows that DID join to postings.csv, where a blank means
        # "the employer did not tick the remote box" rather than "unknown".
        # Rows with no job_type never joined, so they are excluded entirely.
        joined = [j for j, i in enumerate(keep) if rows[i]["job_type"].strip()]
        y_all = ["Remote" if rows[keep[j]]["remote_type"].strip() else "Not-remote"
                 for j in joined]
        labelled = joined
        print(f"  note: remote_type negatives are inferred from an unticked "
              f"LinkedIn checkbox, not an explicit On-site value.", flush=True)
    else:
        y_all = [rows[keep[j]][col].strip() for j in labelled]

    if not labelled:
        print(f"\n{'='*70}\n{col.upper()}: SKIPPED -- no labelled rows\n", flush=True)
        summary.append((col, 0, 0, None, None)); continue

    cnt = collections.Counter(y_all)
    floor = MIN_CLASS.get(col, DEFAULT_MIN_CLASS)
    # Report every dropped class out loud. A silently truncated class reads as
    # "covered" when it was not.
    thin = {lab: c for lab, c in cnt.items() if c < floor}
    sel = [j for j, lab in enumerate(y_all) if cnt[lab] >= floor]

    tx = [texts[labelled[j]] for j in sel]
    gk = [groups[labelled[j]] for j in sel]
    Xs = Xk[[labelled[j] for j in sel]]
    ys = [y_all[j] for j in sel]
    le = LabelEncoder().fit(ys); yi = np.array(le.transform(ys))

    print(f"\n{'='*70}")
    print(f"{col.upper()}: {len(yi):,} labelled rows ({100*len(yi)/len(keep):.1f}% of the set), "
          f"{len(le.classes_)} classes")
    for lab, c in cnt.most_common():
        mark = "  DROPPED (< %d)" % floor if lab in thin else ""
        print(f"    {c:>7,}  {100*c/len(y_all):5.1f}%  {lab}{mark}")
    if thin:
        print(f"  !! {sum(thin.values()):,} rows dropped across {len(thin)} thin class(es) "
              f"below the {floor}-row floor: {', '.join(sorted(thin))}")
    sys.stdout.flush()

    # honest grouped-split OOF: no description group spans train and test
    P = np.zeros((len(yi), len(le.classes_)))
    for tr, te in GroupKFold(n_splits=5).split(np.zeros(len(yi)), yi, groups=gk):
        tf = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=3, max_features=40000,
                             sublinear_tf=True, strip_accents="unicode")
        A = hstack([tf.fit_transform([tx[i] for i in tr]), csr_matrix(Xs[tr])]).tocsr()
        B = hstack([tf.transform([tx[i] for i in te]),     csr_matrix(Xs[te])]).tocsr()
        m = CalibratedClassifierCV(LinearSVC(class_weight="balanced", C=2.0, max_iter=3000), cv=3)
        m.fit(A, yi[tr]); P[te] = m.predict_proba(B)

    acc = accuracy_score(yi, P.argmax(1)) * 100
    mf1 = f1_score(yi, P.argmax(1), average="macro") * 100
    # Majority-class baseline. Without it, 80% on a field that is 80% Full-time
    # looks like a result when the model has learned nothing.
    base = 100 * max(collections.Counter(yi).values()) / len(yi)
    thr, a95, cov = find_thr(P, yi)
    print(f"  HONEST (grouped) accuracy={acc:.1f}%   macro-F1={mf1:.1f}%")
    print(f"  majority-class baseline={base:.1f}%   lift={acc-base:+.1f} pts")
    if thr is None:
        print(f"  !! NOT GATABLE: no confidence cut reaches 95% precision.")
        print(f"     This head cannot be shipped as a high-precision subset --")
        print(f"     every prediction carries the full {acc:.1f}% error rate.", flush=True)
    else:
        print(f"  95%-target threshold={thr:.3f} -> {a95:.1f}% acc @ {cov:.1f}% coverage", flush=True)

    tf = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=3, max_features=40000,
                         sublinear_tf=True, strip_accents="unicode")
    F = hstack([tf.fit_transform(tx), csr_matrix(Xs)]).tocsr()
    clf = CalibratedClassifierCV(LinearSVC(class_weight="balanced", C=2.0, max_iter=3000), cv=3)
    clf.fit(F, yi)
    with open(os.path.join(HERE, f"model_{col}_v4.pkl"), "wb") as f:
        pickle.dump({"kind": "svm", "embedding_model": EMB, "emb_prefix": "query: ",
                     "winner": "linsvc C=2.0 (variant D data)", "classifier": clf, "tfidf": tf,
                     "label_encoder": le, "threshold": thr, "gatable": thr is not None,
                     "honest_accuracy": acc, "honest_macro_f1": mf1,
                     "majority_baseline": base, "n_train": int(len(yi)),
                     "label_source": ("linkedin_employer_form"
                                      if col in ("job_type", "experience_band", "remote_type")
                                      else "curated"),
                     "eval": "GroupKFold on description prefix (leakage-free)"}, f)
    print(f"  saved model_{col}_v4.pkl", flush=True)
    summary.append((col, len(yi), len(le.classes_), acc, base))

print(f"\n{'='*70}\nSUMMARY\n{'='*70}")
print(f"{'column':<18}{'rows':>9}{'classes':>9}{'accuracy':>10}{'baseline':>10}{'lift':>8}")
for col, n, k, acc, base in summary:
    if acc is None:
        print(f"{col:<18}{'-':>9}{'-':>9}{'skipped':>10}{'-':>10}{'-':>8}")
    else:
        print(f"{col:<18}{n:>9,}{k:>9}{acc:>9.1f}%{base:>9.1f}%{acc-base:>+7.1f}")
print("\ncollar: derived from role, not trained (91.6% coverage, see build_training_v4.py)")
print("TRAIN V4 DONE")
