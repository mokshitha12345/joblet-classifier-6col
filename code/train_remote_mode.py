# The fourth classifier: remote_mode  ->  On-site | Remote | Hybrid
#
# WHY THIS EXISTS AFTER I CALLED IT IMPOSSIBLE:
# I was only looking at LinkedIn's `remote_allowed` flag, which is 1-or-null and
# therefore has no Hybrid and no true On-site. But the handover's annotators did
# not use a flag -- START_HERE.md section 2 says they "worked from job text only".
# We have the same job text. So the label is recoverable the same way.
#
# LABEL PROVENANCE -- WEAKER THAN THE OTHER THREE, SAY SO OUT LOUD:
# job_type and experience_level use employer form-fills. These labels are
# derived from explicit statements in the description, so they are text-derived.
# That is fine as TRAINING signal but it means the honest evaluation has to come
# from an independent key (their 3,073 rows), not from these labels. Training on
# pattern-derived labels and grading against the same patterns would be the
# circularity this whole project has been trying to avoid.
#
# PRECISION RULES:
#   * a row is labelled only when EXACTLY ONE of the three signals appears.
#     "hybrid schedule, some on-site days" is ambiguous -> no label, not a guess.
#   * `remote_allowed = 1` promotes a row to Remote only when no conflicting
#     on-site or hybrid phrase is present.
#   * everything else stays unlabelled. Unknown is a real answer.
import csv, os, re, pickle, collections, sys, numpy as np
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, classification_report

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CACHE = r"C:\Users\Mokshitha Sree\Downloads\categorization ml model\embeddings_cache_e5large.npz"
EMB = "intfloat/multilingual-e5-large"
CAP, MIN_CLASS, CALIB_FRACTION = 2, 30, 0.20

# High-precision phrases only. Deliberately narrow: a bare "remote" matches
# "remote monitoring", "remote site", "remote control" and would poison the set.
REMOTE_RE = re.compile(
    r"\b(fully[- ]remote|100%\s*remote|work from home|remote position|remote role|"
    r"telecommute|telecommuting|work remotely|remote[- ]first)\b", re.I)
HYBRID_RE = re.compile(
    r"\b(hybrid(\s+(work|schedule|model|role|position|arrangement))?|"
    r"\d\s*days?\s+(in|at)\s+(the\s+)?office|partially remote|split between home and office)\b", re.I)
ONSITE_RE = re.compile(
    r"\b(on[- ]site|onsite|in[- ]office|office[- ]based|in person|"
    r"must be (located|based) (in|at)|no remote|not a remote)\b", re.I)

csv.field_size_limit(10**7)
rows = list(csv.DictReader(open(os.path.join(ROOT, "data", "training_v4.csv"),
                                encoding="utf-8", newline="")))
Xe = np.load(CACHE, allow_pickle=True)["X"].astype(np.float32)
assert len(rows) == len(Xe)

def norm(s): return re.sub(r"\s+", " ", str(s or "")).strip().lower()
def dkey(s): return norm(s)[:300]
NON_EN = re.compile(r"[äöüßÄÖÜéèêàçñÉÈÀ]|\b(und|für|mit|der|die|das|pour|avec|les|des|dans)\b", re.I)

seen = collections.Counter(); keep = []
for i, r in enumerate(rows):
    if NON_EN.search(r["title"] + " " + r["description"][:200]): continue
    g = dkey(r["description"])
    if seen[g] >= CAP: continue
    seen[g] += 1; keep.append(i)

def remote_label(r):
    text = f"{r['title']}\n{r['description']}"
    hits = []
    if REMOTE_RE.search(text): hits.append("Remote")
    if HYBRID_RE.search(text): hits.append("Hybrid")
    if ONSITE_RE.search(text): hits.append("On-site")
    if len(hits) == 1:
        return hits[0]
    if not hits and r["remote_type"].strip() == "Remote":
        # employer ticked remote_allowed and nothing in the text contradicts it
        return "Remote"
    return None            # ambiguous or silent -> Unknown, never guessed

idx, ys = [], []
ambiguous = 0
for j, i in enumerate(keep):
    lab = remote_label(rows[i])
    if lab: idx.append(j); ys.append(lab)
    else:   ambiguous += 1

texts  = [(rows[i]["title"] + " . " + rows[i]["description"]).strip() for i in keep]
groups = [dkey(rows[i]["description"]) for i in keep]

cnt = collections.Counter(ys)
sel = [k for k, lab in enumerate(ys) if cnt[lab] >= MIN_CLASS]
tx = [texts[idx[k]] for k in sel]
gk = np.array([groups[idx[k]] for k in sel])
Xs = Xe[[keep[idx[k]] for k in sel]]
yy = [ys[k] for k in sel]
le = LabelEncoder().fit(yy); yi = np.array(le.transform(yy))

print("=" * 72)
print(f"REMOTE_MODE: {len(yi):,} labelled rows of {len(keep):,} "
      f"({100*len(yi)/len(keep):.1f}%), {len(le.classes_)} classes")
print(f"  {ambiguous:,} rows left Unknown (ambiguous or silent) -- not imputed")
for lab, c in cnt.most_common():
    print(f"    {c:>7,}  {100*c/len(ys):5.1f}%  {lab}")
sys.stdout.flush()

def fit_fold(tx_tr, X_tr, y_tr, tx_te, X_te):
    tf = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=3, max_features=40000,
                         sublinear_tf=True, strip_accents="unicode")
    A = hstack([tf.fit_transform(tx_tr), csr_matrix(X_tr)]).tocsr()
    B = hstack([tf.transform(tx_te),     csr_matrix(X_te)]).tocsr()
    m = CalibratedClassifierCV(LinearSVC(class_weight="balanced", C=2.0, max_iter=3000), cv=3)
    m.fit(A, y_tr)
    return m.predict_proba(B)

P = np.zeros((len(yi), len(le.classes_)))
for tr, te in GroupKFold(n_splits=5).split(np.zeros(len(yi)), yi, groups=gk):
    P[te] = fit_fold([tx[i] for i in tr], Xs[tr], yi[tr], [tx[i] for i in te], Xs[te])
pred = P.argmax(1)
acc  = accuracy_score(yi, pred) * 100
mf1  = f1_score(yi, pred, average="macro") * 100
kap  = cohen_kappa_score(yi, pred)
base = 100 * max(collections.Counter(yi).values()) / len(yi)
print(f"\n  accuracy={acc:.1f}%  macro-F1={mf1:.1f}%  kappa={kap:.3f}  "
      f"baseline={base:.1f}%  lift={acc-base:+.1f}")

uniq = np.array(sorted(set(gk))); np.random.RandomState(42).shuffle(uniq)
calib = set(uniq[:max(1, int(len(uniq) * CALIB_FRACTION))])
cmask = np.array([g in calib for g in gk])
thr = None
if cmask.sum() >= 200 and len(set(yi[~cmask])) == len(le.classes_):
    Pc = fit_fold([tx[i] for i in np.where(~cmask)[0]], Xs[~cmask], yi[~cmask],
                  [tx[i] for i in np.where(cmask)[0]], Xs[cmask])
    conf, ok = Pc.max(1), (Pc.argmax(1) == yi[cmask])
    for t in np.round(np.arange(0.20, 0.999, 0.005), 3):
        k = conf >= t
        if k.sum() < 100: break
        if ok[k].mean() >= 0.95:
            thr = (float(t), ok[k].mean()*100, k.mean()*100); break
    print(f"  calibration split: {cmask.sum():,} rows, group-disjoint")
    print("  95% gate: " + (f"threshold {thr[0]:.3f} -> {thr[1]:.1f}% @ {thr[2]:.1f}% coverage"
                            if thr else "NOT ACHIEVABLE"))
print()
print(classification_report(yi, pred, target_names=le.classes_, digits=3, zero_division=0))

tf = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=3, max_features=40000,
                     sublinear_tf=True, strip_accents="unicode")
F = hstack([tf.fit_transform(tx), csr_matrix(Xs)]).tocsr()
clf = CalibratedClassifierCV(LinearSVC(class_weight="balanced", C=2.0, max_iter=3000), cv=3)
clf.fit(F, yi)
with open(os.path.join(ROOT, "models", "model_remote_mode_v5.pkl"), "wb") as f:
    pickle.dump({"kind": "svm", "embedding_model": EMB, "emb_prefix": "query: ",
                 "classifier": clf, "tfidf": tf, "label_encoder": le,
                 "threshold": thr[0] if thr else None, "gatable": thr is not None,
                 "threshold_fitted_on": "separate group-disjoint calibration split",
                 "vocabulary": "handover START_HERE.md section 2",
                 "honest_accuracy": acc, "honest_macro_f1": mf1, "kappa": kap,
                 "majority_baseline": base, "n_train": int(len(yi)),
                 "label_source": "TEXT_DERIVED -- weaker than employer form-fill; "
                                 "grade against the handover's 3,073 rows, not against these labels",
                 "eval": "GroupKFold on description prefix (leakage-free)"}, f)
print("  saved models/model_remote_mode_v5.pkl")
print("\n!! These labels are text-derived. The accuracy above measures how well")
print("   the model reproduces explicit statements in the description. It is a")
print("   TRAINING result, not an independent one -- grade it against the")
print("   handover's 3,073 annotator-labelled rows before quoting it.")
