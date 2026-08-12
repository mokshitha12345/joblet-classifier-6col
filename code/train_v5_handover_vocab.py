# Train the heads to emit the HANDOVER document's label vocabularies.
#
# Targets (from START_HERE.md section 2):
#   job_type         Full-time · Contract · Temporary · Part-time · Per diem · Internship
#   experience_level Entry · Mid · Senior · Executive
#   collar           Blue · White
#   remote_mode      On-site · Remote · Hybrid
#
# WHAT THIS SCRIPT CAN AND CANNOT PRODUCE FROM OUR LABELS
#   experience_level  ALL FOUR classes. LinkedIn distinguishes Director from
#                     Executive natively, so their 4-class scheme maps cleanly
#                     and is better supported than the 3-band collapse we shipped.
#   job_type          FIVE of six. LinkedIn has no Per diem category at all, so
#                     those labels must come from the DB or their annotators.
#   collar            Blue/White, but derived from role -- a DIFFERENT target
#                     from their annotator-defined collar (ours is 27.4% Blue,
#                     theirs is 52.9%). Trained here for completeness, not
#                     comparable to their 94.4%.
#   remote_mode       BLOCKED. remote_allowed is 1-or-null: no Hybrid exists in
#                     the source and "not flagged" is not On-site (9.5% of such
#                     rows say "fully remote" in their own text). Asserting
#                     On-site here would manufacture the error we measured.
#
# TWO FIXES vs train_v4.py, both from the handover doc:
#   sec 7  thresholds are fitted on a SEPARATE, group-disjoint calibration split,
#          not on the same out-of-fold predictions the accuracy comes from.
#   sec 4  Cohen's kappa is reported alongside accuracy and macro-F1.
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
CAP, MIN_CLASS = 2, 30
CALIB_FRACTION = 0.20      # share of description GROUPS held out for thresholds

# ---- LinkedIn native vocabulary -> handover vocabulary ---------------------
JOB_TYPE = {                     # 'Per diem' unreachable from this source
    "Full-time": "Full-time", "Contract": "Contract", "Part-time": "Part-time",
    "Internship": "Internship", "Temporary": "Temporary",
    "Other": None, "Volunteer": None,
}
EXPERIENCE = {
    "Internship": "Entry", "Entry level": "Entry",
    "Associate": "Mid", "Mid-Senior level": "Mid",
    "Director": "Senior",
    "Executive": "Executive",
}
COLLAR_BLUE = {"Healthcare Professional", "Driver & Logistics", "Construction & Real Estate"}
COLLAR_WHITE = {
    "Software Developer", "Data Analyst", "Data Scientist", "Data Engineer",
    "Machine Learning Engineer", "DevOps Engineer", "IT Specialist", "QA Engineer",
    "Cybersecurity Specialist", "Product Manager", "Designer", "Marketing Specialist",
    "Sales Representative", "HR Specialist", "Accountant & Finance", "Consultant",
    "Project Manager", "Teacher & Educator", "Executive",
    "Customer Service Representative", "Legal Professional",
}

csv.field_size_limit(10**7)
rows = list(csv.DictReader(open(os.path.join(ROOT, "data", "training_v4.csv"),
                                encoding="utf-8", newline="")))
Xe = np.load(CACHE, allow_pickle=True)["X"].astype(np.float32)
assert len(rows) == len(Xe), f"cache misaligned: {len(rows)} vs {len(Xe)}"

def norm(s): return re.sub(r"\s+", " ", str(s or "")).strip().lower()
def dkey(s): return norm(s)[:300]
NON_EN = re.compile(r"[äöüßÄÖÜéèêàçñÉÈÀ]|\b(und|für|mit|der|die|das|pour|avec|les|des|dans)\b", re.I)

seen = collections.Counter(); keep = []
for i, r in enumerate(rows):
    if NON_EN.search(r["title"] + " " + r["description"][:200]): continue
    g = dkey(r["description"])
    if seen[g] >= CAP: continue
    seen[g] += 1; keep.append(i)
print(f"variant D: {len(keep):,} of {len(rows):,} rows\n", flush=True)

texts  = [(rows[i]["title"] + " . " + rows[i]["description"]).strip() for i in keep]
groups = [dkey(rows[i]["description"]) for i in keep]

def label_for(col, r):
    if col == "job_type":
        return JOB_TYPE.get(r["job_type"].strip() or None, None)
    if col == "experience_level":
        return EXPERIENCE.get(r["experience_band"].strip() or None, None)
    if col == "collar":
        role = r["role"]
        return "Blue" if role in COLLAR_BLUE else ("White" if role in COLLAR_WHITE else None)
    return None

def fit_fold(tx_tr, X_tr, y_tr, tx_te, X_te):
    tf = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=3, max_features=40000,
                         sublinear_tf=True, strip_accents="unicode")
    A = hstack([tf.fit_transform(tx_tr), csr_matrix(X_tr)]).tocsr()
    B = hstack([tf.transform(tx_te),     csr_matrix(X_te)]).tocsr()
    m = CalibratedClassifierCV(LinearSVC(class_weight="balanced", C=2.0, max_iter=3000), cv=3)
    m.fit(A, y_tr)
    return m.predict_proba(B)

summary = []
for col in ("job_type", "experience_level", "collar"):
    idx, ys = [], []
    for j, i in enumerate(keep):
        lab = label_for(col, rows[i])
        if lab: idx.append(j); ys.append(lab)

    cnt = collections.Counter(ys)
    thin = {k: v for k, v in cnt.items() if v < MIN_CLASS}
    sel = [k for k, lab in enumerate(ys) if cnt[lab] >= MIN_CLASS]
    tx = [texts[idx[k]] for k in sel]
    gk = np.array([groups[idx[k]] for k in sel])
    Xs = Xe[[keep[idx[k]] for k in sel]]
    yy = [ys[k] for k in sel]
    le = LabelEncoder().fit(yy); yi = np.array(le.transform(yy))

    print("=" * 72)
    print(f"{col.upper()}: {len(yi):,} rows, {len(le.classes_)} classes")
    for lab, c in cnt.most_common():
        print(f"    {c:>7,}  {100*c/len(ys):5.1f}%  {lab}" + ("   DROPPED" if lab in thin else ""))
    sys.stdout.flush()

    # ---- 1. honest OOF for accuracy / macro-F1 / kappa --------------------
    P = np.zeros((len(yi), len(le.classes_)))
    for tr, te in GroupKFold(n_splits=5).split(np.zeros(len(yi)), yi, groups=gk):
        P[te] = fit_fold([tx[i] for i in tr], Xs[tr], yi[tr], [tx[i] for i in te], Xs[te])
    pred = P.argmax(1)
    acc  = accuracy_score(yi, pred) * 100
    mf1  = f1_score(yi, pred, average="macro") * 100
    kap  = cohen_kappa_score(yi, pred)
    base = 100 * max(collections.Counter(yi).values()) / len(yi)
    print(f"  accuracy={acc:.1f}%  macro-F1={mf1:.1f}%  kappa={kap:.3f}  baseline={base:.1f}%  lift={acc-base:+.1f}")

    # ---- 2. threshold on a SEPARATE group-disjoint calibration split ------
    # Handover sec 7: fitting the threshold on the same OOF predictions the
    # accuracy is reported from inflates its stated precision. Their measurement
    # of the existing heads: 95% claimed -> 86.1% industry / 84.0% role actual.
    uniq = np.array(sorted(set(gk)))
    rng = np.random.RandomState(42); rng.shuffle(uniq)
    calib_groups = set(uniq[:max(1, int(len(uniq) * CALIB_FRACTION))])
    cmask = np.array([g in calib_groups for g in gk])
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
        print(f"  calibration split: {cmask.sum():,} rows, group-disjoint from training")
        print("  95% gate: " + (f"threshold {thr[0]:.3f} -> {thr[1]:.1f}% precision @ {thr[2]:.1f}% coverage"
                                if thr else "NOT ACHIEVABLE"))
    print()
    print(classification_report(yi, pred, target_names=le.classes_, digits=3, zero_division=0))
    sys.stdout.flush()

    tf = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=3, max_features=40000,
                         sublinear_tf=True, strip_accents="unicode")
    F = hstack([tf.fit_transform(tx), csr_matrix(Xs)]).tocsr()
    clf = CalibratedClassifierCV(LinearSVC(class_weight="balanced", C=2.0, max_iter=3000), cv=3)
    clf.fit(F, yi)
    with open(os.path.join(ROOT, "models", f"model_{col}_v5.pkl"), "wb") as f:
        pickle.dump({"kind": "svm", "embedding_model": EMB, "emb_prefix": "query: ",
                     "classifier": clf, "tfidf": tf, "label_encoder": le,
                     "threshold": thr[0] if thr else None, "gatable": thr is not None,
                     "threshold_fitted_on": "separate group-disjoint calibration split",
                     "vocabulary": "handover START_HERE.md section 2",
                     "honest_accuracy": acc, "honest_macro_f1": mf1, "kappa": kap,
                     "majority_baseline": base, "n_train": int(len(yi)),
                     "label_source": "linkedin_employer_form" if col != "collar" else "derived_from_role",
                     "eval": "GroupKFold on description prefix (leakage-free)"}, f)
    print(f"  saved models/model_{col}_v5.pkl\n", flush=True)
    summary.append((col, len(yi), len(le.classes_), acc, mf1, kap, base))

print("=" * 72); print("SUMMARY -- handover vocabularies"); print("=" * 72)
print(f"{'field':<20}{'rows':>9}{'cls':>5}{'acc':>8}{'macroF1':>9}{'kappa':>8}{'base':>8}{'lift':>8}")
for c, n, k, a, m, kp, b in summary:
    print(f"{c:<20}{n:>9,}{k:>5}{a:>7.1f}%{m:>8.1f}%{kp:>8.3f}{b:>7.1f}%{a-b:>+7.1f}")
print("\nremote_mode: NOT TRAINED -- no Hybrid or true On-site exists in this label source.")
print("job_type: 5 of 6 classes -- 'Per diem' has no LinkedIn category.")
print("DONE")
