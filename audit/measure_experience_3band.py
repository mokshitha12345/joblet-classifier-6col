# The one number missing from the v4 report.
#
# train_v4.py trains experience_band on LinkedIn's NATIVE 6 levels
# (Internship / Entry level / Associate / Mid-Senior level / Director /
# Executive) and reports 72.4%. But the product enum is 3 bands, and collapsing
# 6 classes into 3 merges several of the model's most common confusions --
# Associate<->Mid-Senior and Director<->Executive stop counting as errors.
#
# So 72.4% is NOT the number that belongs in the handover. This measures the
# one that does, using the identical leakage-free GroupKFold protocol so the
# two figures are directly comparable.
#
#   python measure_experience_3band.py
import csv, os, re, collections, numpy as np
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CACHE = r"C:\Users\Mokshitha Sree\Downloads\categorization ml model\embeddings_cache_e5large.npz"
CAP, MIN_CLASS = 2, 30

BAND = {"Internship": "Entry", "Entry level": "Entry",
        "Associate": "Mid-Senior", "Mid-Senior level": "Mid-Senior",
        "Director": "Senior", "Executive": "Senior"}

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
    if NON_EN.search(r["title"] + " " + r["description"][:200]):
        continue
    g = dkey(r["description"])
    if seen[g] >= CAP: continue
    seen[g] += 1; keep.append(i)

def run(label_fn, name):
    idx = [j for j, i in enumerate(keep) if rows[i]["experience_band"].strip()]
    y_all = [label_fn(rows[keep[j]]["experience_band"].strip()) for j in idx]
    cnt = collections.Counter(y_all)
    sel = [j for j, lab in enumerate(y_all) if cnt[lab] >= MIN_CLASS]
    tx = [(rows[keep[idx[j]]]["title"] + " . " + rows[keep[idx[j]]]["description"]).strip() for j in sel]
    gk = [dkey(rows[keep[idx[j]]]["description"]) for j in sel]
    Xs = Xe[[keep[idx[j]] for j in sel]]
    ys = [y_all[j] for j in sel]
    le = LabelEncoder().fit(ys); yi = np.array(le.transform(ys))

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
    base = 100 * max(collections.Counter(yi).values()) / len(yi)

    # can it be gated to 95%?
    conf = P.max(1); ok = (P.argmax(1) == yi); thr = None
    for t in np.round(np.arange(0.20, 0.999, 0.005), 3):
        k = conf >= t
        if k.sum() < 200: break
        if ok[k].mean() >= 0.95:
            thr = (float(t), ok[k].mean()*100, k.mean()*100); break

    print(f"\n{'='*68}\n{name}: {len(yi):,} rows, {len(le.classes_)} classes\n{'='*68}")
    for lab, c in cnt.most_common():
        print(f"    {c:>7,}  {100*c/len(y_all):5.1f}%  {lab}")
    print(f"  accuracy = {acc:.1f}%   macro-F1 = {mf1:.1f}%")
    print(f"  majority baseline = {base:.1f}%   lift = {acc-base:+.1f} pts")
    print("  95% gate: " + (f"threshold {thr[0]:.3f} -> {thr[1]:.1f}% @ {thr[2]:.1f}% coverage"
                            if thr else "NOT ACHIEVABLE at any confidence"))
    cm = confusion_matrix(yi, P.argmax(1))
    print(f"\n  confusion (rows = truth, cols = predicted)")
    print("           " + "".join(f"{c[:11]:>13}" for c in le.classes_))
    for i, c in enumerate(le.classes_):
        print(f"  {c[:9]:<9}" + "".join(f"{v:>13,}" for v in cm[i]))
    return acc, mf1, base, thr

print("Comparing the model's native vocabulary against the product enum.")
a6 = run(lambda v: v, "NATIVE 6-LEVEL (what train_v4.py reported)")
a3 = run(lambda v: BAND.get(v, v), "3-BAND PRODUCT ENUM (the handover number)")

print(f"\n{'='*68}\nVERDICT\n{'='*68}")
print(f"  native 6-level : {a6[0]:.1f}% accuracy, {a6[1]:.1f}% macro-F1, {a6[0]-a6[2]:+.1f} lift")
print(f"  3-band product : {a3[0]:.1f}% accuracy, {a3[1]:.1f}% macro-F1, {a3[0]-a3[2]:+.1f} lift")
print(f"  collapsing 6->3 moves accuracy {a3[0]-a6[0]:+.1f} points and lift {(a3[0]-a3[2])-(a6[0]-a6[2]):+.1f}")
print("\n  Lift is the honest comparison. Accuracy always rises when classes merge,")
print("  because merging removes errors without the model improving.")
