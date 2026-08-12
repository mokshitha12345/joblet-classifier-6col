"""Six-column job classifier.

  industry · role · job_type · experience_level · collar · remote_mode

    python predict_6col.py "Travel ICU RN" "13-week contract, night shift"
    python predict_6col.py --csv myjobs.csv        -> predictions_6col.csv
    python predict_6col.py                         -> demo

ARCHITECTURE
    text ──┬─→ e5-large embedding (frozen, 1024-dim)  ─┐
           └─→ TF-IDF (per head, <=40k)  ─────────────┴─→ calibrated LinearSVC ─→ label + conf

    The encoder is the only expensive part and it RUNS ONCE PER JOB. All six
    heads read that same vector, which is why six columns cost barely more than
    two. Do not move the encode() call inside the head loop.

TRUST -- READ THIS BEFORE QUOTING ANY CONFIDENCE FROM THIS SCRIPT
    Four heads rest on independent labels. Two do not, and are marked
    provisional in every output:
      collar       labels were DERIVED FROM ROLE by a mapping, so the head is
                   partly learning to reproduce that mapping.
      remote_mode  labels were derived from TEXT PATTERNS, so the head is
                   partly learning to spot the phrase that made the label.
    Both print high numbers for that reason. Grade them against an independent
    annotator set before believing them. See ../TRUST.md.
"""
import sys, os, csv, pickle, re
import numpy as np
from scipy.sparse import hstack, csr_matrix
from sentence_transformers import SentenceTransformer

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MODELS = os.path.join(ROOT, "models")
sys.path.insert(0, HERE)
from guardrails import apply_guardrails, should_dual

# column -> model file. Order here is the output order.
HEADS = [
    ("industry",         "model_industry_v4.pkl"),
    ("role",             "model_role_v4.pkl"),
    ("job_type",         "model_job_type_v5.pkl"),
    ("experience_level", "model_experience_level_v5.pkl"),
    ("collar",           "model_collar_v5.pkl"),
    ("remote_mode",      "model_remote_mode_v5.pkl"),
]
# Heads whose labels were not independently produced. Their confidence is real
# but its meaning is not, so every prediction carries a provenance warning.
PROVISIONAL = {
    "collar":      "labels derived from role",
    "remote_mode": "labels derived from text patterns",
}

def clean(s):
    s = re.sub(r"<[^>]+>", " ", str(s or ""))
    return re.sub(r"\s+", " ", s).strip()

def load_models():
    models = {}
    for col, fname in HEADS:
        path = os.path.join(MODELS, fname)
        if not os.path.exists(path):
            print(f"warning: {fname} missing -> '{col}' will be blank", file=sys.stderr)
            continue
        with open(path, "rb") as f:
            models[col] = pickle.load(f)
    if not models:
        raise SystemExit(f"no models found in {MODELS}")
    return models

def predict_one(models, emb_model, title, desc=""):
    text = (clean(title) + " . " + clean(desc)[:600]).strip()
    # THE single encode. Every head below reuses this exact vector.
    e = emb_model.encode(["query: " + text], normalize_embeddings=True)

    raw = {}
    for col, m in models.items():
        F = hstack([m["tfidf"].transform([text]), csr_matrix(e)]).tocsr()
        p = m["classifier"].predict_proba(F)[0]
        order = p.argsort()[::-1]
        raw[col] = {
            "top1": str(m["label_encoder"].inverse_transform([order[0]])[0]),
            "top2": (str(m["label_encoder"].inverse_transform([order[1]])[0])
                     if len(order) > 1 else None),
            "conf": float(p[order[0]]),
            "thr":  m.get("threshold"),
            # gatable=False means no confidence cut reached 95% precision on a
            # held-out calibration split, so the score cannot separate safe rows
            # from unsafe ones. Flag everything rather than trust a fallback.
            "gatable": m.get("gatable", m.get("threshold") is not None),
        }

    out = {}
    if "industry" in raw and "role" in raw:
        ind, role, why = apply_guardrails(title, raw["industry"]["top1"], raw["role"]["top1"])
    else:
        ind, role, why = raw.get("industry", {}).get("top1"), raw.get("role", {}).get("top1"), ""

    for col in ("industry", "role"):
        if col not in raw:
            continue
        r = raw[col]
        label = {"industry": ind, "role": role}[col]
        if why:   # a guardrail override is high-precision; never DUAL it
            out[col] = {"value": label, "conf": round(r["conf"]*100, 1),
                        "review": False, "alt": None, "note": f"rule:{why}"}
            continue
        dual = should_dual(label, r["conf"], r["thr"]) if r["thr"] else False
        out[col] = {"value": label, "conf": round(r["conf"]*100, 1),
                    "review": bool(r["thr"]) and r["conf"] < r["thr"],
                    "alt": r["top2"] if dual else None, "note": ""}

    for col in ("job_type", "experience_level", "collar", "remote_mode"):
        if col not in raw:
            out[col] = {"value": "", "conf": 0.0, "review": True,
                        "alt": None, "note": "model_missing"}
            continue
        r = raw[col]
        review = True if not r["gatable"] else r["conf"] < r["thr"]
        note = "" if r["gatable"] else "ungatable"
        if col in PROVISIONAL:
            note = (note + ";" if note else "") + "PROVISIONAL:" + PROVISIONAL[col]
        out[col] = {"value": r["top1"], "conf": round(r["conf"]*100, 1),
                    "review": review, "alt": None, "note": note}
    return out

COLUMNS = [c for c, _ in HEADS]

# ---- batched prediction: the fast path for backfill / serverless ------------
# One encode() call for the WHOLE batch (this is where a GPU earns its rent;
# row-at-a-time wastes it entirely), then each head runs over the full matrix at
# once, then the keyword rules override per row. Returns a plain dict per row.
try:
    from guardrails_4col import apply_4col_guardrails
except Exception:
    apply_4col_guardrails = None

def predict_batch(models, emb_model, rows, encode_batch=256):
    texts = [(clean(r.get("title", "")) + " . " + clean(r.get("description", ""))[:600]).strip()
             for r in rows]
    embs = emb_model.encode(["query: " + t for t in texts],
                            normalize_embeddings=True, batch_size=encode_batch)
    # per-head argmax over the whole batch in one predict_proba call
    head = {}
    for col, m in models.items():
        F = hstack([m["tfidf"].transform(texts), csr_matrix(embs)]).tocsr()
        Pr = m["classifier"].predict_proba(F)
        idx = Pr.argmax(1)
        head[col] = (m["label_encoder"].inverse_transform(idx), Pr.max(1))

    out = []
    for i, r in enumerate(rows):
        rec = {c: (str(head[c][0][i]) if c in head else "") for c in COLUMNS}
        rec["_conf"] = float(head["role"][1][i]) if "role" in head else 0.0
        # keyword rules override the four target fields (Uber->Contract, nurse->Blue,
        # explicit title terms, etc). Same logic as guardrails_4col.py.
        if apply_4col_guardrails:
            jt, ex, co, rm, notes = apply_4col_guardrails(
                r.get("title", ""), r.get("description", ""),
                rec.get("job_type", ""), rec.get("experience_level", ""),
                rec.get("collar", ""), rec.get("remote_mode", ""))
            rec["job_type"], rec["experience_level"] = jt, ex
            rec["collar"], rec["remote_mode"] = co, rm
            rec["_notes"] = notes
        out.append(rec)
    return out

def main():
    args = sys.argv[1:]
    models = load_models()
    print(f"heads loaded: {', '.join(models)}", file=sys.stderr)
    any_model = next(iter(models.values()))
    print(f"loading encoder: {any_model['embedding_model']} ...", file=sys.stderr)
    emb_model = SentenceTransformer(any_model["embedding_model"])

    if args and args[0] == "--csv":
        path = args[1]
        rows = list(csv.DictReader(open(path, encoding="utf-8", newline="")))
        outp = os.path.join(os.path.dirname(os.path.abspath(path)), "predictions_6col.csv")
        header = ["title"]
        for c in COLUMNS:
            header += [c, f"{c}_conf", f"{c}_review", f"{c}_note"]
        with open(outp, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f); w.writerow(header)
            for r in rows:
                p = predict_one(models, emb_model, r.get("title", ""), r.get("description", ""))
                line = [r.get("title", "")]
                for c in COLUMNS:
                    line += [p[c]["value"], p[c]["conf"],
                             "yes" if p[c]["review"] else "", p[c]["note"]]
                w.writerow(line)
        print(f"wrote {len(rows)} rows -> {outp}")
        return

    def show(title, p):
        print(f"\nTitle: {title}\n")
        for c in COLUMNS:
            v = p[c]
            bits = []
            if v["alt"]:    bits.append(f"also {v['alt']}")
            if v["review"]: bits.append("NEEDS REVIEW")
            if v["note"]:   bits.append(v["note"])
            tail = "   [" + ", ".join(bits) + "]" if bits else ""
            print(f"  {c:<18}{(v['value'] or '(none)'):<26}{v['conf']:>5.1f}%{tail}")
        print()

    if args:
        show(args[0], predict_one(models, emb_model, args[0],
                                  args[1] if len(args) > 1 else ""))
        return

    for t, d in [
        ("Travel ICU RN", "13-week contract, night shift critical care, 2 years experience"),
        ("Senior Software Engineer", "fully remote, 8+ years Python, full-time permanent"),
        ("Warehouse Associate - Part Time", "evenings, forklift cert a plus, on-site"),
        ("Marketing Intern", "summer internship supporting the brand team, hybrid schedule"),
    ]:
        show(t, predict_one(models, emb_model, t, d))

if __name__ == "__main__":
    main()
