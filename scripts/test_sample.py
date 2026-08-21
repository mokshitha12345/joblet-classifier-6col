"""TEST the 6-column classifier on a sample of real DB rows — WRITES NOTHING
to the database. Pulls N unclassified rows, classifies them, and:

  1. saves every prediction to a CSV you can open in Excel (accuracy check)
  2. prints coverage per column (% filled + class distribution)
  3. prints run time and rows/sec (speed)

    python scripts/test_sample.py --limit 5000
    python scripts/test_sample.py --limit 100000

Output CSV: test_results_<limit>.csv  (in the project folder)
Safe to run repeatedly — it only READS the database.
"""
import os, sys, time, csv, argparse, collections

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "code"))
TABLE = "jobs_joveo_partner_v2"
COLS = ["industry", "role", "job_type", "experience_level", "collar", "remote_mode"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=5000, help="how many rows to test")
    ap.add_argument("--read", type=int, default=1000, help="rows fetched per DB call")
    ap.add_argument("--encode", type=int, default=256, help="GPU encode batch size")
    ap.add_argument("--out", default=None, help="output CSV path")
    args = ap.parse_args()
    out = args.out or os.path.join(ROOT, f"test_results_{args.limit}.csv")

    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    if not (url and key):
        sys.exit("set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY first")

    from supabase import create_client
    import predict_6col as P
    from sentence_transformers import SentenceTransformer

    print("loading model + encoder (first run downloads e5 ~2GB) ...", flush=True)
    sb = create_client(url, key)
    models = P.load_models()
    enc = SentenceTransformer(next(iter(models.values()))["embedding_model"])

    # ---- pull a representative slice of LIVE jobs (read-only; never writes) ----
    print(f"pulling up to {args.limit:,} rows from the database ...", flush=True)
    rows, last = [], ""
    while len(rows) < args.limit:
        take = min(args.read, args.limit - len(rows))
        page = (sb.table(TABLE)
                  .select("id,title,description")
                  .eq("is_active", True)
                  .gt("id", last).order("id").limit(take).execute().data)
        if not page:
            break
        rows.extend(page); last = page[-1]["id"]
    print(f"got {len(rows):,} rows. classifying ...", flush=True)

    # ---- classify, timing only the GPU work ----
    cov = {c: collections.Counter() for c in COLS}
    results = []
    t0 = time.time()
    for i in range(0, len(rows), args.encode):
        chunk = rows[i:i + args.encode]
        preds = P.predict_batch(models, enc, chunk, encode_batch=args.encode)
        for r, p in zip(chunk, preds):
            row = {"id": r["id"], "title": (r.get("title", "") or "")[:80]}
            for c in COLS:
                v = p.get(c, "") or ""
                row[c] = v
                if v:
                    cov[c][v] += 1
            row["rules_fired"] = p.get("_notes", "")
            results.append(row)
        print(f"  {len(results):,}/{len(rows):,}", flush=True)
    secs = time.time() - t0
    n = len(results)

    # ---- save CSV ----
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "title"] + COLS + ["rules_fired"])
        w.writeheader()
        for row in results:
            w.writerow(row)

    # ---- report ----
    print("\n" + "=" * 64)
    print(f"DONE: {n:,} rows in {secs:.0f}s   =   {n/max(secs,1e-9):.0f} rows/sec")
    print(f"results saved -> {out}")
    print("=" * 64)
    print("\nCOVERAGE (how many rows the model filled for each column):")
    for c in COLS:
        filled = sum(cov[c].values())
        print(f"\n  {c}: {filled:,}/{n:,} filled ({100*filled/max(n,1):.0f}%)")
        for v, k in cov[c].most_common(6):
            print(f"      {k:>7,}  {v}")
    print("\nOpen the CSV to eyeball accuracy — read some titles vs their predicted columns.")

if __name__ == "__main__":
    main()
