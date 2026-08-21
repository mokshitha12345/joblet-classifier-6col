"""Fast Tier-2 backfill: classify jobs_joveo_partner_v2 with the ML model.

    export SUPABASE_URL=...              # jobs project tfbjiyknoagcxjzeoqzw
    export SUPABASE_SERVICE_ROLE_KEY=...
    python scripts/run_backfill.py --limit 2000 --dry-run   # look first
    python scripts/run_backfill.py --limit 2000             # small real slice
    python scripts/run_backfill.py                          # full drain

SPEED -- why this is not the slow version:
  * ENCODE is batched: one emb.encode() call per `--read` rows (default 1000),
    internally sub-batched by `--encode`. Row-at-a-time would waste the GPU.
  * WRITE is batched: one RPC call (apply_6col_batch) updates the whole batch in
    a single SQL statement, instead of one HTTP UPDATE per row.
  * "DONE" MARKER is `classified_6col_at IS NULL`. apply_6col_batch stamps that
    timestamp on every processed row, so rows drop out of the queue and
    re-running resumes exactly where it stopped.

WHAT IT WRITES (fill-blanks -- never overwrites existing values):
  category_name / standard_role (industry/role) : ML value where still blank
  collar, experienceLevel, jobType              : ML+rule value where blank
  remote_mode : NOT written here. It is rule-owned (sql/04_fill_remote_mode.sql)
                because the ML remote head was broken (~50%); rules score ~90%.
Requires the apply_6col_batch() function from sql/03_bulk_update_function.sql.
"""
import os, sys, argparse, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "code"))
TABLE = "jobs_joveo_partner_v2"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="max rows this run (default: all)")
    ap.add_argument("--read", type=int, default=1000, help="rows fetched + written per batch")
    ap.add_argument("--encode", type=int, default=256, help="GPU encode sub-batch")
    ap.add_argument("--dry-run", action="store_true", help="predict but do not write")
    args = ap.parse_args()

    url = os.environ.get("SUPABASE_URL"); key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    if not (url and key):
        sys.exit("set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY (jobs project)")

    from supabase import create_client
    import predict_6col as P
    from sentence_transformers import SentenceTransformer

    sb = create_client(url, key)
    models = P.load_models()
    enc = SentenceTransformer(next(iter(models.values()))["embedding_model"])
    print(f"heads: {', '.join(models)}", flush=True)

    done = 0; t0 = time.time(); last_id = ""
    while True:
        take = args.read if args.limit is None else min(args.read, args.limit - done)
        if take <= 0:
            break
        # Keyset cursor (id > last_id) so the loop ALWAYS advances -- essential in
        # --dry-run, which never stamps classified_6col_at and would otherwise
        # re-read the same first page forever.
        rows = (sb.table(TABLE)
                  .select("id,title,description,jobType")
                  .is_("classified_6col_at", "null")   # the done-marker (Tier-2 tracking)
                  .gt("id", last_id)
                  .order("id").limit(take).execute().data)
        if not rows:
            print("no more unclassified rows."); break
        last_id = rows[-1]["id"]
        done += len(rows)

        preds = P.predict_batch(models, enc, rows, encode_batch=args.encode)

        if args.dry_run:
            for r, pr in list(zip(rows, preds))[:10]:
                print(r["id"], {k: pr[k] for k in ("industry","role","job_type","experience_level","collar")})
            print(f"[dry-run] would write {len(rows)} rows (total {done:,})", flush=True)
            if args.limit and done >= args.limit:
                break
            continue

        # One bulk write per batch. remote_mode is NOT written here -- it is owned
        # by the rule tier (sql/04). apply_6col_batch fills blanks and stamps
        # classified_6col_at so the row is not re-processed.
        updates = [{
            "id": r["id"],
            "industry":         pr["industry"] or None,
            "role":             pr["role"] or None,
            "collar":           pr["collar"] or None,
            "experience_level": pr["experience_level"] or None,
            "job_type":         pr["job_type"] or None,
        } for r, pr in zip(rows, preds)]
        sb.rpc("apply_6col_batch", {"updates": updates}).execute()

        rate = done / max(time.time() - t0, 1e-6)
        print(f"  {done:,} rows  ({rate:,.0f}/s)", flush=True)
        if args.limit and done >= args.limit:
            break

    print(f"done. {done:,} rows in {time.time()-t0:.0f}s.")

if __name__ == "__main__":
    main()
