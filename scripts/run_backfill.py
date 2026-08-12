"""Fast Tier-2 backfill: classify jobs_joveo_partner_v2 and write the four
target columns (collar, remote_mode, jobType, experienceLevel).

    export SUPABASE_URL=...              # jobs project tfbjiyknoagcxjzeoqzw
    export SUPABASE_SERVICE_ROLE_KEY=...
    python scripts/run_backfill.py --limit 2000 --dry-run   # look first
    python scripts/run_backfill.py --limit 2000             # small real slice
    python scripts/run_backfill.py                          # full drain

SPEED -- why this is not the slow version:
  * ENCODE is batched: one emb.encode() call per `--read` rows (default 1000),
    internally sub-batched by `--encode`. Row-at-a-time would waste the GPU and
    take days.
  * WRITE is batched: one RPC call (apply_6col_batch) updates the whole batch in
    a single SQL statement, instead of one HTTP UPDATE per row.
  * "DONE" MARKER is `remote_mode IS NULL`. Every processed row gets a
    remote_mode, so processed rows drop out of the queue automatically -- no
    tracking column, and re-running resumes exactly where it stopped.

WHAT IT WRITES (matches the real table -- no dropped columns):
  collar, remote_mode, experienceLevel : filled from model+rules (were empty)
  jobType                              : filled ONLY where currently blank, so
                                         employer-supplied feed values are kept.
  category_name / standard_role (industry/role) : NOT touched -- the existing
                                         canon pipeline owns those.
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

    done = t0 = 0; t0 = time.time()
    while True:
        take = args.read if args.limit is None else min(args.read, args.limit - done)
        if take <= 0:
            break
        rows = (sb.table(TABLE)
                  .select("id,title,description,jobType")
                  .is_("remote_mode", "null")      # the done-marker
                  .order("id").limit(take).execute().data)
        if not rows:
            print("no more unclassified rows."); break

        preds = P.predict_batch(models, enc, rows, encode_batch=args.encode)

        if args.dry_run:
            for r, pr in list(zip(rows, preds))[:10]:
                print(r["id"], {k: pr[k] for k in ("job_type","experience_level","collar","remote_mode")})
            print(f"[dry-run] would write {len(rows)} rows");
            if args.limit: break
            else: continue

        # one bulk write for the whole batch. The RPC decides per-column policy
        # (jobType only if blank; the rest overwrite the empty columns).
        updates = [{
            "id": r["id"],
            "collar":           pr["collar"] or None,
            "remote_mode":      pr["remote_mode"] or None,
            "experience_level": pr["experience_level"] or None,
            "job_type":         pr["job_type"] or None,
        } for r, pr in zip(rows, preds)]
        sb.rpc("apply_6col_batch", {"updates": updates}).execute()

        done += len(rows)
        rate = done / max(time.time() - t0, 1e-6)
        print(f"  {done:,} rows  ({rate:,.0f}/s)", flush=True)
        if args.limit and done >= args.limit:
            break

    print(f"done. {done:,} rows in {time.time()-t0:.0f}s.")

if __name__ == "__main__":
    main()
