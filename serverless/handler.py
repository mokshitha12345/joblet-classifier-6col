"""Runpod Serverless handler: the daily Tier-2 worker.

Invoked by the GitHub Actions cron after ingestion. Wakes the GPU, drains rows
where remote_mode IS NULL (batched encode + one bulk RPC write per batch),
scales back to zero. Same fast path as scripts/run_backfill.py.

Requires apply_6col_batch() from sql/03_bulk_update_function.sql.

Input (all optional):
    {"input": {"max_rows": 50000, "read": 1000, "encode": 256, "dry_run": false}}
Output:
    {"processed": N, "seconds": S, "rate_per_s": R, "dry_run": bool}
"""
import os, sys, time
import runpod

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "code"))

from supabase import create_client
import predict_6col as P
from sentence_transformers import SentenceTransformer

TABLE = "jobs_joveo_partner_v2"
_SB = _ENC = _MODELS = None

def _init():
    global _SB, _ENC, _MODELS
    if _SB is not None:
        return
    _SB = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])
    _MODELS = P.load_models()
    _ENC = SentenceTransformer(next(iter(_MODELS.values()))["embedding_model"])

def handler(event):
    _init()
    cfg = (event or {}).get("input", {}) or {}
    max_rows = int(cfg.get("max_rows", 50000))
    read     = int(cfg.get("read", 1000))
    encode   = int(cfg.get("encode", 256))
    dry_run  = bool(cfg.get("dry_run", False))

    t0 = time.time(); done = 0
    while done < max_rows:
        take = min(read, max_rows - done)
        rows = (_SB.table(TABLE)
                  .select("id,title,description,jobType")
                  .is_("remote_mode", "null")
                  .order("id").limit(take).execute().data)
        if not rows:
            break
        preds = P.predict_batch(_MODELS, _ENC, rows, encode_batch=encode)
        if not dry_run:
            updates = [{
                "id": r["id"],
                "collar":           pr["collar"] or None,
                "remote_mode":      pr["remote_mode"] or None,
                "experience_level": pr["experience_level"] or None,
                "job_type":         pr["job_type"] or None,
            } for r, pr in zip(rows, preds)]
            _SB.rpc("apply_6col_batch", {"updates": updates}).execute()
        done += len(rows)

    secs = round(time.time() - t0, 1)
    return {"processed": done, "seconds": secs,
            "rate_per_s": round(done / max(secs, 1e-6), 1), "dry_run": dry_run}

runpod.serverless.start({"handler": handler})
