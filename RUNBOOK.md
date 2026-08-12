# Runbook — how to actually run this

Plain steps. Two phases: the one-time backfill (rent a GPU, drain 1.8M rows
once), then the daily auto-run (serverless, cheap, hands-off).

---

## How it fits together

```
  a job row
     │
     ▼
  ONE Python worker (predict_6col.predict_batch)
     ├─ e5 embedding, batched
     ├─ 6 model heads  → industry, role, job_type, experience, collar, remote
     └─ keyword RULES override the obvious ones (Uber→Contract, nurse→Blue, …)
     │
     ▼
  bulk write via apply_6col_batch()  → collar, remote_mode, experienceLevel,
                                        jobType (only if blank)
```

The "rule-based thing" and the "ML thing" are **the same worker** now — the
rules run right after the model in one step. No separate SQL job to babysit.

Writes land in the real columns: `collar`, `remote_mode`, `experienceLevel`,
`jobType`. Industry/role (`category_name`, `standard_role`) are left to your
existing pipeline.

---

## One-time DB setup (you run these)

In Supabase SQL editor, project `tfbjiyknoagcxjzeoqzw`:

```sql
-- the bulk-write function the worker calls
-- (paste the whole contents of sql/03_bulk_update_function.sql)
```

That's the only SQL step. `collar` and `remote_mode` columns already exist.

---

## Phase 1 — the one-time backfill on a rented GPU

### 1. Rent the GPU

- runpod.io → **Deploy → Pods → Community Cloud**
- GPU: **RTX 4090** (or L4 / A5000 if none free)
- Template: **RunPod PyTorch 2.x**, container disk ~20 GB
- Start it, open the web terminal.

### 2. Get the code + models onto it

Easiest: push this folder to a **private GitHub repo** (see DEPLOY.md), then on
the pod:

```bash
git clone https://github.com/<you>/joblet-classifier-6col.git
cd joblet-classifier-6col
pip install -r requirements.txt
```

### 3. Set the keys and test on a tiny slice FIRST

```bash
export SUPABASE_URL="https://tfbjiyknoagcxjzeoqzw.supabase.co"
export SUPABASE_SERVICE_ROLE_KEY="<jobs project service role key>"

# look, don't write:
python scripts/run_backfill.py --limit 200 --dry-run
```

Read the printed predictions. If they look sane, write a small real slice:

```bash
python scripts/run_backfill.py --limit 2000
```

Check those 2000 rows in Supabase. Happy? Drain the rest:

```bash
python scripts/run_backfill.py
```

Watch the `rows (rate/s)` line. On a 4090 expect very roughly **1-2 hours** for
the full table. It is restartable — if the pod dies, re-run the same command and
it resumes (processed rows have remote_mode set, so they're skipped).

### 4. STOP the pod

Runpod → your pod → **Stop**. You stop paying. Don't leave it running.

---

## Phase 2 — daily automatic run (serverless, scale-to-zero)

After the backfill, only a few thousand new rows arrive per day. This runs them
automatically and costs cents.

### 1. Build + push the serverless image (once)

```bash
docker build -t <registry>/joblet-6col:latest -f serverless/Dockerfile .
docker push <registry>/joblet-6col:latest
```

### 2. Create the endpoint

Runpod → **Serverless → New Endpoint** → your image → GPU RTX 4090 →
**min workers 0** (this is what makes it scale to zero) → add endpoint secrets
`SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY`. Copy the endpoint id.

### 3. Wire the daily trigger

GitHub repo → **Settings → Secrets and variables → Actions**, add:
`SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY`, `RUNPOD_API_KEY`,
`RUNPOD_ENDPOINT_ID`.

`.github/workflows/daily-classify.yml` already runs daily at 03:00 UTC and pokes
the endpoint. Change the cron to sit ~30 min after your ingestion finishes.

### 4. Test it

GitHub → **Actions → daily-classify → Run workflow**. The log shows the worker's
`{"processed": N, ...}`. Run it once with the endpoint's default `dry_run` off
only after a dry run looks right.

---

## Cost, roughly

- Backfill: one afternoon on a 4090 ≈ **$2-4 total**, then you stop the pod.
- Daily: a few thousand rows, seconds of GPU ≈ **cents/day**.

---

## Safety reminders

- **Dry-run first**, always, before any real write.
- `collar` and `remote_mode` are the two provisional fields (see TRUST.md).
  Keep them out of the public site until validated — the model writes them, but
  don't show them to users yet.
- The worker never overwrites a `jobType` that already has an employer value.
- If a run goes wrong, the fix is per-field, e.g. wipe just remote_mode and
  re-run: `update jobs_joveo_partner_v2 set remote_mode = null;` (worker refills).
