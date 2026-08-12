# Deployment & operations

How this runs in production: the schema change, the daily two-tier pipeline, the
GPU for the model tier, and pushing the repo to git.

---

## The pipeline, end to end

```
  daily ingestion writes new rows to jobs_joveo_partner_v2
        │
        ▼
  TIER 1  SQL keyword rules   ── free, instant, no GPU
        │  classify_6col_keyword_chunk()
        │  labels only the OBVIOUS jobs (drivers, nurses, "Intern", "Remote")
        │  leaves the rest NULL
        ▼
  TIER 2  ML model            ── needs a GPU for the encode
        │  predict_6col.py over rows the rules left NULL
        │  writes label + confidence + method='model'
        ▼
  low-confidence rows stay NULL / flagged for review
```

Same two-tier shape you already run for role/industry (`canon_keyword_chunk` then
the classifier). Tier 1 handles the easy majority for free; Tier 2 is the
expensive model, run only on what's left.

---

## Cadence: daily, incremental, after ingestion

This is NOT a one-time backfill. It runs every day on that day's new rows.

Both tiers are naturally incremental because they key off `classified_6col_at IS
NULL` and per-field `*_method` — a row already labelled is skipped, so a daily
run only touches new/unclassified rows.

**Order matters:** ingestion → Tier 1 → Tier 2. Tier 1 must finish first so the
model only spends GPU on rows the rules couldn't settle.

### Tier 1 — schedule with pg_cron (runs in the database, no server)

```sql
-- after applying 01_add_columns.sql and 02_keyword_backfill_6col.sql
SELECT cron.schedule(
  'classify-6col-keyword',
  '30 2 * * *',                    -- 02:30 daily, AFTER your ingestion window
  'SELECT public.classify_6col_keyword_chunk();'
);
```

The function does 1000 rows per call and returns the count. For a daily delta
that fits in one call; if a day's ingestion is large, call it in a short loop or
drop the cron to `*/2 * * * *` for the drain window — it self-limits via the
advisory lock so overlapping calls are safe.

### Tier 2 — automated GPU model via Runpod Serverless

**GitHub has no GPU** — its runners are CPU-only. So the GPU lives on Runpod
Serverless (scale-to-zero), and a GitHub Actions cron triggers it. You pay only
for the seconds the worker runs — cents a day for a daily delta.

```
  ingestion ──▶ GitHub Actions cron (CPU, orchestrator only)
                     │  1. drains Tier 1 keyword SQL via PostgREST rpc
                     │  2. POSTs the Runpod serverless endpoint
                     ▼
              Runpod Serverless (GPU wakes, drains NULL rows, scales to zero)
                     │  serverless/handler.py -> code/predict_6col.py -> models/*.pkl
                     ▼
              predictions written back, method='model'
```

Pieces, all in this repo:

| File | Role |
|---|---|
| `.github/workflows/daily-classify.yml` | the daily cron: Tier 1 then trigger Tier 2 |
| `serverless/handler.py` | the GPU worker — pulls NULL rows, classifies, writes back |
| `serverless/Dockerfile` | container with models + encoder baked in |
| `scripts/run_backfill.py` | same logic for the one-time full backfill from a plain pod |

**Deploy the endpoint once:**

1. Build and push the image (any registry Runpod can pull — Docker Hub, GHCR):
   ```bash
   docker build -t <registry>/joblet-6col:latest -f serverless/Dockerfile .
   docker push <registry>/joblet-6col:latest
   ```
2. Runpod → **Serverless → New Endpoint** → your image → GPU **RTX 4090**,
   min workers **0** (scale to zero), max workers 1-2.
3. Add endpoint secrets: `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY`.
4. Copy the **endpoint id**.
5. In GitHub → repo **Settings → Secrets → Actions**, add: `SUPABASE_URL`,
   `SUPABASE_SERVICE_ROLE_KEY`, `RUNPOD_API_KEY`, `RUNPOD_ENDPOINT_ID`.
6. Test: GitHub → **Actions → daily-classify → Run workflow**. Watch the log
   for the worker's `{"processed": N, "written": M}` summary.

**Serverless cold start:** the first call after idle takes ~20-40 s to boot the
GPU worker; the encoder is baked into the image so it does not re-download. For a
once-a-day job that is fine. If you ever need it always-warm, set min workers 1
(you then pay hourly — usually not worth it here).

**Prefer CPU?** For a daily delta of a few thousand rows a GPU is optional —
e5-large does that on CPU in minutes. Deploy the same image to a **CPU**
serverless endpoint and the cost drops further. The code is identical.

`scripts/run_backfill.py` is the batch driver for the one-time full backfill from
a plain rented pod; see its header.

---

## GPU: do you even need to rent one?

**For daily inference — probably not.** The model is a frozen e5-large encode
plus linear heads. e5-large runs on CPU at ~1-2 jobs/sec. A daily delta of a few
thousand new jobs is minutes on CPU. **Start CPU-only and measure** before paying
for a GPU.

**You need a rented GPU for two things:**
- the **first full backfill** of the ~1.8M existing rows (CPU would take days),
- **retraining** if you regenerate embeddings for a new dataset.

### Which Runpod GPU — pick the RTX 4090

The bottleneck is the e5-large encoder forward pass, which is memory-bandwidth
bound. That makes card choice matter, and it makes the T4 the WRONG pick -- old
Turing, ~320 GB/s, no bf16, the slowest thing Runpod rents for this.

**Pick: RTX 4090 (24 GB).** ~1 TB/s bandwidth (~3-5x a T4's real encode
throughput), fast fp16/bf16, and e5-large's 2 GB leaves room for batch 256-512
where the 4090 pulls further ahead. It costs more per hour but finishes the
backfill ~5x sooner, so total cost is LOWER.

| Card (Runpod) | ~$/hr | vs T4 | Full 1.8M backfill | Total |
|---|---|---|---|---|
| T4 16GB | ~$0.30 | 1x | ~15-20 hr | ~$5-6 |
| **RTX 4090 24GB** | ~$0.40-0.70 | **~4-5x** | **~3-4 hr** | **~$2-3** |
| L40S 48GB | ~$0.80-1.10 | ~5x | ~3 hr | ~$3-4 |
| A100 80GB | ~$1.60-2.50 | ~5-6x | ~2.5 hr | ~$5-7 |

Prices fluctuate; check live stock. The A100 is a trap here -- you'd pay for HBM
and interconnect an embedding encode never uses.

**If no 4090 is in stock:** L4 (24 GB) or A5000 (24 GB). Both beat a T4 and are
usually available. Past the T4, exact card matters less than batch size.

Two things that decide whether the card earns its price:
  * Use **Community Cloud**, not Secure -- roughly half price, and reliability is
    irrelevant for a restartable job (run_backfill.py resumes from
    classified_6col_at IS NULL).
  * The 4090's edge is real ONLY with batching. Pass `--batch 256` (or 512).
    Row-at-a-time wastes the card and a T4 would nearly match it.

### Runpod, step by step

1. runpod.io → sign up → add credit ($10 covers the full backfill many times).
2. **Deploy → Pods → GPU** → **Community Cloud** → pick **RTX 4090** (or L4 /
   A5000 if none in stock) → template **"RunPod PyTorch 2.x"**.
3. Set container disk to ~20 GB (e5-large download + models).
4. Deploy, then **Connect → Web Terminal** (or SSH).
5. In the terminal:
   ```bash
   git clone <your-repo-url> && cd joblet-classifier-6col
   pip install -r requirements.txt
   # models are in the repo; encoder downloads on first run (~2 GB)
   python scripts/run_backfill.py --limit 5000        # test on 5k first
   ```
6. Set the DB connection via env vars (never commit them):
   ```bash
   export SUPABASE_URL=...            # jobs project (tfbjiyknoagcxjzeoqzw)
   export SUPABASE_SERVICE_ROLE_KEY=...
   ```
7. Validate the 5k, then run the full backfill, then **stop the pod** so billing
   ends. Per-second billing means a stopped pod costs nothing but disk.

Cost estimate for the full backfill: ~1.8M rows minus whatever Tier 1 caught,
at GPU throughput ~50-100 jobs/sec batched ≈ 5-10 GPU-hours ≈ **$2-4 total.**

---

## Pushing to git

The repo is not yet a git repo. Before the first push:

### 1. Handle the large files

`data/training_v4.csv` (28 MB) and the model `.pkl` files (2-26 MB each) are
under GitHub's 100 MB hard limit but over the 50 MB warning, and git is a poor
home for binaries. Two options:

**Option A — Git LFS (recommended if the models must live in the repo):**
```bash
git lfs install
git lfs track "*.pkl" "*.csv" "*.npz"
git add .gitattributes
```

**Option B — keep binaries out of git** (recommended if the GPU box can fetch
them from storage): add them to `.gitignore` and store models + training data in
Supabase Storage or S3, downloaded by `run_backfill.py`. Keeps the repo small
and clone-fast.

### 2. Create `.gitignore` regardless

```
# never commit
*.env
.env*
**/secrets*
embeddings_cache_e5large.npz     # 220 MB, regenerable
logs/*.log

# with Option B, also:
# models/*.pkl
# data/*.csv
```

### 3. Init and push

```bash
cd joblet-classifier-6col
git init && git branch -M main
git add -A && git commit -m "Six-column job classifier: schema, keyword tier, model, docs"
gh repo create joblet-classifier-6col --private --source=. --push
# or point at an existing remote:
# git remote add origin <url> && git push -u origin main
```

Push to a **branch and open a PR**, don't push straight to a shared main — and
remember the `remote` column change in `01_add_columns.sql` needs its two app
edits (jobDetailFromRow.js, jobListingQueryFilters.js) in the SAME PR or the live
site breaks silently. Those live in the main `joblet1.0` repo, not this one.

---

## Order of operations (the checklist)

1. [ ] Apply `sql/01_add_columns.sql` in a low-traffic window (it rewrites
       `remote`), **with** the two app-code edits in the same deploy.
2. [ ] Apply `sql/02_keyword_backfill_6col.sql` (creates the function).
3. [ ] Run one manual `SELECT classify_6col_keyword_chunk();`, eyeball the
       results against `TRUST.md`.
4. [ ] Schedule Tier 1 via pg_cron, after the ingestion window.
5. [ ] Stand up the GPU box, run `run_backfill.py --limit 5000`, validate.
6. [ ] Full backfill of existing rows, then stop the pod.
7. [ ] Wire Tier 2 into the daily schedule after Tier 1.
8. [ ] Leave `collar` and `remote_mode` in shadow (don't show on the site) until
       validated against an independent label set — see `TRUST.md`.
