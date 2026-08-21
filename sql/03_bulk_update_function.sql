-- Bulk writer for the ML tier (Tier 2). ONE call updates a whole batch in a
-- single UPDATE instead of one HTTP round-trip per row.
--
-- Run once in the Supabase SQL editor (project tfbjiyknoagcxjzeoqzw).
-- Only UPDATEs existing rows by id; never inserts or deletes.
--
-- WHAT CHANGED (v2): the ML tier now also writes industry/role (fill-blanks),
-- does NOT write remote_mode (that column is rule-owned by sql/04), and stamps
-- `classified_6col_at` so the row is not re-processed. That timestamp is the
-- Tier-2 queue marker -- a pure tracking column, needed because remote_mode is
-- no longer ML-set and can't serve as the marker.
--
-- FILL-BLANKS: every content column keeps whatever is already there. This keeps
-- Tier 2 from overwriting the keyword tier, the canon pipeline, or employer data.

-- Tracking column for the Tier-2 queue (metadata-only add, fast on 1.8M rows).
alter table jobs_joveo_partner_v2
  add column if not exists classified_6col_at timestamptz;

create index if not exists idx_jjpv2_unclassified_6col
  on jobs_joveo_partner_v2 (id) where classified_6col_at is null;

create or replace function apply_6col_batch(updates jsonb)
returns integer
language plpgsql
as $$
declare
  n integer;
begin
  update jobs_joveo_partner_v2 j
  set
    category_name     = coalesce(nullif(btrim(j.category_name), ''), u.industry),
    standard_role     = coalesce(nullif(btrim(j.standard_role), ''), u.role),
    collar            = coalesce(j.collar, u.collar),
    "experienceLevel" = coalesce(nullif(btrim(j."experienceLevel"), ''), u.experience_level),
    "jobType"         = case
                          when j."jobType" is null or j."jobType" = ''
                            then u.job_type
                          else j."jobType"
                        end,
    classified_6col_at = now()          -- always stamped -> row leaves the queue
  from jsonb_to_recordset(updates) as u(
    id               text,
    industry         text,
    role             text,
    collar           text,
    experience_level text,
    job_type         text
  )
  where j.id = u.id;

  get diagnostics n = row_count;
  return n;
end;
$$;

-- self-test (updates nothing): select apply_6col_batch('[]'::jsonb);  -> 0
