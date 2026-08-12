-- Bulk writer for the ML worker (Tier 2). ONE call updates a whole batch in a
-- single UPDATE instead of one HTTP round-trip per row (the write-side speedup).
--
-- Run once in the Supabase SQL editor (project tfbjiyknoagcxjzeoqzw).
-- Only UPDATEs existing rows by id; never inserts or deletes.
--
-- FILL-BLANKS ONLY. Every column keeps whatever is already there and writes the
-- model value only when the column is empty. This is what stops Tier 2 from
-- overwriting the keyword tier (Tier 1) -- coalesce(EXISTING, new), existing wins.
--
-- The camelCase columns MUST stay double-quoted: "jobType", "experienceLevel".

create or replace function apply_6col_batch(updates jsonb)
returns integer
language plpgsql
as $$
declare
  n integer;
begin
  update jobs_joveo_partner_v2 j
  set
    collar            = coalesce(j.collar, u.collar),
    remote_mode       = coalesce(j.remote_mode, u.remote_mode),
    "experienceLevel" = coalesce(j."experienceLevel", u.experience_level),
    "jobType"         = case
                          when j."jobType" is null or j."jobType" = ''
                            then u.job_type
                          else j."jobType"
                        end
  from jsonb_to_recordset(updates) as u(
    id               text,
    collar           text,
    remote_mode      text,
    experience_level text,
    job_type         text
  )
  where j.id = u.id;

  get diagnostics n = row_count;
  return n;
end;
$$;

-- self-test (updates nothing): select apply_6col_batch('[]'::jsonb);  -> 0
