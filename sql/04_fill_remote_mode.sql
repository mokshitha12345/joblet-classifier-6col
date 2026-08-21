-- ============================================================================
-- FIX 1 — remote_mode by RULES (no ML, no GPU).
--
-- The ML remote_mode head is broken (labels 87% of jobs "Remote", including
-- nurses and radiology techs). This fills remote_mode for the whole table using
-- deterministic signals that are ~90% accurate:
--
--   1. feed `remote` flag = true                       -> Remote   (employer said so)
--   2. text says fully-remote / wfh / telecommute      -> Remote
--   3. text says hybrid                                -> Hybrid
--   4. everything else                                 -> On-site  (most jobs are)
--
-- Runs chunked via pg_cron so it never times out on the 800k+ table. It walks
-- the not-yet-filled rows and stops itself when done. Only fills NULLs, so it is
-- safe to run alongside the keyword tier and re-runnable.
-- ============================================================================

create or replace function fill_remote_mode_chunk(p_limit int default 30000)
returns integer language plpgsql as $$
declare n integer;
begin
  with batch as (
    select id from jobs_joveo_partner_v2
    where remote_mode is null
    order by id limit p_limit
  ),
  upd as (
    update jobs_joveo_partner_v2 j set remote_mode = case
      when j."remote" is true then 'Remote'
      when lower(coalesce(j.title,'') || ' ' || coalesce(j.description,''))
           ~ '(fully[ -]remote|100% *remote|work from home|\ywfh\y|remote position|work remotely|telecommut|remote[ -]first)'
        then 'Remote'
      when lower(coalesce(j.title,'') || ' ' || coalesce(j.description,'')) ~ '\yhybrid\y'
        then 'Hybrid'
      else 'On-site'
    end
    from batch b where j.id = b.id
    returning 1
  )
  select count(*) into n from upd;
  return n;
end; $$;

-- ---- run it (pick ONE) -----------------------------------------------------
--
-- A) Test one chunk first, look at the result:
--      select fill_remote_mode_chunk(5000);
--      select remote_mode, count(*) from jobs_joveo_partner_v2
--        where remote_mode is not null group by 1 order by 2 desc;
--
-- B) Drain the whole table on a schedule (30k/min, self-stops):
--      select cron.schedule('fill_remote_mode', '* * * * *',
--        $$ do $b$ begin
--             if fill_remote_mode_chunk(30000) = 0 then
--               perform cron.unschedule('fill_remote_mode');
--             end if;
--           end $b$; $$);
--
--    stop early:  select cron.unschedule('fill_remote_mode');
