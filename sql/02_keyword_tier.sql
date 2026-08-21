-- ============================================================================
-- TIER 1 — keyword pass, all 6 columns, in the 25-role / 15-industry taxonomy.
-- role + industry use the SAME rules the ML model was trained on, so the two
-- tiers are consistent. runs in Postgres, chunked by pg_cron, no timeout.
--
-- role/industry: OVERWRITE when a keyword matches (fixes the old mixed vocab to
--                25/15); keep existing when nothing matches.
-- collar/remote/experience: fill-blanks. jobType: fill only when blank.
-- ============================================================================

create table if not exists classify_state (
  tier text primary key, after_id text not null default '', updated_at timestamptz default now()
);
insert into classify_state(tier) values ('keyword') on conflict do nothing;

create or replace function keyword_classify_chunk(p_after text, p_limit int default 20000)
returns text language plpgsql as $$
declare last_id text;
begin
  with batch as (
    select id, lower(coalesce(title,'')) as t
    from jobs_joveo_partner_v2
    where id > p_after order by id limit p_limit
  ),
  calc as (
    select id, t,
      -- ROLE (25 taxonomy) --------------------------------------------------
      case
        when t ~ 'data scien' then 'Data Scientist'
        when t ~ '(data engineer|\yetl\y|big data)' then 'Data Engineer'
        when t ~ '(machine learning|ml engineer|ai engineer|deep learning)' then 'Machine Learning Engineer'
        when t ~ '(cybersecur|information security|security engineer|security analyst|infosec|penetration|soc analyst)' then 'Cybersecurity Specialist'
        when t ~ '(devops|\ysre\y|site reliability|platform engineer)' then 'DevOps Engineer'
        when t ~ '(\yqa\y|quality assurance|test engineer|\ytester\y|\ysdet\y)' then 'QA Engineer'
        when t ~ '(it support|help desk|service desk|desktop support|sysadmin|system administrator|network engineer|information technology|it specialist|it technician)' then 'IT Specialist'
        when t ~ '(nurse|nursing|\yrn\y|\ylpn\y|\ycna\y|physician|surgeon|\ydoctor\y|dentist|dental|hygienist|therapist|respiratory|pharmac|veterinar|paramedic|\yemt\y|phleboto|radiolog|caregiv|hospice|midwif|medical|clinician|clinical|patient care)' then 'Healthcare Professional'
        when t ~ '(attorney|lawyer|paralegal|litigation|\ycounsel\y|law clerk|legal)' then 'Legal Professional'
        when t ~ '(teacher|professor|lecturer|tutor|instructor|educat|faculty|adjunct|childcare|preschool|teaching|curriculum|librarian)' then 'Teacher & Educator'
        when t ~ '(driver|truck|\ycdl\y|courier|delivery|freight|dispatch|warehouse|fleet|hauler|forklift|supply chain|logistic)' then 'Driver & Logistics'
        when t ~ '(construction|contractor|electrician|plumber|carpenter|\yhvac\y|welder|mason|roofer|real estate|realtor|property manag|leasing|superintendent|surveyor|estimator)' then 'Construction & Real Estate'
        when t ~ '(accountant|accounting|bookkeep|auditor|\ytax\y|payroll|controller|treasur|underwrit|\yloan\y|mortgage|financial analyst|finance|financial|banker|banking|teller|actuar)' then 'Accountant & Finance'
        when t ~ '(human resources|\yhr\y|recruit|talent acquisition|staffing)' then 'HR Specialist'
        when t ~ '(project manager|program manager|scrum master)' then 'Project Manager'
        when t ~ '(product manager|product owner)' then 'Product Manager'
        when t ~ '(consultant|advisory)' then 'Consultant'
        when t ~ '(customer service|call center|customer support|receptionist|\ycsr\y|client service|customer care)' then 'Customer Service Representative'
        when t ~ '(marketing|advertis|\yseo\y|social media|public relations|copywriter|brand manager|communications|market research)' then 'Marketing Specialist'
        when t ~ '(designer|graphic|\yux\y|\yui\y|illustrat|photograph|videograph|animat|art director|video editor)' then 'Designer'
        when t ~ '(sales|retail|store manager|store associate|merchandis|cashier|account executive|business development|account manager)' then 'Sales Representative'
        when t ~ '(data analy|business analyst|business intelligence|reporting analyst|systems analyst|\yanalyst\y)' then 'Data Analyst'
        when t ~ '(software|developer|programmer|full.?stack|front.?end|back.?end|web develop|mobile develop|database|\ydba\y|cloud|solutions architect)' then 'Software Developer'
        when t ~ '(\yceo\y|\ycfo\y|\ycoo\y|\ycto\y|chief|president|general manager|managing director|executive director|vice president|\yvp\y)' then 'Executive'
        else null end as new_role,
      -- INDUSTRY (15 taxonomy) ----------------------------------------------
      case
        when t ~ '(nurse|nursing|physician|medical|clinical|dental|therapist|caregiv|pharmac|veterinar|hospice|patient|health|radiolog|surgeon|paramedic)' then 'Healthcare & Caregiving'
        when t ~ '(driver|truck|\ycdl\y|delivery|logistic|warehouse|dispatch|freight|courier|fleet|forklift|supply chain|shipping)' then 'Logistics & Transportation'
        when t ~ '(attorney|lawyer|paralegal|legal|litigation|police|firefighter|correction|public safety|court)' then 'Legal & Government'
        when t ~ '(teacher|professor|lecturer|tutor|instructor|educat|faculty|school|childcare)' then 'Education & Research'
        when t ~ '(accountant|accounting|bookkeep|auditor|\ytax\y|finance|financial|bank|teller|loan|mortgage|underwrit|actuar|investment|payroll)' then 'Finance & Banking'
        when t ~ '(software|developer|programmer|devops|cybersecur|\yit\y|information technology|data|network|sysadmin|cloud|computer)' then 'Technology & IT'
        when t ~ '(construction|electrician|plumber|carpenter|\yhvac\y|welder|mason|roofer|contractor|real estate|realtor|property|civil|structural|surveyor)' then 'Construction & Real Estate'
        when t ~ '(manufactur|production|assembler|machine operator|plant|fabricat|\ycnc\y|industrial|inspector|mechanic|millwright|machinist)' then 'Manufacturing & Industrial'
        when t ~ '(cook|chef|server|bartender|barista|hotel|restaurant|hospitality|housekeep|food|banquet|culinary|waiter|lodging|tourism)' then 'Hospitality & Travel'
        when t ~ '(marketing|advertis|\yseo\y|social media|public relations|copywriter|brand|journalist|editor|media|communications|broadcast)' then 'Marketing & Media'
        when t ~ '(designer|graphic|\yux\y|creative|illustrat|photograph|videograph|animat|\yart\y|musician|artist)' then 'Creative & Design'
        when t ~ '(energy|\ypower\y|solar|\yoil\y|\ygas\y|petroleum|utility|lineman|drilling|refinery|pipeline|mining)' then 'Energy & Utilities'
        when t ~ '(farm|agricultur|\ycrop\y|livestock|forestry|fisher|landscap|greenhouse|harvest|ranch)' then 'Agriculture & Primary'
        when t ~ '(retail|cashier|store|sales associate|merchandis|stocker)' then 'Retail & E-commerce'
        else null end as new_ind,
      -- COLLAR: clinical -> white -> blue (same precedence as guardrails_4col).
      -- The white row must come BEFORE the trades row so "Warehouse Manager" /
      -- "Construction Project Manager" resolve to White, not Blue.
      case
        when t ~ '\y(nurse|rn|lpn|cna|physician|surgeon|therapist|pharmacist|paramedic|emt|caregiver|phlebotom|patient care|home health|medical assistant|dental hygienist)\y' then 'Blue'
        when t ~ '\y(engineer|developer|programmer|architect|analyst|scientist|accountant|bookkeeper|auditor|actuary|underwriter|controller|economist|consultant|attorney|lawyer|paralegal|counsel|manager|director|planner|designer|strategist|recruiter|marketer)\y' then 'White'
        when t ~ '\y(driver|driving|cdl|trucker|warehouse|forklift|welder|plumber|electrician|mechanic|labou?rer|carpenter|roofer|janitor|custodian|cleaner|housekeep|security guard|assembler|machinist|picker|packer|construction|maintenance|cook|chef|server|bartender|barista|cashier|stocker|courier|delivery|landscap)\y' then 'Blue'
        else null end as new_collar,
      -- REMOTE_MODE ---------------------------------------------------------
      case
        when t ~ '(fully[ -]remote|100% *remote|work from home|\ywfh\y|\( *remote *\)|telecommut)' then 'Remote'
        when t ~ '\yhybrid\y' then 'Hybrid'
        when t ~ '\y(on[ -]?site|onsite|in[ -]person|in[ -]office)\y' then 'On-site'
        when t ~ '\y(nurse|rn|lpn|cna|physician|surgeon|therapist|pharmacist|paramedic|caregiver|driver|driving|cdl|trucker|warehouse|forklift|welder|plumber|electrician|mechanic|labou?rer|carpenter|roofer|janitor|custodian|cleaner|housekeep|security guard|assembler|machinist|picker|packer|construction|maintenance|cook|chef|server|bartender|barista|cashier|stocker|courier|delivery|landscap)\y' then 'On-site'
        else null end as new_remote,
      -- JOBTYPE -------------------------------------------------------------
      case
        when t ~ '\y(per diem|perdiem|prn)\y' then 'Per Diem'
        when t ~ '\y(travel (nurse|rn|allied|therapist|tech)|locum)\y' or t ~ '\y\d{1,2} *[- ]?week (contract|assignment)\y' then 'Contract'
        when t ~ '\y(uber|lyft|doordash|instacart|grubhub|postmates|shipt|amazon *flex|gopuff|spark driver|roadie)\y' and t ~ '\y(driv|deliver|courier|rideshare|shopper|gig)\y' then 'Contract'
        when t ~ '\yinternship?\y' then 'Internship'
        when t ~ '\ypart[ -]?time\y' then 'Part-time'
        when t ~ '\yfull[ -]?time\y' then 'Full-time'
        when t ~ '\y(temp|temporary|seasonal)\y' then 'Temporary'
        when t ~ '\y(contract|contractor|1099)\y' then 'Contract'
        else null end as new_jobtype,
      -- EXPERIENCELEVEL -----------------------------------------------------
      case
        when t ~ '\y(chief|c[etofi]o|president|founder|partner)\y' then 'Executive'
        when t ~ '\y(vp|vice president|head of|director)\y' then 'Senior'
        when t ~ '\y(principal|staff|lead|senior|sr\.?)\y' then 'Senior'
        when t ~ '\y(junior|jr\.?|entry[ -]?level|trainee|apprentice|intern|graduate|new grad)\y' then 'Entry'
        else null end as new_exp
    from batch
  ),
  did as (
    update jobs_joveo_partner_v2 j set
      standard_role            = coalesce(c.new_role, j.standard_role),
      category_name            = coalesce(c.new_ind,  j.category_name),
      standard_role_method     = case when c.new_role is not null then 'keyword_25v1' else j.standard_role_method end,
      standard_role_confidence = case when c.new_role is not null then 0.95 else j.standard_role_confidence end,
      standard_role_updated_at = case when c.new_role is not null then now() else j.standard_role_updated_at end,
      collar            = coalesce(j.collar, c.new_collar),
      remote_mode       = coalesce(j.remote_mode, c.new_remote),
      "jobType"         = case when j."jobType" is null or j."jobType" = '' then c.new_jobtype else j."jobType" end,
      "experienceLevel" = coalesce(j."experienceLevel", c.new_exp)
    from calc c where j.id = c.id
  )
  select max(id) into last_id from batch;
  return last_id;
end; $$;

create or replace function keyword_classify_tick(p_limit int default 20000)
returns text language plpgsql as $$
declare cur text; last text;
begin
  select after_id into cur from classify_state where tier='keyword';
  last := keyword_classify_chunk(cur, p_limit);
  if last is null then perform cron.unschedule('keyword_classify'); return 'DONE'; end if;
  update classify_state set after_id=last, updated_at=now() where tier='keyword';
  return last;
end; $$;
