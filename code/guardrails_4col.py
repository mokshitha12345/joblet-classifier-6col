# Guardrail / rule layer for the four new columns.
#
# Design rule (unchanged): only override when the signal is UNAMBIGUOUS. A rule
# that fires on a maybe makes good predictions worse.
#
# WHAT CHANGED (v2, after the 5,000-row live test):
#   * collar: added a WHITE-collar title rule. The model called office jobs
#     ("Manufacturing Engineer", "Demand Planner", "Project Manager") Blue.
#     A clear white-collar title now forces White. Order: clinical -> white -> blue.
#   * job_type: added CDD (French fixed-term) and strengthened travel/contract.
#   * experience: added a "X years experience" parser. When the title has no
#     seniority word, the description's stated years now sets the level.
#   * remote_mode: now RULE-BASED, not model-trusting. The model over-predicted
#     "Remote" (87%). We ignore its guess: explicit remote/hybrid text wins,
#     physical/clinical roles are On-site, and everything else defaults On-site
#     (most jobs are). Measured ~90% vs the model's ~50%.
import re

# ---- job_type signals ------------------------------------------------------
GIG_PLATFORM = re.compile(
    r"\b(uber|lyft|doordash|instacart|grubhub|postmates|shipt|amazon\s*flex"
    r"|gopuff|spark\s+driver|favor|roadie)\b", re.I)
GIG_ACTIVITY = re.compile(
    r"\b(driv\w+|deliver\w+|courier|rideshare|ride\s*share|trips?|gig"
    r"|shopper|earn.{0,20}\bper\s+(hour|trip|delivery)\b)\b", re.I)
TRAVEL_CONTRACT = re.compile(
    r"\b(travel\s+(nurse|rn|allied|therap\w*|tech\w*|radiolog\w*|sonograph\w*"
    r"|contract|assignment|position)"
    r"|\d{1,2}\s*[- ]?week\s+(travel\s+)?(contract|assignment|position)"
    r"|locum\s*tenens?|\bcdd\b|contrat\s+.{0,6}dur.e\s+d.termin)\b", re.I)
PER_DIEM = re.compile(r"\b(per\s*diem|perdiem|\bprn\b)\b", re.I)
TITLE_JOB_TYPE = [
    (re.compile(r"\b(intern|internship|praktikant)\b(?!\w)", re.I),        "Internship"),
    (re.compile(r"\bpart[\s-]?time\b", re.I),                              "Part-time"),
    (re.compile(r"\bfull[\s-]?time\b", re.I),                              "Full-time"),
    (re.compile(r"\b(temp|temporary|seasonal)\b(?!\w)", re.I),             "Temporary"),
    (re.compile(r"\b(contractor|1099|c2c|corp[\s-]?to[\s-]?corp)\b", re.I), "Contract"),
    # bare "contract" only as an employment type, NOT as a subject-matter word
    # ("Contract Manager", "Contracts Administrator" are permanent roles).
    (re.compile(r"\bcontract\b(?!\s*(manager|management|administrat|admin|"
                r"specialist|analyst|coordinator|attorney|negotiat|officer|"
                r"lead|director|counsel|clerk|assistant))", re.I),        "Contract"),
]

# ---- experience signals ----------------------------------------------------
# Order matters and is checked top-down. VP/vice-president is Senior and MUST be
# tested before the Executive "president" row (else "Vice President" -> Executive).
# "partner" (Delivery/Sales Partner = gig) and "staff" (Staff Accountant/Nurse =
# entry/mid) were removed -- they misfired on common titles.
TITLE_EXPERIENCE = [
    (re.compile(r"\b(vp|vice\s+president|svp|evp|head\s+of|director)\b", re.I),   "Senior"),
    (re.compile(r"\b(chief|\bc[etofi]o\b|founder|president)\b", re.I),            "Executive"),
    (re.compile(r"\b(principal|lead|senior|\bsr\.?\b)\b", re.I),                  "Senior"),
    (re.compile(r"\b(junior|jr\.?|entry[\s-]?level|trainee|apprentice"
                r"|intern|internship|graduate|new\s+grad)\b", re.I),             "Entry"),
]
# Require the number to sit next to "experience" -- otherwise "18 years of age",
# "1 year contract", "vest after 1 year" get misread as seniority. Take the
# MINIMUM stated experience (the entry bar).
YEARS_RE = re.compile(
    r"(\d{1,2})\s*\+?\s*(?:to\s*\d{1,2}\s*)?years?"
    r"(?:['’]?\s*(?:of\s+)?(?:relevant\s+|related\s+|professional\s+|work(?:ing)?\s+"
    r"|industry\s+|hands[\s-]on\s+|prior\s+|direct\s+)*(?:experience|exp\b))", re.I)
# Explicit "anyone can do this" signals -> Entry (survey/gig posts, trainee roles).
NO_EXP = re.compile(
    r"\b(no\s+(prior\s+)?experience(\s+(needed|required|necessary))?"
    r"|entry[\s-]?level|will\s+train|no\s+degree\s+required|beginners?\s+welcome)\b", re.I)

# ---- collar signals --------------------------------------------------------
CLINICAL_TITLE = re.compile(
    r"\b(nurse|\brn\b|\blpn\b|\bcna\b|physician|surgeon|therapist|pharmacist"
    r"|medical\s+assistant|dental\s+hygienist|hygienist|radiolog\w*|sonograph\w*"
    r"|caregiver|caretaker|phlebotom\w*|paramedic|\bemt\b|patient\s+care|home\s+health)\b", re.I)
# White-collar/professional titles. Clear office roles that the model wrongly
# called Blue. Kept high-precision: no ambiguous words (coordinator/specialist).
WHITE_TITLE = re.compile(
    r"\b(engineer|developer|programmer|architect|analyst|scientist|accountant"
    r"|bookkeeper|auditor|actuary|underwriter|controller|economist|consultant"
    r"|attorney|lawyer|paralegal|counsel|manager|director|planner|designer"
    r"|strategist|recruiter|marketer|financial\s+advisor)\b", re.I)
BLUE_TITLE = re.compile(
    r"\b(driver|driving|cdl|trucker|warehouse|forklift|welder|plumber|electrician"
    r"|mechanic|labou?rer|carpenter|roofer|janitor|custodian|cleaner|housekeep\w*"
    r"|security\s+guard|assembler|machinist|operator|technician|picker|packer"
    r"|landscap\w*|construction|maintenance|cook|chef|server|bartender|barista"
    r"|cashier|stocker|courier|delivery)\b", re.I)

# ---- remote_mode signals ---------------------------------------------------
REMOTE_TEXT = re.compile(
    r"\b(fully[\s-]remote|100%\s*remote|work\s+from\s+home|\bwfh\b|remote\s+position"
    r"|remote\s+role|work\s+remotely|remote[\s-]first|telecommut\w*"
    r"|from\s+the\s+comfort\s+of\s+your\s+home|distributed\s+team)\b", re.I)
HYBRID_TEXT = re.compile(r"\bhybrid\b", re.I)
ONSITE_TEXT = re.compile(r"\b(on[\s-]?site|onsite|in[\s-]person|in\s+office|in[\s-]office)\b", re.I)
PHYSICAL = re.compile(
    r"\b(nurse|\brn\b|\bcna\b|\blpn\b|dental|hygienist|radiolog\w*|\bmri\b|\bct\b"
    r"|surgical|surgeon|therapist|patient|clinic|hospital|bedside|phlebotom\w*|paramedic"
    r"|driver|truck|\bcdl\b|warehouse|forklift|welder|plumber|electrician|mechanic"
    r"|carpenter|roofer|janitor|custodian|cleaner|housekeep\w*|cook|chef|server"
    r"|waiter|cashier|barista|stocker|picker|packer|construction|landscap\w*"
    r"|retail|store\s+associate|teacher|classroom|caregiver|security\s+guard)\b", re.I)


def _years_level(desc):
    """Return Entry/Mid/Senior from stated years OF EXPERIENCE, or None.

    YEARS_RE already requires the number to be followed by 'experience', so
    ages/tenures/contract-durations don't reach here. Take the minimum stated
    (the entry bar)."""
    best = None
    for m in YEARS_RE.finditer(desc):
        y = int(m.group(1))
        best = y if best is None else min(best, y)
    if best is None:
        return None
    if best <= 2:
        return "Entry"
    if best <= 5:
        return "Mid"
    return "Senior"


def apply_4col_guardrails(title, desc, job_type, experience_level, collar, remote_mode):
    """Return (job_type, experience_level, collar, remote_mode, notes)."""
    t = title or ""
    d = desc or ""
    text = f"{t}\n{d}"
    notes = []

    # --- job_type: per-diem > travel/contract > gig > explicit title ---------
    if PER_DIEM.search(text):
        job_type = "Per Diem"; notes.append("job_type:per_diem")
    elif TRAVEL_CONTRACT.search(text):
        job_type = "Contract"; notes.append("job_type:travel_contract")
    elif GIG_PLATFORM.search(t) and GIG_ACTIVITY.search(text):
        job_type = "Contract"; notes.append("job_type:gig_platform")
    else:
        for rx, val in TITLE_JOB_TYPE:
            if rx.search(t):
                if job_type != val:
                    notes.append(f"job_type:title_{val}")
                job_type = val
                break

    # --- experience: title seniority first, then stated years ----------------
    hit = False
    for rx, val in TITLE_EXPERIENCE:
        if rx.search(t):
            if experience_level != val:
                notes.append(f"experience:title_{val}")
            experience_level = val
            hit = True
            break
    if not hit:
        yl = _years_level(d)
        if yl:
            if experience_level != yl:
                notes.append(f"experience:years_{yl}")
            experience_level = yl
        elif NO_EXP.search(text):
            if experience_level != "Entry":
                notes.append("experience:no_experience")
            experience_level = "Entry"

    # --- collar: clinical -> white -> blue -----------------------------------
    if CLINICAL_TITLE.search(t):
        if collar != "Blue":
            notes.append("collar:clinical_veto")
        collar = "Blue"
    elif WHITE_TITLE.search(t):
        if collar != "White":
            notes.append("collar:white_title")
        collar = "White"
    elif BLUE_TITLE.search(t):
        if collar != "Blue":
            notes.append("collar:trade_veto")
        collar = "Blue"

    # --- remote_mode: pure rules; ignore the (broken) model prediction --------
    if HYBRID_TEXT.search(text):
        new_remote = "Hybrid"
    elif REMOTE_TEXT.search(text) and not PHYSICAL.search(t):
        new_remote = "Remote"
    elif ONSITE_TEXT.search(text) or PHYSICAL.search(t):
        new_remote = "On-site"
    else:
        new_remote = "On-site"   # default: most jobs are on-site
    if new_remote != remote_mode:
        notes.append(f"remote:rule_{new_remote}")
    remote_mode = new_remote

    return job_type, experience_level, collar, remote_mode, ";".join(notes)


if __name__ == "__main__":
    CASES = [
        ("Manufacturing Engineer", "advanced manufacturing engineering role",
         ("Full-time", "Mid", "Blue", "Remote"), ("Full-time", "Mid", "White", "On-site")),
        ("Travel Radiologic Technologist", "13-week travel assignment at a facility",
         ("Full-time", "Entry", "Blue", "Remote"), ("Contract", "Entry", "Blue", "On-site")),
        ("Registered Nurse PRN", "as-needed shifts at our medical center",
         ("Full-time", "Mid", "White", "Remote"), ("Per Diem", "Mid", "Blue", "On-site")),
        ("Senior Software Engineer (Remote)", "8+ years python, fully remote",
         ("Full-time", "Mid", "White", "On-site"), ("Full-time", "Senior", "White", "Remote")),
        ("Software Developer", "we require 3 years of experience building web apps",
         ("Full-time", "Entry", "White", "Remote"), ("Full-time", "Mid", "White", "On-site")),
        ("Demand Planner", "supply chain planning role, hybrid schedule",
         ("Full-time", "Mid", "Blue", "Remote"), ("Full-time", "Mid", "White", "Hybrid")),
        # --- regressions caught by the adversarial review ---
        ("Vice President, Marketing", "lead the marketing org",
         ("Full-time", "Mid", "White", "Remote"), ("Full-time", "Senior", "White", "On-site")),
        ("Delivery Partner", "drive and deliver with our platform, flexible",
         ("Full-time", "Mid", "White", "Remote"), ("Full-time", "Mid", "Blue", "On-site")),
        ("Staff Accountant", "prepare journal entries, no experience needed",
         ("Full-time", "Senior", "White", "Remote"), ("Full-time", "Entry", "White", "On-site")),
        ("Contract Manager", "oversee vendor contracts, permanent salaried role",
         ("Full-time", "Mid", "White", "Remote"), ("Full-time", "Mid", "White", "On-site")),
        ("Cashier", "must be at least 18 years of age, cash handling",
         ("Full-time", "Mid", "White", "Remote"), ("Full-time", "Mid", "Blue", "On-site")),
    ]
    ok = 0
    for title, desc, model_out, expected in CASES:
        got = apply_4col_guardrails(title, desc, *model_out)
        passed = got[:4] == expected
        ok += passed
        print(f"{'PASS' if passed else 'FAIL'}  {title[:40]:42} -> {got[:4]}")
        if not passed:
            print(f"      expected {expected}")
        if got[4]:
            print(f"      rules: {got[4]}")
    print(f"\n{ok}/{len(CASES)} passed")
