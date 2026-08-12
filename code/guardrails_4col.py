# Guardrail layer for the four new columns, mirroring guardrails.py.
#
# Same design rule as the original: only override when the signal is
# UNAMBIGUOUS. A guardrail that fires on a maybe will make good predictions
# worse, and these heads are weak enough already that a bad rule does real
# damage.
#
# WHAT THESE ARE FOR: the model is trained on LinkedIn postings, which are
# white-collar and professional. The live Joblet feed is ~44% gig-driving plus
# heavy healthcare staffing. Those two slices have systematic, mechanical
# failures the model cannot learn because its training data barely contains
# them. That is exactly the gap guardrails.py was written to close for
# industry/role, and the same gap exists here.
#
# Every rule below is stated as: what fires it, what it fixes, why it is safe.
import re

# ---- 1) gig platforms -> Contract, not Full-time ---------------------------
# Uber/Lyft/DoorDash drivers are independent contractors. The model sees a job
# posting shaped like any other and defaults to Full-time (80% of its training
# data). This is the single largest job_type error on the live feed given
# gig-driving's share of it.
# Safe because: these platforms do not offer employee roles through these posts.
# Their corporate roles carry the company name plus a normal职 title and do not
# match the driving/delivery patterns required below.
GIG_PLATFORM = re.compile(
    r"\b(uber|lyft|doordash|instacart|grubhub|postmates|shipt|amazon\s*flex"
    r"|gopuff|spark\s+driver|favor|roadie)\b", re.I)
GIG_ACTIVITY = re.compile(
    r"\b(driv\w+|deliver\w+|courier|rideshare|ride\s*share|trips?|gig"
    r"|shopper|earn.{0,20}\bper\s+(hour|trip|delivery)\b)\b", re.I)

# ---- 2) healthcare staffing -> Contract / Per Diem -------------------------
# Travel nursing and per-diem shifts are a large share of the feed and are
# NEVER Full-time employment, but read as ordinary clinical postings.
# 'Per Diem' is not in the trained head at all (no LinkedIn category), so
# without this rule that class can never be emitted.
TRAVEL_CONTRACT = re.compile(
    r"\b(travel\s+(nurse|rn|allied|therapist|tech)|travel\s+contract"
    r"|\d{1,2}\s*[- ]?week\s+(contract|assignment)|locum\s*tenens?)\b", re.I)
PER_DIEM = re.compile(r"\b(per\s*diem|perdiem|\bprn\b)\b", re.I)

# ---- 3) explicit job_type in the title -------------------------------------
# When the title states it outright the model should never disagree.
TITLE_JOB_TYPE = [
    (re.compile(r"\b(intern|internship)\b(?!\w)", re.I),           "Internship"),
    (re.compile(r"\bpart[\s-]?time\b", re.I),                      "Part-time"),
    (re.compile(r"\bfull[\s-]?time\b", re.I),                      "Full-time"),
    (re.compile(r"\b(temp|temporary|seasonal)\b(?!\w)", re.I),     "Temporary"),
    (re.compile(r"\b(contract|contractor|1099|c2c|corp[\s-]?to[\s-]?corp)\b", re.I), "Contract"),
]

# ---- 4) explicit seniority in the title ------------------------------------
# Ordered most-senior first: "Senior Director" is Executive-adjacent, so the
# first match must win rather than the last.
TITLE_EXPERIENCE = [
    (re.compile(r"\b(chief|\bc[etofi]o\b|president|partner|founder)\b", re.I), "Executive"),
    (re.compile(r"\b(vp|vice\s+president|head\s+of|director)\b", re.I),        "Senior"),
    (re.compile(r"\b(principal|staff|lead|senior|sr\.?)\b", re.I),             "Senior"),
    (re.compile(r"\b(junior|jr\.?|entry[\s-]?level|trainee|apprentice"
                r"|intern|internship|graduate|new\s+grad)\b", re.I),           "Entry"),
]

# ---- 5) collar: trades and clinical work -----------------------------------
# Matches the product policy in scripts/geo/production/classification.py:182-202.
BLUE_TITLE = re.compile(
    r"\b(driver|driving|cdl|trucker|warehouse|forklift|welder|plumber|electrician"
    r"|mechanic|labou?rer|carpenter|roofer|janitor|custodian|cleaner|housekeep\w*"
    r"|security\s+guard|assembler|machinist|operator|technician|picker|packer"
    r"|landscap\w*|construction|maintenance|cook|chef|server|bartender|barista"
    r"|cashier|stocker|courier|delivery)\b", re.I)
CLINICAL_TITLE = re.compile(
    r"\b(nurse|\brn\b|\blpn\b|\bcna\b|physician|surgeon|therapist|pharmacist"
    r"|medical\s+assistant|dental\s+hygienist|caregiver|caretaker|phlebotom\w*"
    r"|paramedic|\bemt\b|patient\s+care|home\s+health)\b", re.I)

# ---- 6) remote mode stated outright ----------------------------------------
# Narrow on purpose. A bare "remote" matches "remote site" and "remote
# monitoring" and would fire on on-site jobs in rural locations.
REMOTE_TITLE = re.compile(
    r"(fully[\s-]remote|100%\s*remote|work\s+from\s+home|\bwfh\b"
    r"|\(\s*remote\s*\)|\bremote\s*[-–]|[-–]\s*remote\b|telecommut\w*)", re.I)
HYBRID_TITLE = re.compile(r"\bhybrid\b", re.I)
ONSITE_TITLE = re.compile(r"\b(on[\s-]?site|onsite|in[\s-]person|in[\s-]office)\b", re.I)


def apply_4col_guardrails(title, desc, job_type, experience_level, collar, remote_mode):
    """Return (job_type, experience_level, collar, remote_mode, notes).

    `notes` names every rule that fired, so an override is always auditable and
    never silent. Pass the model's predictions in; whatever comes back is what
    should be stored.
    """
    t = title or ""
    text = f"{t}\n{desc or ''}"
    notes = []

    # --- job_type -----------------------------------------------------------
    # Order matters: per-diem beats travel-contract beats gig beats title text.
    if PER_DIEM.search(text):
        job_type, _ = "Per Diem", notes.append("job_type:per_diem")
    elif TRAVEL_CONTRACT.search(text):
        job_type, _ = "Contract", notes.append("job_type:travel_contract")
    elif GIG_PLATFORM.search(t) and GIG_ACTIVITY.search(text):
        # Both required. "Uber" alone could be a corporate software role.
        job_type, _ = "Contract", notes.append("job_type:gig_platform")
    else:
        for rx, val in TITLE_JOB_TYPE:
            if rx.search(t):
                if job_type != val:
                    notes.append(f"job_type:title_states_{val}")
                job_type = val
                break

    # --- experience_level ---------------------------------------------------
    for rx, val in TITLE_EXPERIENCE:
        if rx.search(t):
            if experience_level != val:
                notes.append(f"experience:title_states_{val}")
            experience_level = val
            break

    # --- collar -------------------------------------------------------------
    # Clinical wins over trades: "Surgical Technician" is clinical, and the
    # trades pattern would otherwise claim it on "technician".
    if CLINICAL_TITLE.search(t):
        if collar != "Blue":
            notes.append("collar:clinical_veto")
        collar = "Blue"
    elif BLUE_TITLE.search(t):
        if collar != "Blue":
            notes.append("collar:trade_veto")
        collar = "Blue"

    # --- remote_mode --------------------------------------------------------
    # Only fire on the TITLE. Descriptions routinely mention all three modes in
    # boilerplate ("our hybrid culture", "some roles are remote"), so matching
    # there produces confident nonsense.
    if HYBRID_TITLE.search(t):
        remote_mode, _ = "Hybrid", notes.append("remote:title_hybrid")
    elif REMOTE_TITLE.search(t):
        remote_mode, _ = "Remote", notes.append("remote:title_remote")
    elif ONSITE_TITLE.search(t):
        remote_mode, _ = "On-site", notes.append("remote:title_onsite")

    return job_type, experience_level, collar, remote_mode, ";".join(notes)


if __name__ == "__main__":
    # The cases these rules exist for. Second tuple = what the model tends to
    # say unaided; third = what the guardrail should produce.
    CASES = [
        ("Earn $22/hr Driving With Uber", "flexible hours, drive when you want",
         ("Full-time", "Mid", "White", "On-site"),
         ("Contract", "Mid", "Blue", "On-site")),
        ("Travel ICU RN - 13 Week Contract", "critical care, night shift",
         ("Full-time", "Mid", "White", "On-site"),
         ("Contract", "Mid", "Blue", "On-site")),
        ("Registered Nurse PRN", "as-needed shifts at our medical center",
         ("Full-time", "Mid", "White", "On-site"),
         ("Per Diem", "Mid", "Blue", "On-site")),
        ("Senior Software Engineer (Remote)", "8+ years python",
         ("Full-time", "Mid", "White", "On-site"),
         ("Full-time", "Senior", "White", "Remote")),
        ("Marketing Intern - Hybrid", "summer programme",
         ("Full-time", "Mid", "White", "On-site"),
         ("Internship", "Entry", "White", "Hybrid")),
        ("Chief Financial Officer", "lead the finance organisation",
         ("Full-time", "Senior", "White", "On-site"),
         ("Full-time", "Executive", "White", "On-site")),
        ("CDL-A Truck Driver", "dedicated regional route, home weekly",
         ("Full-time", "Mid", "White", "On-site"),
         ("Full-time", "Mid", "Blue", "On-site")),
    ]
    ok = 0
    for title, desc, model_out, expected in CASES:
        got = apply_4col_guardrails(title, desc, *model_out)
        passed = got[:4] == expected
        ok += passed
        print(f"{'PASS' if passed else 'FAIL'}  {title[:42]:44} -> {got[:4]}")
        if not passed:
            print(f"      expected {expected}")
        if got[4]:
            print(f"      rules: {got[4]}")
    print(f"\n{ok}/{len(CASES)} passed")
