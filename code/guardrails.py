# Guardrail layer: high-precision rules applied AFTER the model, targeting the
# systematic REAL-WORLD failures measured on held-out Joblet feed jobs.
#
# Design rule: only override when a title pattern is UNAMBIGUOUS. Otherwise leave
# the model's prediction untouched — rules must never make good predictions worse.
import re

# ---- 1) junk / paid-participation detector ----------------------------------
# These are paid-research/survey participation offers, not jobs. The model is
# misled by stray domain words ("Medical Surveys" -> Healthcare Professional).
#
# CRITICAL DESIGN NOTE (learned from a review that found real breakage):
# money in a title is NOT a junk signal. Real jobs advertise pay --
#   "Earn up to $100,000 as a Real Estate Agent"   <- real job
#   "Earn $30/hour as a Caregiver"                 <- real job
#   "Earn $22/hr Driving With Uber"                <- real gig job
# An earlier version fired on bare "earn $..." and "no experience needed" and
# destroyed all of the above. So we now require an EXPLICIT participation marker;
# money and "no experience" are never sufficient on their own.
JUNK = re.compile(
    r"(survey\s*taker|paid\s+(for\s+)?surveys?|for\s+surveys?\b"
    r"|participat\w*\s+in\s+[^.]{0,30}?(survey|stud(y|ies)|trial|research)"
    r"|\b(online|medical|consumer|paid)\s+surveys?\b"
    r"|focus\s+group|research\s+(panelist|panellist|volunteer|participant)"
    r"|stud(y|ies)\s+participant|clinical\s+trial\s+(participant|volunteer)"
    r"|product\s+feedback\s+participant|market\s+research\s+participant"
    r"|\btask\s+reviewer\b|cashapp)", re.I)
# Even with a participation marker, never junk a clearly-named profession.
JUNK_EXEMPT = re.compile(
    r"\b(nurse|rn|lpn|cna|driver|driving|drive|uber|lyft|doordash|instacart|courier"
    r"|delivery|trips?|cdl|engineer|developer|teacher|therapist|technician|assistant"
    r"|manager|analyst|specialist|coordinator|agent|caregiver|associate|clerk"
    r"|representative|supervisor|scientist|researcher)\b", re.I)

# ---- 2) specific title traps ------------------------------------------------
TRAPS = [
    # (title pattern, industry, role)  -- None = leave model's answer
    (re.compile(r"\bdata\s+entry\b", re.I),                    "Business Services & Consulting", "Others"),
    (re.compile(r"\bnon.?destructive\s+test|(\bndt\b)", re.I),  "Manufacturing & Industrial",     "Others"),
    (re.compile(r"\b(cyber\s*security|information\s+security|infosec|security\s+(architect|engineer|analyst))\b", re.I),
                                                                "Technology & IT",                "Cybersecurity Specialist"),
]

# ---- 3) "Director/VP/Head of <function>" -> route by the FUNCTION ------------
# Fixes the classic black hole where any leadership title -> generic bucket.
# Must tolerate punctuation variants: "Director of Finance", "Director, Finance",
# "Director - Finance", "VP, Marketing", "Head - Nursing".
LEADER = re.compile(r"\b(director|vp|vice\s+president|head)\b\s*[,:–—-]?\s*(of\s+)?", re.I)
FUNCTION_MAP = [
    (re.compile(r"\b(rehab|nursing|clinical|patient|medical|health)\b", re.I),
     "Healthcare & Caregiving", "Healthcare Professional"),
    (re.compile(r"\b(engineering|software|technology|\bit\b)\b", re.I),
     "Technology & IT", None),
    (re.compile(r"\b(finance|accounting|tax)\b", re.I),
     "Finance & Banking", "Accountant & Finance"),
    (re.compile(r"\b(marketing|brand)\b", re.I),
     "Marketing & Media", "Marketing Specialist"),
]

def apply_guardrails(title, industry, role, ind_conf=1.0, role_conf=1.0):
    """Returns (industry, role, reason) after high-precision overrides."""
    t = str(title or "")

    # 1) junk/gig spam -> Others (unless it names a real profession)
    if JUNK.search(t) and not JUNK_EXEMPT.search(t):
        return "Business Services & Consulting", "Others", "junk-gig"

    # 2) unambiguous title traps
    for rx, ind, rl in TRAPS:
        if rx.search(t):
            return (ind or industry), (rl or role), "trap"

    # 3) leadership title -> route by function
    if LEADER.search(t):
        for rx, ind, rl in FUNCTION_MAP:
            if rx.search(t):
                return (ind or industry), (rl or role), "leader-function"

    return industry, role, ""

def should_dual(top1_role, conf, threshold):
    """DUAL only when genuinely torn between two REAL categories.
    If top-1 is 'Others', low confidence means 'junk/unclassifiable', not
    'ambiguous between two valid categories' -> a 2nd label would just be noise."""
    if conf >= threshold:
        return False
    if top1_role == "Others":
        return False          # junk signal, not ambiguity
    return True
