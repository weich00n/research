"""Singapore fertility policy instruments (CLAUDE.md Policy Scenarios).

All policies are grounded in real Singapore instruments. `expected_pathways`
records which TPB constructs each policy is hypothesised to move — used for
mechanism-validity checks, never fed to agents as an instruction.

**Two eras, deliberately kept apart.** `POLICIES` holds the eight instruments
as they stood before the National Day Rally of August 2026. They are the
*validated core*: the Marriage & Parenthood Survey 2021 reports resident
priorities over exactly this kind of instrument, so simulated effects on them
can be checked against real survey evidence. `NDR2026_POLICIES` holds the four
instruments announced at NDR 2026 (effective 2027 onward). No attitudinal or
uptake data exists for those yet, so they are **opt-in** and reported
separately — never silently mixed into the validated core.

Keeping them in separate lists is also load-bearing for reproducibility:
`news.build_news_schedule` round-robins over `get_policies(category)` in list
order, so appending to `POLICIES` would change which policy lands in which week
for every existing C2/C3 run. `get_policies()` therefore returns the original
eight, in the original order, unless `include_ndr2026=True` is passed.
"""

FINANCIAL = "financial"
CAREGIVING = "caregiving"
HOUSING = "housing"  # new with NDR 2026; no pre-2026 instrument in this study

PRE_NDR2026 = "pre_ndr2026"
NDR2026 = "ndr2026"


class Policy:
    """One real Singapore fertility-support instrument.

    `expected_pathways` is the researcher's hypothesis about which TPB
    construct(s) this policy should move (a subset of attitude/norm/pbc). It is
    metadata for analysis only — it is never shown to the agent or the LLM.

    `era` marks which policy generation the instrument belongs to
    (`pre_ndr2026` = the survey-validated core, `ndr2026` = announced at the
    National Day Rally of Aug 2026). Analysis code filters on it so NDR results
    are reported apart from the validated core; agents never see it.
    """

    def __init__(self, name, category, description, expected_pathways,
                 era=PRE_NDR2026):
        self.name = name
        self.category = category
        self.description = description
        self.expected_pathways = expected_pathways  # subset of {"attitude", "norm", "pbc"}
        self.era = era

    def to_dict(self):
        return {
            "name": self.name,
            "category": self.category,
            "description": self.description,
            "expected_pathways": self.expected_pathways,
            "era": self.era,
        }

    def __repr__(self):
        return f"Policy({self.name!r}, {self.category})"


POLICIES = [
    # ── Financial ──────────────────────────────────────────────────────────
    Policy(
        name="Baby Bonus & Child Development Account",
        category=FINANCIAL,
        description=(
            "The Government provides a Baby Bonus cash gift of up to S$11,000 per "
            "child, plus dollar-for-dollar matching in the Child Development Account "
            "(CDA) usable for childcare, preschool, and healthcare expenses."
        ),
        expected_pathways=["pbc"],
    ),
    Policy(
        name="Large Family Scheme",
        category=FINANCIAL,
        description=(
            "Families receive up to S$16,000 of additional support for each third or "
            "subsequent child: an increased S$10,000 CDA First Step Grant, a S$5,000 "
            "MediSave grant for the mother, and S$1,000 per year in LifeSG credits "
            "per child until age 6."
        ),
        expected_pathways=["pbc"],
    ),
    Policy(
        name="Child LifeSG Credits",
        category=FINANCIAL,
        description=(
            "Parents receive Child LifeSG credits for every child aged 0 to 12, "
            "usable for daily household and child-raising expenses."
        ),
        expected_pathways=["pbc"],
    ),
    # ── Caregiving ─────────────────────────────────────────────────────────
    Policy(
        name="Enhanced Paternity Leave",
        category=CAREGIVING,
        description=(
            "Government-paid paternity leave is doubled from 2 to 4 weeks, with an "
            "additional 2 weeks made mandatory for employers from April 2025, so "
            "fathers can share early infant care."
        ),
        expected_pathways=["pbc", "attitude"],
    ),
    Policy(
        name="Shared Parental Leave",
        category=CAREGIVING,
        description=(
            "A new Shared Parental Leave scheme gives parents up to 10 weeks of "
            "additional government-paid leave to share between mother and father — "
            "6 weeks for children born from April 2025, rising to 10 weeks from "
            "April 2026."
        ),
        expected_pathways=["pbc", "attitude"],
    ),
    Policy(
        name="Flexible Work Arrangement Request Guidelines",
        category=CAREGIVING,
        description=(
            "Tripartite Guidelines require all employers to fairly consider formal "
            "requests for flexible work arrangements such as flexi-place, flexi-time, "
            "and flexi-load."
        ),
        expected_pathways=["pbc", "attitude"],
    ),
    Policy(
        name="Preschool & Infant Care Subsidies",
        category=CAREGIVING,
        description=(
            "Preschool and infant care subsidies are enhanced and fee caps lowered at "
            "government-supported centres, making full-day childcare more affordable "
            "for working parents."
        ),
        expected_pathways=["pbc"],
    ),
    Policy(
        name="Infant Childminding Pilot",
        category=CAREGIVING,
        description=(
            "A government-supported childminding pilot offers home-based care for "
            "infants aged 2 to 18 months, adding a flexible alternative to "
            "centre-based infant care."
        ),
        expected_pathways=["pbc"],
    ),
]


# ── NDR 2026 (announced 17 Aug 2026, effective 2027 onward) ────────────────
# Opt-in only — see the module docstring. Facts pinned to the National Population
# and Talent Division summary of the Rally's marriage & parenthood measures
# (population.gov.sg). Measures still described by the Government as "under
# study" or "under review" (Large Family MediSave Grant repositioning, transport
# assistance, further large-family housing support) are deliberately excluded:
# only firm, dated instruments are modelled.
NDR2026_POLICIES = [
    Policy(
        name="SG Child Support Package",
        category=FINANCIAL,
        era=NDR2026,
        description=(
            "A new SG Child Support Package replaces the Baby Bonus and Large "
            "Families Schemes, giving every Singapore Citizen child the same "
            "support regardless of birth order: up to S$62,000 per child over "
            "their growing years, comprising a S$10,000 Baby Gift, S$32,000 in "
            "Child Credits paid at S$2,000 a year from age 1 to 16, a S$5,000 "
            "CDA First Step Grant, up to S$5,000 in Government CDA co-matching, "
            "and a S$10,000 top-up to the Post-Secondary Education Account when "
            "the child turns 17. It applies to children born on or after "
            "1 April 2027."
        ),
        expected_pathways=["pbc"],
    ),
    Policy(
        name="Enhanced Childcare Leave",
        category=CAREGIVING,
        era=NDR2026,
        description=(
            "Childcare Leave and Extended Childcare Leave are merged into a "
            "single scheme with more days: parents receive 8, 10, or 12 days of "
            "childcare leave a year depending on whether they have one, two, or "
            "three or more Singapore Citizen children aged 12 and below, up "
            "from 6 days today. The Government fully reimburses employers for "
            "child-related leave up to a cap. The new scheme starts on "
            "1 April 2027."
        ),
        expected_pathways=["pbc", "attitude"],
    ),
    Policy(
        name="Preschool & Infant Care Fee Reduction",
        category=CAREGIVING,
        era=NDR2026,
        description=(
            "Fees at Government-supported preschools will be cut to about "
            "S$150 a month for full-day childcare and S$300 a month for "
            "full-day infant care before means-tested subsidies — less than "
            "half of today's fees. The reductions are phased in from 2028 and "
            "fully in place by 2030, and full subsidies will be extended to "
            "families with Singapore Citizen children regardless of the "
            "applicant's working status."
        ),
        expected_pathways=["pbc"],
    ),
    Policy(
        name="Additional BTO Ballot Chance for Families",
        category=HOUSING,
        era=NDR2026,
        description=(
            "From the February 2027 sales exercise, first-timer families with "
            "or expecting children receive one additional ballot chance for "
            "each Singapore Citizen child aged 18 and below when applying for "
            "a Build-To-Order or Sale of Balance Flats unit, improving their "
            "odds of securing a flat sooner."
        ),
        expected_pathways=["pbc"],
    ),
]

# Which validated-core instruments each NDR 2026 policy supersedes. Analysis
# metadata: lets a report flag that a run mixing both eras double-counts the
# same underlying support. Never shown to agents.
SUPERSEDED_BY = {
    "Baby Bonus & Child Development Account": "SG Child Support Package",
    "Large Family Scheme": "SG Child Support Package",
    "Preschool & Infant Care Subsidies": "Preschool & Infant Care Fee Reduction",
}


def get_policies(category=None, include_ndr2026=False, era=None):
    """Policy instruments, filtered.

    Returns the eight pre-NDR-2026 instruments in their original order by
    default — the validated core, and the order every existing C2/C3 run was
    scheduled against. Pass `include_ndr2026=True` to append the four NDR 2026
    instruments, or `era=` ('pre_ndr2026' / 'ndr2026') to select one generation
    on its own.

    `category` filters to 'financial', 'caregiving', or 'housing' (the last
    exists only in the NDR 2026 set).
    """
    if era == NDR2026:
        policies = list(NDR2026_POLICIES)
    elif era == PRE_NDR2026:
        policies = list(POLICIES)
    elif era is not None:
        raise ValueError(f"era must be {PRE_NDR2026!r}, {NDR2026!r}, or None")
    else:
        policies = list(POLICIES)
        if include_ndr2026:
            policies += list(NDR2026_POLICIES)
    if category is None:
        return policies
    return [p for p in policies if p.category == category]
