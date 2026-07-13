"""Singapore fertility policy instruments (CLAUDE.md Policy Scenarios).

All policies are grounded in real Singapore instruments. `expected_pathways`
records which TPB constructs each policy is hypothesised to move — used for
mechanism-validity checks, never fed to agents as an instruction.
"""

FINANCIAL = "financial"
CAREGIVING = "caregiving"


class Policy:
    """One real Singapore fertility-support instrument.

    `expected_pathways` is the researcher's hypothesis about which TPB
    construct(s) this policy should move (a subset of attitude/norm/pbc). It is
    metadata for analysis only — it is never shown to the agent or the LLM.
    """

    def __init__(self, name, category, description, expected_pathways):
        self.name = name
        self.category = category
        self.description = description
        self.expected_pathways = expected_pathways  # subset of {"attitude", "norm", "pbc"}

    def to_dict(self):
        return {
            "name": self.name,
            "category": self.category,
            "description": self.description,
            "expected_pathways": self.expected_pathways,
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


def get_policies(category=None):
    """All policies, or only one category ('financial' / 'caregiving')."""
    if category is None:
        return list(POLICIES)
    return [p for p in POLICIES if p.category == category]
