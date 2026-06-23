"""Prompts for initialising two LLM-inferred persona attributes.

v1 = lifted verbatim from `fark_agent llama.ipynb` (cells 7a and 7b) — the
prompts that built the current agent files. Never edit these.

v2 = re-engineered set (2026-06-12): fin gains per-band Singapore anchors,
an explicit multi-cue weighing instruction, and richer input fields
(persona + career goals); rel is deliberately neutral — no base rates or
directional cue lists, only "don't default" and "cite this profile's
details". Caveat: fin v2 changes instructions AND inputs together, so a
v1-vs-v2 pilot shows whether the prompt matters, not which half did it.

Select a set via PROMPT_SETS["v1"|"v2"].
"""

FIN_SYSTEM_PROMPT = """You are a Singapore socio-economic analyst. Given an adult's demographic profile, estimate their financial security on a 1-5 scale where:
1 = Low (financially struggling, minimal savings/assets)
2 = Lower-middle
3 = Middle (stable, modest savings)
4 = Upper-middle
5 = Upper (high income, substantial assets)

Use cues: occupation seniority, industry pay norms in Singapore, education level, Singapore planning area (HDB heartland vs private estate / mature vs non-mature), and age (career stage). Be calibrated — most working adults are 2-4, not 5.

Respond ONLY with valid JSON: {"score": <int 1-5>, "reasoning": "<one or two sentences>"}"""

REL_SYSTEM_PROMPT = """You are a behavioural researcher reading an adult's profile to judge whether they are most plausibly:
- "Single": not in any romantic relationship right now
- "Dating": in a non-marital romantic relationship (boyfriend/girlfriend, partner, courtship)

Weigh cues such as: career intensity, lifestyle and hobbies, age, persona narrative (any mention of partners, dating, social life, family orientation), cultural and educational background. Do not default to one label; reason from the specific profile.

Respond ONLY with valid JSON: {"status": "Single" | "Dating", "reasoning": "<one or two sentences>"}"""


FIN_SYSTEM_PROMPT_V2 = """You are a Singapore socio-economic analyst. Given an adult's profile, estimate their financial security on a 1-5 scale:
1 = Low — financially struggling, little to no savings (e.g., low-wage service/manual work, gig work with no buffer)
2 = Lower-middle — gets by month to month, thin savings (e.g., junior non-degree roles, entry clerical/retail)
3 = Middle — stable income, modest savings and CPF, can absorb small shocks (e.g., mid-level executives, technicians)
4 = Upper-middle — comfortable, healthy savings/investments (e.g., established professionals, managers, specialists)
5 = Upper — high income, substantial assets (e.g., senior management, top professionals, business owners)

Weigh ALL of these cues together, not just one:
- Occupation seniority and typical pay for that role in Singapore
- Industry pay norms
- Education level
- Planning area (HDB heartland vs private-estate areas; mature vs non-mature)
- Age / career stage (the same job at 25 vs 42 implies different accumulation)
- Persona narrative and career goals, if they signal financial circumstances

Be calibrated: across many Singapore working adults aged 21-45, scores should spread across 2-4, with 1 and 5 reserved for clear cases. Your reasoning must cite at least two different cues and note any conflict between them.

Respond ONLY with valid JSON: {"score": <int 1-5>, "reasoning": "<two or three sentences citing the cues used>"}"""

REL_SYSTEM_PROMPT_V2 = """You are a behavioural researcher judging whether an unmarried adult in Singapore is most plausibly:
- "Single": not in any romantic relationship right now
- "Dating": in a non-marital romantic relationship (boyfriend/girlfriend, partner, courtship)

Do not default to either label. Most profiles mention neither a partner nor being single, so the absence of an explicit mention is weak evidence in either direction. Weigh the overall picture of this specific person — life stage, lifestyle, values, social orientation, and stated priorities — and make the call that best fits the whole profile.

Your reasoning must name the specific details from THIS profile that drove your call — do not give generic reasoning.

Respond ONLY with valid JSON: {"status": "Single" | "Dating", "reasoning": "<two or three sentences citing the profile details used>"}"""


def build_fin_prompt(row) -> str:
    """v1 financial user message: demographics only (age/gender/edu/occ/industry/area)."""
    return (
        f"Age: {row['age']}\n"
        f"Gender: {row['gender']}\n"
        f"Education level: {row['education']}\n"
        f"Occupation: {row['occupation']}\n"
        f"Industry: {row.get('industry') or 'Not specified'}\n"
        f"Singapore planning area: {row['planning_area']}"
    )


def build_fin_prompt_v2(row) -> str:
    """v2 financial user message: the v1 fields PLUS persona + career goals."""
    return (
        build_fin_prompt(row) + "\n"
        f"Persona: {row['persona']}\n"
        f"Career goals: {row['career_goals']}"
    )


def build_rel_prompt(row) -> str:
    """Relationship user message: demographics + persona/cultural/hobbies/career (shared by v1 & v2)."""
    return (
        f"Age: {row['age']}\n"
        f"Gender: {row['gender']}\n"
        f"Education level: {row['education']}\n"
        f"Occupation: {row['occupation']}\n"
        f"Persona: {row['persona']}\n"
        f"Cultural background: {row['cultural_bg']}\n"
        f"Hobbies and interests: {row['hobbies']}\n"
        f"Career goals: {row['career_goals']}"
    )


PROMPT_SETS = {
    "v1": {"fin_system": FIN_SYSTEM_PROMPT,    "rel_system": REL_SYSTEM_PROMPT,
           "build_fin": build_fin_prompt,      "build_rel": build_rel_prompt},
    "v2": {"fin_system": FIN_SYSTEM_PROMPT_V2, "rel_system": REL_SYSTEM_PROMPT_V2,
           "build_fin": build_fin_prompt_v2,   "build_rel": build_rel_prompt},
}
