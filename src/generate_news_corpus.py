"""One-off LLM generation of the factual policy news corpus for C2/C3.

VacSim (arXiv 2503.09639) pre-generates its news corpus so agents never read
the same text twice; variety, not framing, prevents belief ratcheting. This
script adapts that to the policy channel: for each of the eight registered
policies in sandbox.policy it generates ~6 distinct neutral articles
(announcement / explainer / family_impact / roundup) that build_news_schedule
then serves one per week — announcement first, ongoing-coverage types on
repeat cycles. Design rationale + validation plan:
outputs/analysis/news_dissemination_design.md.

Facts are pinned by per-policy fact blocks distilled from the official Made
For Families M&P booklet (Apr 2025),
https://www.madeforfamilies.gov.sg/docs/default-source/default-document-library/m-p-booklet-(apr-2025).pdf
(Child LifeSG Credits: Budget 2025 announcement). Prompts forbid invented
statistics/quotes and contain no TPB vocabulary (theory-blind inputs).

Usage (from src/, against the local Qwen vLLM):
    python generate_news_corpus.py --output ../outputs/news/news_corpus_qwen.json
"""

import argparse
import datetime
import json
import os

from sandbox.policy import get_policies
from utils.generate_utils import LLMClient
from utils.logging_utils import get_logger, setup_logger

logger = get_logger("news_corpus")

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUTPUT = os.path.join(HERE, "..", "outputs", "news", "news_corpus.json")

TEMPERATURE = 0.9  # diversity between variants; facts are pinned by the fact block
MAX_TRY = 6
WORD_MIN, WORD_MAX = 80, 230  # lint bounds (target 120-200 in the prompt)

# One article of each type per policy, announcements exactly once (the
# schedule serves the announcement on a policy's first week and the
# ongoing-coverage types on repeat cycles).
TYPE_PLAN = ["announcement", "explainer", "explainer",
             "family_impact", "family_impact", "roundup"]

TYPE_INSTRUCTIONS = {
    "announcement": (
        "Report the scheme as a Government announcement/enhancement. Cover its "
        "key features, amounts, and start date."
    ),
    "explainer": (
        "Write a practical explainer for parents: who is eligible, the exact "
        "amounts or entitlements, and how the scheme works. Do not use "
        "announcement framing - the scheme is already in effect."
    ),
    "family_impact": (
        "Illustrate what the scheme means for a typical Singaporean family, "
        "using a clearly illustrative example built only from the facts "
        "provided (e.g. 'a couple having their second child would receive "
        "...'). Do not present the example as a real named family, and do not "
        "invent surnames or quotes. Arithmetic rules: pick the amounts that "
        "match the example's birth order / income tier exactly as stated in "
        "the facts; if you state a total, it must be the exact sum of the "
        "individual amounts you mention in the article; NEVER calculate "
        "out-of-pocket costs, fees after subsidy, savings versus market "
        "rates, or any figure not directly given in the facts."
    ),
    "roundup": (
        "Write a recap piece reminding readers that this existing scheme is "
        "available, summarising its key benefits. Frame it explicitly as "
        "support that is already in place, not as news of something new."
    ),
}

# Facts per policy, distilled from the M&P booklet (Apr 2025) unless noted.
# Keys must match Policy.name in sandbox/policy.py exactly.
FACT_BLOCKS = {
    "Baby Bonus & Child Development Account": (
        "Baby Bonus Cash Gift: S$11,000 for the first and second child, "
        "S$13,000 for the third and subsequent child, paid into the Child "
        "Savings Account in instalments every six months until the child turns "
        "six-and-a-half. Child Development Account (CDA): First Step Grant of "
        "S$5,000 for the first and second child (S$10,000 for third and "
        "subsequent children born on or after 18 Feb 2025) deposited "
        "automatically, plus dollar-for-dollar Government co-matching of "
        "parents' savings up to S$4,000 (first child), S$7,000 (second), "
        "S$9,000 (third and fourth), S$15,000 (fifth and beyond). CDA funds "
        "are usable at Baby Bonus Approved Institutions for childcare, "
        "preschool and healthcare expenses of the child or their siblings."
    ),
    "Large Family Scheme": (
        "The Large Families Scheme supports couples with, or aspiring to, "
        "three or more children: up to S$16,000 of additional support for each "
        "third or subsequent Singapore Citizen child born on or after "
        "18 Feb 2025. Components: an increased CDA First Step Grant of "
        "S$10,000 (up from S$5,000); a S$5,000 Large Family MediSave Grant "
        "deposited into the mother's MediSave, usable for pregnancy and "
        "delivery expenses and approved dependants' medical bills; and S$1,000 "
        "per year in Large Family LifeSG Credits in the years the child turns "
        "one to six (S$6,000 in total), disbursed via the LifeSG app for "
        "groceries, utilities, pharmacy items and transport. The LifeSG "
        "credits also apply to existing large families with at least one "
        "child aged six or below in 2025. Large families additionally enjoy "
        "merchant discounts from corporate partners."
    ),
    "Child LifeSG Credits": (
        # Source: Budget 2025 (SG60 package) - not detailed in the M&P booklet.
        "Announced at Budget 2025 as part of the SG60 package: a one-off "
        "S$500 in Child LifeSG Credits for every Singapore Citizen child "
        "aged 12 and below in 2025 (not an annual payment), disbursed via "
        "the LifeSG app, usable for daily household and child-raising "
        "expenses such as groceries, pharmacy items and transport."
    ),
    "Enhanced Paternity Leave": (
        "Government-Paid Paternity Leave (GPPL) is doubled from 2 to 4 weeks. "
        "The additional two weeks, provided on a voluntary basis since "
        "1 Jan 2024, are mandatory for employers from 1 Apr 2025: eligible "
        "working fathers of Singapore Citizen children born on or after "
        "1 Apr 2025 receive 4 weeks of GPPL. Fathers not eligible because of "
        "their work arrangements (e.g. irregular employment) can apply for "
        "the Government-Paid Paternity Benefit instead."
    ),
    "Shared Parental Leave": (
        "Since 1 Apr 2025 a new Shared Parental Leave (SPL) scheme replaces "
        "the previous one: 10 weeks of paid parental leave shared between "
        "both parents, on top of the existing 16 weeks of maternity leave and "
        "4 weeks of paternity leave, with flexibility to reallocate each "
        "parent's share to the other. Phased in: 6 weeks of SPL for children "
        "born on or after 1 Apr 2025, rising to 10 weeks for children born on "
        "or after 1 Apr 2026. Parents not eligible because of their work "
        "arrangements may apply for the Shared Parental Leave Benefit, a "
        "cash-equivalent scheme."
    ),
    "Flexible Work Arrangement Request Guidelines": (
        "The Tripartite Guidelines on Flexible Work Arrangement Requests, in "
        "effect since 1 Dec 2024, require all employers to fairly consider "
        "formal requests for flexible work arrangements - flexi-place (e.g. "
        "working from home), flexi-time (e.g. staggered start and end times) "
        "and flexi-load (e.g. reduced hours with commensurate pay) - based on "
        "business needs, and to properly communicate the outcomes. The "
        "Tripartite Alliance for Fair and Progressive Employment Practices "
        "(TAFEP) provides resources, templates and workshops to help "
        "companies implement flexible work."
    ),
    "Preschool & Infant Care Subsidies": (
        "Working mothers receive a Basic Subsidy of S$600 per month for "
        "full-day infant care and S$300 per month for full-day childcare, "
        "with Additional Subsidies of up to S$710 (infant care) and S$467 "
        "(childcare) per month for households earning up to S$12,000. Since "
        "9 Dec 2024, families with gross monthly household income up to "
        "S$6,000 qualify for full subsidies for their income tier regardless "
        "of employment status. Full-day childcare fee caps at "
        "Government-supported centres were lowered by S$40 from 1 Jan 2025 to "
        "S$640 (Anchor Operator) and S$680 (Partner Operator) per month "
        "excluding GST, and will be lowered by a further S$30 from 1 Jan 2026 "
        "to S$610 and S$650. By end 2025, 80% of preschoolers can have a "
        "place in a Government-supported preschool; close to 40,000 new "
        "infant and childcare places are being developed from 2025 to 2029."
    ),
    "Infant Childminding Pilot": (
        "A three-year childminding pilot launched on 1 Dec 2024 for infants "
        "aged 2 to 18 months. Operators appointed by the Early Childhood "
        "Development Agency (ECDA) engage childminders to provide home-based "
        "childminding as an additional infant care option. Parents' "
        "out-of-pocket expense for full-time childminding under the pilot is "
        "around S$700 per month, lower than the cost of most childminding "
        "services today."
    ),
}

# Lint: a generated article must mention its policy (any one of these
# lowercase key terms) or it is rejected and regenerated.
KEY_TERMS = {
    "Baby Bonus & Child Development Account": ["baby bonus", "child development account"],
    "Large Family Scheme": ["large famil"],
    "Child LifeSG Credits": ["lifesg"],
    "Enhanced Paternity Leave": ["paternity"],
    "Shared Parental Leave": ["shared parental"],
    "Flexible Work Arrangement Request Guidelines": ["flexible work", "flexi"],
    "Preschool & Infant Care Subsidies": ["infant care", "childcare", "preschool"],
    "Infant Childminding Pilot": ["childmind"],
}

# Theory-blind check: inputs must never carry TPB vocabulary.
TPB_TERMS = ["subjective norm", "behavioural control", "behavioral control",
             "theory of planned behaviour", "theory of planned behavior"]

# Style-only few-shot exemplars about topics OUTSIDE the corpus (housing,
# transport), so style cannot leak policy content between articles.
STYLE_EXAMPLES = """Example article 1:
More new flats, shorter waits as HDB ramps up supply

The Housing and Development Board will launch more than 50,000 new flats \
between 2025 and 2027, exceeding its earlier commitment of 100,000 flats from \
2021 to 2025. Median waiting times have come down to pre-pandemic levels of \
three to four years, and 2,000 to 3,000 Shorter Waiting Time flats will be \
launched every year. Eligible first-timer families can receive up to S$120,000 \
in grants on top of market discounts when buying a new flat. Couples awaiting \
the completion of their new flats, with household incomes of S$7,000 or below, \
can rent an interim flat from HDB at subsidised rates under the Parenthood \
Provisional Housing Scheme, whose supply has more than doubled since 2021.

Example article 2:
Getting around with young children made easier on public transport

All public buses have been fitted with stroller restraint devices, and baby \
care rooms are provided at all new bus interchanges and integrated transport \
hubs, as well as at MRT interchange stations on the Thomson-East Coast Line. \
Ride-hail operators must let commuters travelling with children below 1.35m \
indicate the need for a child seat at the point of booking. Children below \
seven years of age travel free on public transport with a child concession \
card, which parents can apply for at any SimplyGo office."""

SYSTEM_PROMPT = (
    "You are a journalist at a mainstream Singapore news outlet writing "
    "short, factual, neutral news articles about government family support "
    "schemes. Write in British/Singapore English. Use ONLY the facts "
    "provided to you. Do not invent statistics, survey results, take-up "
    "figures, named people, officials, or quotes. Do not editorialise or "
    "give opinions. Amounts and dates must be copied exactly from the facts "
    "provided; any total you state must equal the sum of the individual "
    "amounts you mention. Output the article only: a headline on the first "
    "line, then the body."
)


def _is_short_fact_block(policy_name):
    """Policies with thin fact blocks (e.g. Child LifeSG Credits) can't honestly
    fill 120+ words without padding beyond the facts - ask for less."""
    return len(FACT_BLOCKS[policy_name].split()) < 60


def build_user_prompt(policy, article_type):
    length = "80-150" if _is_short_fact_block(policy.name) else "120-200"
    return (
        f"Scheme: {policy.name}\n"
        f"Summary: {policy.description}\n"
        f"Facts you may use (use only these):\n{FACT_BLOCKS[policy.name]}\n\n"
        f"Task: {TYPE_INSTRUCTIONS[article_type]}\n"
        f"Length: {length} words. Headline on the first line, then the body. "
        f"No other text.\n\n"
        f"Here are two examples of the expected style (different schemes - "
        f"do not reuse their content):\n{STYLE_EXAMPLES}"
    )


def lint_article(text, policy_name):
    """Return None if the article passes, else the reason it fails."""
    words = len(text.split())
    word_min = 55 if _is_short_fact_block(policy_name) else WORD_MIN
    if not (word_min <= words <= WORD_MAX):
        return f"length {words} words outside [{word_min}, {WORD_MAX}]"
    lower = text.lower()
    if not any(term in lower for term in KEY_TERMS[policy_name]):
        return f"missing key term for {policy_name!r}"
    for term in TPB_TERMS:
        if term in lower:
            return f"contains TPB vocabulary {term!r}"
    return None


def generate_corpus(llm, policies, max_try=MAX_TRY):
    """Return the article list; slots that fail max_try times are skipped
    (build_news_schedule falls back to reminder framing for missing variants)."""
    articles = []
    for policy in policies:
        counters = {}
        for article_type in TYPE_PLAN:
            counters[article_type] = counters.get(article_type, 0) + 1
            news_id = (f"{policy.name.lower().replace(' ', '_').replace('&', 'and')}"
                       f"__{article_type}_{counters[article_type]}")
            user_prompt = build_user_prompt(policy, article_type)
            text = None
            for attempt in range(1, max_try + 1):
                try:
                    candidate = llm.chat(SYSTEM_PROMPT, user_prompt,
                                         temperature=TEMPERATURE).strip()
                except RuntimeError as e:
                    logger.warning(f"{news_id}: LLM error on attempt {attempt}: {e}")
                    continue
                reason = lint_article(candidate, policy.name)
                if reason is None:
                    text = candidate
                    break
                logger.warning(f"{news_id}: rejected attempt {attempt} ({reason})")
            if text is None:
                logger.error(f"{news_id}: failed {max_try} attempts, slot skipped")
                continue
            articles.append({
                "news_id": news_id,
                "policy_name": policy.name,
                "policy_category": policy.category,
                "article_type": article_type,
                "text": text,
                "model": llm.model,
                "temperature": TEMPERATURE,
            })
            logger.info(f"{news_id}: ok ({len(text.split())} words)")
    return articles


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--category", choices=["financial", "caregiving"], default=None,
                        help="limit to one policy category (default: all 8 policies)")
    args = parser.parse_args()

    setup_logger(log_path=os.path.splitext(args.output)[0] + ".log")

    policies = get_policies(args.category)
    missing = [p.name for p in policies if p.name not in FACT_BLOCKS]
    if missing:
        raise ValueError(f"No fact block for: {missing}")

    llm = LLMClient()
    logger.info(f"LLM: {llm.provider} / {llm.model}")
    logger.info(f"Generating {len(TYPE_PLAN)} articles x {len(policies)} policies")

    articles = generate_corpus(llm, policies)

    corpus = {
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "model": llm.model,
        "temperature": TEMPERATURE,
        "source": ("Made For Families M&P booklet (Apr 2025); "
                   "Child LifeSG Credits: Budget 2025. See "
                   "outputs/analysis/news_dissemination_design.md"),
        "articles": articles,
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(corpus, f, indent=2, ensure_ascii=False)
    logger.info(f"{len(articles)} articles saved to {args.output}")
