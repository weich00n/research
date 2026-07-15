"""One-off LLM generation of the mixed-valence ambient context corpus.

The policy corpus (generate_news_corpus.py) is uniformly positive, so C2
constructs only ever drift up. This corpus supplies the everyday Singapore
media environment around it — real pressures (cost of living, housing, job
market, cost of raising a child), real relief (moderating inflation, wage
growth, household support, housing supply), and neutral placebo items — each
article tagged with a `valence` so the schedule's draw composition becomes
the experimental dial ({balanced, negative, positive, neutral} context
modes; VacSim §4.2 pool-ratio design). Design + framing rationale:
outputs/analysis/news_dissemination_design.md.

Hard rule: context articles must NEVER mention the fertility policy
instruments (Baby Bonus, parental leave, childcare subsidies, ...) — those
are the treatment channel; context must not be policy stance in either
direction. Enforced by prompt + lint.

Usage (from src/, against the local Qwen vLLM):
    python generate_context_corpus.py --output ../outputs/news/context_corpus_qwen.json
"""

import argparse
import datetime
import json
import os

from generate_news_corpus import TPB_TERMS, WORD_MAX, WORD_MIN
from utils.generate_utils import LLMClient
from utils.logging_utils import get_logger, setup_logger

logger = get_logger("context_corpus")

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUTPUT = os.path.join(HERE, "..", "outputs", "news", "context_corpus.json")

TEMPERATURE = 0.9
MAX_TRY = 6

# The fertility policy instruments (the treatment channel) must never appear
# in context articles, in any stance. Lowercase substrings, lint-enforced.
POLICY_TERMS = [
    "baby bonus", "cash gift", "child development account", "cda",
    "lifesg", "paternity", "maternity leave", "shared parental",
    "parental leave", "childmind", "subsid", "tripartite",
    "flexible work arrangement", "large famil", "medisave grant",
]

# (topic, valence) -> fact block. Only well-established figures; where a
# precise number is uncertain the block stays qualitative rather than risking
# invention (the audit number-diffs every article against its block).
TOPICS = {
    # ── negative: pressures ─────────────────────────────────────────────
    ("Cost of Living", "negative"): (
        "GST rose to 9% on 1 January 2024 (from 8% in 2023, 7% before that). "
        "Singapore has repeatedly been ranked among the world's most "
        "expensive cities in the Economist Intelligence Unit's Worldwide "
        "Cost of Living survey. Households report higher spending on food, "
        "hawker meals, utilities and transport than in previous years; "
        "everyday prices have risen noticeably since 2022."
    ),
    ("Housing Pressure", "negative"): (
        "The typical wait for a new Build-To-Order (BTO) flat is 3 to 4 "
        "years from application to key collection. HDB resale prices rose "
        "for many consecutive quarters through 2024, and rents climbed "
        "sharply in 2022-2023 before stabilising at high levels. Some young "
        "couples delay moving in together, or live with parents, while "
        "waiting for their flat to be completed."
    ),
    ("Job Market Pressure", "negative"): (
        "Global technology and finance firms carried out repeated rounds of "
        "layoffs in 2023-2024, affecting Singapore-based staff. Ministry of "
        "Manpower data showed retrenchments rising in 2023 compared with "
        "2022. Employees in Singapore work some of the longest hours among "
        "developed economies, and many professionals describe a highly "
        "competitive environment for promotions and job security."
    ),
    ("Cost of Raising a Child", "negative"): (
        "Full-day childcare at government-supported preschool centres costs "
        "around S$640-S$680 per month before any support, and private or "
        "premium options cost substantially more. Commonly cited estimates "
        "put the cumulative cost of raising a child in Singapore to "
        "adulthood in the hundreds of thousands of dollars, spanning "
        "childcare, education, enrichment classes, healthcare and housing "
        "space. Surveys of young Singaporeans regularly list cost as a top "
        "consideration in family planning."
    ),
    # ── positive: relief / favourable conditions ────────────────────────
    ("Inflation & Wages", "positive"): (
        "Core inflation in Singapore has eased substantially from its "
        "2022-2023 peak. Median household incomes have grown in real terms "
        "over the past decade, and wage growth outpaced inflation again as "
        "price pressures moderated."
    ),
    ("Household Support", "positive"): (
        "The Government has disbursed CDC vouchers to every Singaporean "
        "household in recent years (S$300 per household in 2024) to offset "
        "daily expenses at hawker stalls, heartland merchants and "
        "supermarkets, alongside further Budget support such as utility "
        "rebates for households."
    ),
    ("Labour Market Strength", "positive"): (
        "Singapore's unemployment rate remains low by international "
        "standards, at around 2%. The labour market stayed tight through "
        "2024-2025, with employers competing for workers in many sectors "
        "and resident employment above pre-pandemic levels."
    ),
    ("Housing Supply Relief", "positive"): (
        "HDB will launch more than 50,000 new flats between 2025 and 2027, "
        "exceeding its earlier commitment of 100,000 flats from 2021-2025. "
        "Median BTO waiting times have come down to pre-pandemic levels of "
        "3-4 years, 2,000 to 3,000 Shorter Waiting Time flats are being "
        "launched every year, and interim rental flat supply has been "
        "expanded to 4,000 units by the second half of 2025."
    ),
    # ── neutral: placebo (no plausible fertility valence) ───────────────
    ("Hawker Culture", "neutral"): (
        "Singapore's hawker culture was inscribed on the UNESCO "
        "Representative List of the Intangible Cultural Heritage of "
        "Humanity in December 2020. Hawker centres remain daily gathering "
        "places across the island, and community events regularly "
        "celebrate heritage stalls and new-generation hawkers."
    ),
    ("Public Transport", "neutral"): (
        "The Thomson-East Coast MRT line opened in stages through 2024, "
        "and construction continues on the upcoming Jurong Region Line and "
        "Cross Island Line. Singapore's rail network keeps expanding, with "
        "new stations shortening journeys across the island."
    ),
    ("Parks & Leisure", "neutral"): (
        "Singapore's park connector network spans hundreds of kilometres, "
        "linking parks, waterways and nature reserves for cyclists and "
        "runners. Attractions such as Gardens by the Bay and the Rail "
        "Corridor draw steady weekend crowds."
    ),
    ("Sports & Community", "neutral"): (
        "Community sports facilities under ActiveSG see high weekend "
        "usage, and mass participation events such as the Standard "
        "Chartered Singapore Marathon return annually with tens of "
        "thousands of runners."
    ),
}

# variants per (topic, valence): negative/positive 3 each, neutral 2 each
# -> 4*3 + 4*3 + 4*2 = 32 articles
VARIANTS = {"negative": ["report", "report", "feature"],
            "positive": ["report", "report", "feature"],
            "neutral": ["report", "feature"]}

TYPE_INSTRUCTIONS = {
    "report": (
        "Write a straight news report on this topic. Factual reporting "
        "register; describe the situation and, where the facts support it, "
        "how households or workers experience it."
    ),
    "feature": (
        "Write a short human-angle feature on this topic, using a clearly "
        "illustrative unnamed example (e.g. 'a young couple renting while "
        "waiting for their flat...'). Do not invent named people, quotes, "
        "or statistics beyond the facts provided."
    ),
}

# lint key terms per topic (any one must appear, lowercase)
KEY_TERMS = {
    "Cost of Living": ["cost of living", "gst", "price", "expensive"],
    "Housing Pressure": ["bto", "flat", "resale", "rent", "housing"],
    "Job Market Pressure": ["layoff", "retrench", "job", "working hours", "employ"],
    "Cost of Raising a Child": ["childcare", "raising a child", "cost"],
    "Inflation & Wages": ["inflation", "wage", "income"],
    "Household Support": ["cdc voucher", "voucher", "rebate"],
    "Labour Market Strength": ["unemployment", "labour market", "employ"],
    "Housing Supply Relief": ["flat", "hdb", "waiting time", "housing"],
    "Hawker Culture": ["hawker"],
    "Public Transport": ["mrt", "rail", "transport", "line"],
    "Parks & Leisure": ["park", "garden", "trail", "corridor"],
    "Sports & Community": ["sport", "marathon", "activesg", "run"],
}

SYSTEM_PROMPT = (
    "You are a journalist at a mainstream Singapore news outlet writing "
    "short, factual articles about the economy and everyday life in "
    "Singapore. Write in British/Singapore English. Use ONLY the facts "
    "provided to you; amounts and dates must be copied exactly. Do not "
    "invent statistics, survey results, named people, officials, or quotes. "
    "Do not editorialise. Do not mention any government family or "
    "parenthood scheme (parental leave, baby bonuses, childcare support "
    "schemes, and similar) — this article is not about family policy. "
    "Output the article only: a headline on the first line, then the body."
)


def build_user_prompt(topic, valence, article_type):
    return (
        f"Topic: {topic} ({'pressure on' if valence == 'negative' else 'positive development for' if valence == 'positive' else 'everyday life in'} Singapore households)\n"
        f"Facts you may use (use only these):\n{TOPICS[(topic, valence)]}\n\n"
        f"Task: {TYPE_INSTRUCTIONS[article_type]}\n"
        f"Length: 100-180 words. Headline on the first line, then the body. "
        f"No other text."
    )


def lint_article(text, topic):
    words = len(text.split())
    if not (60 <= words <= WORD_MAX):
        return f"length {words} words outside [60, {WORD_MAX}]"
    lower = text.lower()
    if not any(term in lower for term in KEY_TERMS[topic]):
        return f"missing key term for {topic!r}"
    for term in TPB_TERMS:
        if term in lower:
            return f"contains TPB vocabulary {term!r}"
    for term in POLICY_TERMS:
        if term in lower:
            return f"mentions fertility-policy term {term!r}"
    return None


def generate_corpus(llm, max_try=MAX_TRY):
    articles = []
    for (topic, valence), _facts in TOPICS.items():
        counters = {}
        for article_type in VARIANTS[valence]:
            counters[article_type] = counters.get(article_type, 0) + 1
            key = topic.lower().replace(" ", "_").replace("&", "and")
            news_id = f"ctx__{key}__{valence}_{article_type}_{counters[article_type]}"
            user_prompt = build_user_prompt(topic, valence, article_type)
            text = None
            for attempt in range(1, max_try + 1):
                try:
                    candidate = llm.chat(SYSTEM_PROMPT, user_prompt,
                                         temperature=TEMPERATURE).strip()
                except RuntimeError as e:
                    logger.warning(f"{news_id}: LLM error on attempt {attempt}: {e}")
                    continue
                reason = lint_article(candidate, topic)
                if reason is None:
                    text = candidate
                    break
                logger.warning(f"{news_id}: rejected attempt {attempt} ({reason})")
            if text is None:
                logger.error(f"{news_id}: failed {max_try} attempts, slot skipped")
                continue
            articles.append({
                "news_id": news_id,
                "policy_name": topic,          # topic label; harness makes a context stub
                "policy_category": "context",
                "article_type": f"{valence}_{article_type}",  # valence visible in run JSONs
                "valence": valence,
                "text": text,
                "model": llm.model,
                "temperature": TEMPERATURE,
            })
            logger.info(f"{news_id}: ok ({len(text.split())} words)")
    return articles


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    setup_logger(log_path=os.path.splitext(args.output)[0] + ".log")

    llm = LLMClient()
    logger.info(f"LLM: {llm.provider} / {llm.model}")
    n = sum(len(VARIANTS[v]) for (_, v) in TOPICS)
    logger.info(f"Generating {n} context articles over {len(TOPICS)} topic/valence blocks")

    articles = generate_corpus(llm)

    corpus = {
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "model": llm.model,
        "temperature": TEMPERATURE,
        "source": ("Ambient Singapore context, mixed valence; facts pinned in "
                   "generate_context_corpus.py (GST/EIU/MOM/HDB/booklet). See "
                   "outputs/analysis/news_dissemination_design.md"),
        "articles": articles,
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(corpus, f, indent=2, ensure_ascii=False)
    logger.info(f"{len(articles)} articles saved to {args.output}")
