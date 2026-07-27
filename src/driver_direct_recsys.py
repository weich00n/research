"""Run the VacSim-recsys variant: engine_direct's no-TPB mechanics + VacSim's
actual embedding-similarity recommenders for news/tweet exposure (~/VacSim:
src/recommenders/). See engines/engine_direct_recsys.py's module docstring
for exactly what changes vs driver_direct.py.

Same seeded agent file, network and news/context corpora as driver_direct.py,
so results are comparable to both the TPB study and the fixed-schedule direct
replication -- only the exposure-selection mechanism differs.

From the src/ directory:

    python driver_direct_recsys.py --condition C3 --news-corpus ../outputs/news/news_corpus_qwen.json

Output goes to run_<condition>_recsys.json / .log by default, so this never
collides with a TPB run or a fixed-schedule direct run of the same condition.
"""

import argparse
import json
import os

from engines.engine import CONDITIONS
from engines.engine_direct_recsys import (
    DEFAULT_FOLLOW_ALPHA,
    DEFAULT_NEWS_TOP_K,
    DEFAULT_TWEET_TOP_K,
    RecsysDirectSimulation,
)
from sandbox.agent_direct import DirectAgent, load_agents
from sandbox.lesson import reseed_id_counter as reseed_lesson_ids
from sandbox.news import build_news_schedule
from sandbox.tweet import reseed_id_counter as reseed_tweet_ids
from utils.generate_utils import LLMClient
from utils.network_utils import load_network

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_AGENTS = os.path.join(HERE, "..", "agents_final_100_seeded.json")
DEFAULT_OUTPUT_DIR = os.path.join(HERE, "..", "outputs", "runs")
DEFAULT_NETWORK = os.path.join(HERE, "..", "outputs", "networks", "social_network.json")


def main():
    parser = argparse.ArgumentParser(
        description="VacSim-recsys fertility ABM replication (no-TPB + real recommenders)")
    parser.add_argument("--agents", default=DEFAULT_AGENTS)
    parser.add_argument("--network", default=DEFAULT_NETWORK)
    parser.add_argument("--condition", choices=list(CONDITIONS), default="C0")
    parser.add_argument("--timesteps", type=int, default=12)
    parser.add_argument("--policy-category", choices=["financial", "caregiving", "combined"],
                        default="combined",
                        help="policy scenario feeding the cumulative news pool (C2/C3)")
    parser.add_argument("--news-corpus", default=None,
                        help="pre-generated article corpus (generate_news_corpus.py output)")
    parser.add_argument("--context-corpus", default=None,
                        help="ambient context corpus (generate_context_corpus.py output)")
    parser.add_argument("--context-mix", choices=["balanced", "negative", "positive", "neutral"],
                        default="balanced")
    parser.add_argument("--news-top-k", type=int, default=DEFAULT_NEWS_TOP_K,
                        help="articles recommended per agent per week, from the "
                             "cumulative pool revealed so far")
    parser.add_argument("--tweet-top-k", type=int, default=DEFAULT_TWEET_TOP_K,
                        help="tweets recommended per agent per week, from the whole "
                             "population's tweet pool so far")
    parser.add_argument("--follow-alpha", type=float, default=DEFAULT_FOLLOW_ALPHA,
                        help="additive rank boost for tweets from a followed agent "
                             "(VacSim TweetRecommender default: 0.3)")
    parser.add_argument("--num-agents", type=int, default=None,
                        help="limit to first N agents (for cheap test runs)")
    parser.add_argument("--no-gating", action="store_true",
                        help="disable update gating (run the intention-update call "
                             "even in weeks with no new inputs)")
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--resume", action="store_true",
                        help="continue outputs/runs/<run_name>.json from its last "
                             "checkpoint instead of starting fresh")
    args = parser.parse_args()

    run_name = args.run_name or f"run_{args.condition}_recsys"
    run_path = os.path.join(args.output_dir, f"{run_name}.json")

    resume_state = None
    if args.resume and os.path.exists(run_path):
        with open(run_path, encoding="utf-8") as f:
            resume_state = json.load(f)
        agents = [DirectAgent(p) for p in resume_state["agents"]]
        print(f"Resuming '{run_name}' from {run_path}: {len(agents)} agents, "
              f"completed through week {resume_state['current_timestep']}")
    else:
        if args.resume:
            print(f"--resume set but no checkpoint at {run_path}; starting fresh")
        agents = load_agents(args.agents)
        if args.num_agents:
            agents = agents[: args.num_agents]
    reseed_lesson_ids([l for a in agents for l in a.lessons])
    reseed_tweet_ids([t for a in agents for t in a.tweets])
    print(f"Loaded {len(agents)} agents")

    llm = LLMClient()
    print(f"LLM: {llm.provider} / {llm.model}")

    policy_on, social_on = CONDITIONS[args.condition]

    network = {}
    if social_on:
        network = load_network(args.network)
        missing = [a.agent_id for a in agents if a.agent_id not in network]
        if missing:
            raise RuntimeError(
                f"{len(missing)} agents missing from network {args.network} "
                f"(run generate_social_network.py first)")
        if args.num_agents:
            loaded_ids = {a.agent_id for a in agents}
            network = {aid: [f for f in follows if f in loaded_ids]
                       for aid, follows in network.items() if aid in loaded_ids}

    news_schedule = None
    if policy_on:
        category = None if args.policy_category == "combined" else args.policy_category
        news_schedule = build_news_schedule(args.timesteps, category=category,
                                            corpus_path=args.news_corpus,
                                            context_corpus_path=args.context_corpus,
                                            context_mix=args.context_mix)

    sim = RecsysDirectSimulation(
        agents=agents,
        network=network,
        condition=args.condition,
        llm=llm,
        news_schedule=news_schedule,
        output_dir=args.output_dir,
        run_name=run_name,
        concurrency=args.concurrency,
        gate_no_input=not args.no_gating,
        news_top_k=args.news_top_k,
        tweet_top_k=args.tweet_top_k,
        follow_alpha=args.follow_alpha,
    )

    done_through = resume_state["current_timestep"] if resume_state else 0
    sim.current_timestep = done_through
    remaining = args.timesteps - done_through
    if remaining <= 0:
        print(f"Run already complete through week {done_through} "
              f"(requested {args.timesteps} timesteps); nothing to do.")
    else:
        sim.run(remaining, start_timestep=done_through + 1)
    print(f"\nDone. Results: {sim.save()}")


if __name__ == "__main__":
    main()
