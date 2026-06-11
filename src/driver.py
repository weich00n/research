"""Run the fertility ABM (VacSim: driver.py).

From the src/ directory:

    # one-off: generate the LLM social network
    python generate_social_network.py

    # run a condition
    python driver.py --condition C3 --timesteps 12

Defaults: agents from ../agents_initialised.json, network at
../outputs/social_network.json, results in ../outputs/.
"""

import argparse
import os

from engines.engine import CONDITIONS, Simulation
from LLM_judge import RelevanceScorer
from sandbox.agent import load_agents
from sandbox.news import build_news_schedule
from utils.generate_utils import LLMClient
from utils.network_utils import load_network

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_AGENTS = os.path.join(HERE, "..", "agents_initialised.json")
DEFAULT_OUTPUT_DIR = os.path.join(HERE, "..", "outputs")
DEFAULT_NETWORK = os.path.join(DEFAULT_OUTPUT_DIR, "social_network.json")


def main():
    parser = argparse.ArgumentParser(description="Singapore fertility intention ABM")
    parser.add_argument("--agents", default=DEFAULT_AGENTS)
    parser.add_argument("--network", default=DEFAULT_NETWORK)
    parser.add_argument("--condition", choices=list(CONDITIONS), default="C0")
    parser.add_argument("--timesteps", type=int, default=12)
    parser.add_argument("--policy-category", choices=["financial", "caregiving", "combined"],
                        default="combined",
                        help="policy scenario for the news schedule (C2/C3)")
    parser.add_argument("--relevance-mode", choices=["llm", "cosine"], default="llm")
    parser.add_argument("--num-agents", type=int, default=None,
                        help="limit to first N agents (for cheap test runs)")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()

    agents = load_agents(args.agents)
    if args.num_agents:
        agents = agents[: args.num_agents]
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

    news_schedule = None
    if policy_on:
        category = None if args.policy_category == "combined" else args.policy_category
        news_schedule = build_news_schedule(args.timesteps, category=category)

    scorer = RelevanceScorer(mode=args.relevance_mode, llm=llm)

    sim = Simulation(
        agents=agents,
        network=network,
        condition=args.condition,
        llm=llm,
        scorer=scorer,
        news_schedule=news_schedule,
        output_dir=args.output_dir,
        run_name=args.run_name,
    )
    sim.run(args.timesteps)
    print(f"\nDone. Results: {sim.save()}")


if __name__ == "__main__":
    main()
