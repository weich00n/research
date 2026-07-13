"""Run the fertility ABM (VacSim: driver.py).

From the src/ directory:

    # one-off: generate the LLM social network
    python generate_social_network.py

    # one-off: freeze seeds + the t=0 baseline into a shared agent file so every
    # condition starts identical (writes ../agents_final_100_seeded.json)
    python driver.py --init-only

    # run a condition off the shared seeded file
    python driver.py --condition C3 --agents ../agents_final_100_seeded.json --timesteps 12

Defaults: agents from ../agents_final_100.json, network at
../outputs/networks/social_network.json, results in ../outputs/runs/.
"""

import argparse
import json
import os

from engines.engine import CONDITIONS, Simulation
from LLM_judge import RelevanceScorer
from sandbox.agent import Agent, load_agents
from sandbox.lesson import reseed_id_counter as reseed_lesson_ids
from sandbox.news import build_news_schedule
from sandbox.tweet import reseed_id_counter as reseed_tweet_ids
from utils.generate_utils import LLMClient
from utils.network_utils import load_network

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_AGENTS = os.path.join(HERE, "..", "agents_final_100.json")
DEFAULT_OUTPUT_DIR = os.path.join(HERE, "..", "outputs", "runs")
DEFAULT_NETWORK = os.path.join(HERE, "..", "outputs", "networks", "social_network.json")


def main():
    """Parse CLI args, build the agents/LLM/network/scorer, and run the simulation."""
    parser = argparse.ArgumentParser(description="Singapore fertility intention ABM")
    parser.add_argument("--agents", default=DEFAULT_AGENTS)
    parser.add_argument("--network", default=DEFAULT_NETWORK)
    parser.add_argument("--condition", choices=list(CONDITIONS), default="C0")
    parser.add_argument("--timesteps", type=int, default=12)
    parser.add_argument("--policy-category", choices=["financial", "caregiving", "combined"],
                        default="combined",
                        help="policy scenario for the news schedule (C2/C3)")
    parser.add_argument("--news-corpus", default=None,
                        help="pre-generated article corpus for the news schedule "
                             "(generate_news_corpus.py output); each week then serves "
                             "a distinct factual article instead of the verbatim "
                             "announce/remind text. Default: legacy behaviour")
    parser.add_argument("--relevance-mode", choices=["llm", "cosine", "hybrid"], default="llm",
                        help="how memory TPB relevance is scored: 'llm' (judge every "
                             "memory at creation), 'cosine' (embedding similarity), or "
                             "'hybrid' (cosine prefilter top-K per construct -> LLM rerank)")
    parser.add_argument("--rerank-top-k", type=int, default=12,
                        help="hybrid mode: cosine shortlist size per construct that "
                             "gets LLM-reranked (union across constructs)")
    parser.add_argument("--num-agents", type=int, default=None,
                        help="limit to first N agents (for cheap test runs)")
    parser.add_argument("--concurrency", type=int, default=32,
                        help="agents processed in parallel per week (concurrent LLM "
                             "requests); raise to saturate a vLLM server / 4 GPUs")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--resume", action="store_true",
                        help="continue outputs/runs/<run_name>.json from its last checkpoint "
                             "instead of starting fresh (skips already-finished agents)")
    parser.add_argument("--init-only", action="store_true",
                        help="generate seed memories + the t=0 baseline, write the seeded "
                             "agent file (--init-out) and exit. Run ONCE; every condition "
                             "then loads that file via --agents for an identical start.")
    parser.add_argument("--init-out",
                        default=os.path.join(HERE, "..", "agents_final_100_seeded.json"),
                        help="output path for --init-only (seeded + baselined agents)")
    args = parser.parse_args()

    # Resolve the run name / checkpoint path the same way Simulation does, so
    # --resume can find the file a previous run wrote.
    run_name = args.run_name or f"run_{args.condition}"
    run_path = os.path.join(args.output_dir, f"{run_name}.json")

    # Resume rebuilds agents (with their seed memories, beliefs, history, tweets)
    # from the last checkpoint; a fresh run loads pristine profiles. On resume the
    # id counters are advanced past the loaded ids so new memories don't collide.
    resume_state = None
    if args.resume and os.path.exists(run_path):
        with open(run_path, encoding="utf-8") as f:
            resume_state = json.load(f)
        agents = [Agent(p) for p in resume_state["agents"]]
        print(f"Resuming '{run_name}' from {run_path}: {len(agents)} agents, "
              f"completed through week {resume_state['current_timestep']}")
    else:
        if args.resume:
            print(f"--resume set but no checkpoint at {run_path}; starting fresh")
        agents = load_agents(args.agents)
        if args.num_agents:
            agents = agents[: args.num_agents]
    # Advance the memory/tweet id counters past any ids already present in the
    # loaded agents — needed on resume AND when loading a pre-seeded agent file
    # (agents_final_100_seeded.json), so freshly minted ids don't collide with the
    # frozen seed ids. A no-op for a pristine profile file (no memories yet).
    reseed_lesson_ids([l for a in agents for l in a.lessons])
    reseed_tweet_ids([t for a in agents for t in a.tweets])
    print(f"Loaded {len(agents)} agents")

    llm = LLMClient()
    print(f"LLM: {llm.provider} / {llm.model}")
    scorer = RelevanceScorer(mode=args.relevance_mode, llm=llm)

    # --init-only: generate seeds + the t=0 baseline once, write the seeded agent
    # file, and exit. Condition is irrelevant to initialisation (no policy/social),
    # so use C0 with no network/news; a distinct run_name keeps its seed-phase
    # checkpoint from colliding with a real run_C0.json.
    if args.init_only:
        init_sim = Simulation(
            agents=agents, network={}, condition="C0", llm=llm, scorer=scorer,
            output_dir=args.output_dir, run_name="agents_init",
            concurrency=args.concurrency, rerank_top_k=args.rerank_top_k,
        )
        init_sim.initialise_seed_memories()
        init_sim.initialise_baseline()
        with open(args.init_out, "w", encoding="utf-8") as f:
            json.dump([a.to_dict() for a in init_sim.agents], f, indent=2)
        print(f"\nSeeded + baselined {len(init_sim.agents)} agents -> {args.init_out}")
        return

    # The chosen condition unpacks into two switches that control what inputs
    # agents receive: C0=(off,off), C1=(off,on), C2=(on,off), C3=(on,on).
    policy_on, social_on = CONDITIONS[args.condition]

    # The network is only needed when social posts are on. It must cover every
    # agent we're running (it's keyed by agent_id) — otherwise a follower lookup
    # would fail mid-run, so we validate up front and fail fast.
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
        news_schedule = build_news_schedule(args.timesteps, category=category,
                                            corpus_path=args.news_corpus)

    sim = Simulation(
        agents=agents,
        network=network,
        condition=args.condition,
        llm=llm,
        scorer=scorer,
        news_schedule=news_schedule,
        output_dir=args.output_dir,
        run_name=run_name,
        concurrency=args.concurrency,
        rerank_top_k=args.rerank_top_k,
    )

    # Continue from the last completed week (0 for a fresh run). Seed init and
    # finished agents are skipped inside the engine, so only outstanding work runs.
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
