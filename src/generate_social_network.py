"""LLM-generated fixed directed exposure graph (VacSim: generate_social_network.py).

Mirrors VacSim: each agent is prompted in first person ("Pretend you are ...
Which of these people will you become friends with?") with a compact one-line
profile of every other agent, and returns a comma-separated list of agent IDs.
Each chosen ID becomes a directed edge agent -> friend, meaning the agent
follows that friend and reads their posts (posted at t-1) at timestep t.

The network is generated once before the simulation and stays fixed.

Usage (from src/):
    python generate_social_network.py
    python generate_social_network.py --num-agents 20   # cheap test run
"""

import argparse
import os

import numpy as np

from sandbox.agent import load_agents
from utils.generate_utils import LLMClient
from utils.logging_utils import get_logger, setup_logger
from utils.network_utils import load_network, save_network

RANDOM_STATE = 42

logger = get_logger("network")

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_AGENTS = os.path.join(HERE, "..", "agents_final_100.json")
DEFAULT_OUTPUT = os.path.join(HERE, "..", "outputs", "networks", "social_network.json")


def _parse_friend_indices(response, self_idx, num_agents):
    """Pull friend indices out of the LLM's free-text reply.

    Requires the reply to actually BE the instructed "ID, ID, ID" format:
    split on commas and require every non-empty token to be a bare integer.
    This is deliberately strict, not lenient -- a reply like "I'll pick 3
    people: 5, 12, 44" fails to parse (raising, so the caller retries)
    instead of a looser digit-scan silently absorbing the stray "3" as a
    real friend edge. This network is generated once and frozen for the
    whole simulation, so a wrong edge here would be undetectable downstream.
    """
    tokens = [t.strip() for t in response.strip().split(",")]
    if not all(t.isdigit() for t in tokens if t):
        raise ValueError(f"Response is not a clean comma-separated ID list: "
                          f"{response[:100]!r}")
    indices = []
    for t in tokens:
        if not t:
            continue
        i = int(t)
        if i != self_idx and 0 <= i < num_agents and i not in indices:
            indices.append(i)
    if not indices:
        raise ValueError(f"No valid friend indices in response: {response[:100]!r}")
    return indices


def generate_llm_network(agents, llm, max_try=10, fallback_k=5, seed=RANDOM_STATE,
                         verbose=True, include_area=True, include_persona=False,
                         checkpoint_path=None):
    """Return {agent_id: [followed agent_ids]} chosen by the LLM per agent.

    If the LLM fails max_try times for an agent (VacSim leaves them edgeless),
    we instead fall back to `fallback_k` seeded-random friends so no agent is
    isolated; set fallback_k=0 for strict VacSim behaviour.

    `include_area=False` drops the residence field from every profile (and from
    the prompt's field list) for the with/without-area homophily comparison.
    `include_persona=True` appends each agent's narrative persona (experimental;
    ~9.9k-token prompt — needs the vLLM server at MAX_MODEL_LEN >= 16384).

    `checkpoint_path`, if given, makes this resumable: any agent already
    present in the JSON at that path is skipped (its edges kept as-is), and
    the network is re-saved after every agent so a crash partway through a
    ~100-agent / up-to-1000-call LLM job doesn't lose all prior progress.
    """
    # One independent RNG substream per agent (not one shared, sequentially-
    # advancing rng) -- otherwise which agents land on the random-fallback
    # path, and what they draw, depends on the live pattern of LLM failures,
    # which differs run to run. That makes seed=42 reproducibility illusory
    # whenever the failure set differs. Spawning per-agent substreams means
    # agent i's fallback draw (if it ever needs one) is always the same,
    # regardless of what happened to any other agent.
    agent_rngs = [np.random.default_rng(s)
                 for s in np.random.SeedSequence(seed).spawn(len(agents))]
    profile_kwargs = {"include_area": include_area, "include_persona": include_persona}
    profile_lines = [f"{i}. {a.get_profile_str(**profile_kwargs)}"
                     for i, a in enumerate(agents)]
    schema = "ID. Gender\tAge\tRelationship\tEducation\tOccupation\tIndustry"
    if include_area:
        schema += "\tArea"
    if include_persona:
        schema += "\tAbout"

    network = {}
    if checkpoint_path and os.path.exists(checkpoint_path):
        network = load_network(checkpoint_path)
        logger.info(f"Resuming from {checkpoint_path}: {len(network)} agents already done")

    for idx, agent in enumerate(agents):
        if agent.agent_id in network:
            continue
        rng = agent_rngs[idx]
        others = [line for i, line in enumerate(profile_lines) if i != idx]
        system_prompt = (
            f"Pretend you are a person with the following profile: "
            f"{agent.get_profile_str(**profile_kwargs)}. "
            f"You are joining a social network in "
            f"Singapore. You will be provided a list of people in the network, "
            f"where each person is described as '{schema}'. Which of these people will "
            f"you become friends with? Provide a list of *YOUR* friends in the "
            f"format ID, ID, ID, etc. Do not include any other text in your "
            f"response. Do not include any people who are not listed below."
        )
        user_prompt = (
            f"Here are the people in the social network, separated by semicolon: "
            f"{'; '.join(others)}. Please ONLY provide a list of other people you "
            f"would like to be friends with separated by commas. "
            f"DO NOT PROVIDE OTHER TEXTS"
        )

        friends = None
        for _ in range(max_try):
            try:
                response = llm.chat(system_prompt, user_prompt, temperature=0.7)
                friends = _parse_friend_indices(response, idx, len(agents))
                break
            except (RuntimeError, ValueError):
                continue

        if friends is None:
            if fallback_k > 0:
                candidates = [i for i in range(len(agents)) if i != idx]
                friends = rng.choice(candidates, size=fallback_k, replace=False).tolist()
                logger.warning(f"{agent.agent_id}: LLM failed {max_try} times, "
                               f"using {fallback_k} random friends")
            else:
                friends = []
                logger.warning(f"{agent.agent_id}: LLM failed {max_try} times, "
                               f"left edgeless (fallback_k=0)")

        # `friends` are list positions; translate them back to agent_ids for the
        # stored graph ({agent_id: [followed agent_ids]}).
        network[agent.agent_id] = [agents[i].agent_id for i in friends]
        if verbose:
            logger.info(f"{agent.agent_id}: {len(friends)} friends")
        if checkpoint_path:
            save_network(network, checkpoint_path)

    return network


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agents", default=DEFAULT_AGENTS)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--num-agents", type=int, default=None,
                        help="limit to first N agents (for cheap test runs)")
    parser.add_argument("--fallback-k", type=int, default=5,
                        help="random friends if the LLM keeps failing (0 = strict VacSim)")
    parser.add_argument("--no-area", action="store_true",
                        help="hide planning area from the friendship prompt "
                             "(with/without-area homophily comparison)")
    parser.add_argument("--persona", action="store_true",
                        help="append the narrative general_persona to each profile "
                             "(experimental; needs vLLM MAX_MODEL_LEN >= 16384)")
    args = parser.parse_args()

    setup_logger(log_path=os.path.splitext(args.output)[0] + ".log")

    agents = load_agents(args.agents)
    if args.num_agents:
        agents = agents[: args.num_agents]
    logger.info(f"Loaded {len(agents)} agents")

    llm = LLMClient()
    logger.info(f"LLM: {llm.provider} / {llm.model}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    network = generate_llm_network(agents, llm, fallback_k=args.fallback_k,
                                   include_area=not args.no_area,
                                   include_persona=args.persona,
                                   checkpoint_path=args.output)
    save_network(network, args.output)
    logger.info(f"Network saved to {args.output}")
