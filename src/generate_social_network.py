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
import re

import numpy as np

from sandbox.agent import load_agents
from utils.generate_utils import LLMClient
from utils.logging_utils import get_logger, setup_logger
from utils.network_utils import save_network

RANDOM_STATE = 42

logger = get_logger("network")

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_AGENTS = os.path.join(HERE, "..", "agents_final_100.json")
DEFAULT_OUTPUT = os.path.join(HERE, "..", "outputs", "networks", "social_network.json")


def _parse_friend_indices(response, self_idx, num_agents):
    """Pull friend indices out of the LLM's free-text reply.

    Grabs every run of digits (`\\d+`), then keeps those that are valid: not the
    agent itself, in range [0, num_agents), and not already seen. Brittle by
    design — it trusts the prompt's "IDs only" instruction; stray numbers in the
    text (e.g. "agent 12") would be misread. Raises if nothing valid is found so
    the caller can retry.
    """
    indices = []
    for token in re.findall(r"\d+", response):
        i = int(token)
        if i != self_idx and 0 <= i < num_agents and i not in indices:
            indices.append(i)
    if not indices:
        raise ValueError(f"No valid friend indices in response: {response[:100]!r}")
    return indices


def generate_llm_network(agents, llm, max_try=10, fallback_k=5, seed=RANDOM_STATE,
                         verbose=True):
    """Return {agent_id: [followed agent_ids]} chosen by the LLM per agent.

    If the LLM fails max_try times for an agent (VacSim leaves them edgeless),
    we instead fall back to `fallback_k` seeded-random friends so no agent is
    isolated; set fallback_k=0 for strict VacSim behaviour.
    """
    rng = np.random.default_rng(seed)
    profile_lines = [f"{i}. {a.get_profile_str()}" for i, a in enumerate(agents)]
    network = {}

    for idx, agent in enumerate(agents):
        others = [line for i, line in enumerate(profile_lines) if i != idx]
        system_prompt = (
            f"Pretend you are a person with the following profile: "
            f"{agent.get_profile_str()}. You are joining a social network in "
            f"Singapore. You will be provided a list of people in the network, "
            f"where each person is described as 'ID. Gender\tAge\tRelationship\t"
            f"Education\tOccupation\tIndustry\tArea'. Which of these people will "
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

    return network


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agents", default=DEFAULT_AGENTS)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--num-agents", type=int, default=None,
                        help="limit to first N agents (for cheap test runs)")
    parser.add_argument("--fallback-k", type=int, default=5,
                        help="random friends if the LLM keeps failing (0 = strict VacSim)")
    args = parser.parse_args()

    setup_logger(log_path=os.path.splitext(args.output)[0] + ".log")

    agents = load_agents(args.agents)
    if args.num_agents:
        agents = agents[: args.num_agents]
    logger.info(f"Loaded {len(agents)} agents")

    llm = LLMClient()
    logger.info(f"LLM: {llm.provider} / {llm.model}")

    network = generate_llm_network(agents, llm, fallback_k=args.fallback_k)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    save_network(network, args.output)
    logger.info(f"Network saved to {args.output}")
