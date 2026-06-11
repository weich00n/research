"""Save/load helpers for the social network (VacSim: utils/network_utils.py).

The network is a plain dict {agent_id: [followed agent_ids]} stored as JSON.
An edge A -> B means A follows B, i.e. A reads B's posts (posted at t-1) at
timestep t.
"""

import json


def save_network(network, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(network, f, indent=2)


def load_network(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)
