"""Sanity-check the LLM-generated social network (does it 'make sense'?).

Read-only. Loads a network ({agent_id: [followed ids]}, an A->B = "A follows B"
directed graph) + the agent profiles, and reports:

  1. Structure   — nodes, edges, density, in/out-degree, isolates.
  2. Reciprocity — fraction of follows that are mutual.
  3. Homophily   — do connected agents share attributes (area, relationship,
     gender, education) or ages MORE than a random pair would? Observed
     edge same-rate vs the random-mixing baseline (lift > 1 = homophily).
  4. Popularity  — most/least-followed agents.
  5. A few agents' friend lists side-by-side with their profile, to eyeball.

Run from src/:
    python validation/inspect_network.py
    python validation/inspect_network.py --network ../outputs/networks/social_network_qwen.json
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sandbox.agent import load_agents  # noqa: E402
from utils.network_utils import load_network  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
DEFAULT_NETWORK = os.path.join(ROOT, "outputs", "networks", "social_network_qwen.json")
DEFAULT_AGENTS = os.path.join(ROOT, "agents_final_100.json")

CAT_ATTRS = ["planning_area", "relationship_status", "gender", "education"]
SAMPLE = ["agent_001", "agent_050", "agent_100"]


def cat_baseline(values):
    """P(two distinct random nodes share this categorical value) under random mixing."""
    _, counts = np.unique(values, return_counts=True)
    n = len(values)
    return float(sum(c * (c - 1) for c in counts) / (n * (n - 1)))


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--network", default=DEFAULT_NETWORK)
    ap.add_argument("--agents", default=DEFAULT_AGENTS)
    ap.add_argument("--report", default=os.path.join(ROOT, "outputs", "analysis", "network_validation", "network_inspection.md"))
    args = ap.parse_args()

    net = load_network(args.network)
    agents = {a.agent_id: a for a in load_agents(args.agents)}
    out = []
    p = out.append

    # ── coverage / validity ─────────────────────────────────────────────────
    nodes = set(agents)
    keyed = set(net)
    missing_keys = sorted(nodes - keyed)          # agents with no entry at all
    extra_keys = sorted(keyed - nodes)            # entries for unknown agents
    edges = []  # (follower, followed)
    bad_targets = self_loops = 0
    for a, followed in net.items():
        for b in followed:
            if b == a:
                self_loops += 1
            elif b not in agents:
                bad_targets += 1
            else:
                edges.append((a, b))
    E = len(edges)
    N = len(agents)

    p(f"# Social network inspection — {os.path.basename(args.network)}\n")
    p(f"Agents (profile file): {N} | network entries: {len(net)}")
    p(f"Valid directed edges (A follows B): {E} | density {E / (N * (N - 1)):.3f}")
    if missing_keys:
        p(f"⚠ agents with NO network entry ({len(missing_keys)}): {missing_keys[:10]}")
    if extra_keys:
        p(f"⚠ network entries for unknown agents: {extra_keys[:10]}")
    if self_loops:
        p(f"⚠ self-loops: {self_loops}")
    if bad_targets:
        p(f"⚠ edges to non-existent agents: {bad_targets}")
    p("")

    # ── degree ──────────────────────────────────────────────────────────────
    outdeg = {a: 0 for a in agents}
    indeg = {a: 0 for a in agents}
    for a, b in edges:
        outdeg[a] += 1
        indeg[b] += 1
    od = np.array(list(outdeg.values()))
    idg = np.array(list(indeg.values()))
    p("## Degree")
    p(f"- out-degree (friends chosen): mean {od.mean():.1f}, median {np.median(od):.0f}, "
      f"min {od.min()}, max {od.max()} | agents following nobody: {int((od == 0).sum())}")
    p(f"- in-degree (followers):       mean {idg.mean():.1f}, median {np.median(idg):.0f}, "
      f"min {idg.min()}, max {idg.max()} | agents nobody follows: {int((idg == 0).sum())}")
    p("")

    # ── reciprocity ─────────────────────────────────────────────────────────
    edge_set = set(edges)
    mutual = sum(1 for a, b in edges if (b, a) in edge_set)
    p("## Reciprocity")
    p(f"- mutual follows: {mutual}/{E} edges ({100 * mutual / E:.0f}%) are reciprocated\n")

    # ── homophily ───────────────────────────────────────────────────────────
    p("## Homophily (do friends resemble each other more than chance?)")
    p("| attribute | edge same-rate | random baseline | lift |\n|---|---|---|---|")
    for attr in CAT_ATTRS:
        vals = {a: getattr(ag, attr) for a, ag in agents.items()}
        same = np.mean([vals[a] == vals[b] for a, b in edges])
        base = cat_baseline([vals[a] for a in agents])
        lift = same / base if base else float("nan")
        p(f"| {attr} | {same:.3f} | {base:.3f} | {lift:.2f}× |")
    # age (numeric): mean |Δage| over edges vs all distinct pairs
    age = {a: ag.age for a, ag in agents.items()}
    edge_dage = np.mean([abs(age[a] - age[b]) for a, b in edges])
    allpairs = [abs(age[a] - age[b]) for a in agents for b in agents if a != b]
    p(f"\n- age gap |Δ|: {edge_dage:.1f} yrs over friendships vs "
      f"{np.mean(allpairs):.1f} yrs for random pairs "
      f"({'younger gap = age homophily' if edge_dage < np.mean(allpairs) else 'no age homophily'})\n")

    # ── popularity ──────────────────────────────────────────────────────────
    top = sorted(agents, key=lambda a: indeg[a], reverse=True)[:5]
    p("## Popularity (most-followed)")
    for a in top:
        ag = agents[a]
        p(f"- {a} ({ag.age}{ag.gender[0]}, {ag.relationship_status}, {ag.occupation}, "
          f"{ag.planning_area}): {indeg[a]} followers")
    p("")

    # ── eyeball a few friend lists ──────────────────────────────────────────
    p("## Sample friend lists")
    for a in SAMPLE:
        if a not in net:
            continue
        ag = agents[a]
        p(f"\n**{a}** — {ag.age}{ag.gender[0]}, {ag.relationship_status}, {ag.occupation}, "
          f"{ag.planning_area} → follows {len(net[a])}:")
        for b in net[a][:8]:
            if b in agents:
                fb = agents[b]
                p(f"  - {b}: {fb.age}{fb.gender[0]}, {fb.relationship_status}, "
                  f"{fb.occupation}, {fb.planning_area}")

    report = "\n".join(out)
    print(report)
    with open(args.report, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nSaved -> {args.report}")


if __name__ == "__main__":
    main()
