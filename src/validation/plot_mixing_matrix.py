"""Region friendship mixing matrix — the crisp homophily figure.

A 5x5 heatmap of observed / expected directed edges between Singapore URA
regions. Expected assumes random mixing (edges placed proportional to region
sizes). The DIAGONAL lighting up (ratio > 1) = within-region friendships are
over-represented = geographic homophily, in one glance (clearer than the
force-directed node-link plot). Read-only.

Run from src/:
    ../.venv/Scripts/python validation/plot_mixing_matrix.py
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sandbox.agent import load_agents            # noqa: E402
from utils.network_utils import load_network      # noqa: E402
from validation.plot_network import AREA_REGION   # noqa: E402  (reuse the area->region map)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
REGIONS = ["Central", "East", "North", "North-East", "West"]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--network", default=os.path.join(ROOT, "outputs", "networks", "social_network_qwen.json"))
    ap.add_argument("--agents", default=os.path.join(ROOT, "agents_final_100.json"))
    ap.add_argument("--out", default=os.path.join(ROOT, "outputs", "analysis", "network_validation", "network_mixing_matrix.png"))
    args = ap.parse_args()

    net = load_network(args.network)
    agents = {a.agent_id: a for a in load_agents(args.agents)}
    region = {aid: AREA_REGION.get(ag.planning_area, "Central") for aid, ag in agents.items()}
    ri = {r: i for i, r in enumerate(REGIONS)}

    # observed directed edges region[a] -> region[b]
    obs = np.zeros((5, 5))
    E = 0
    for a, followed in net.items():
        if a not in region:
            continue
        for b in followed:
            if b in region and b != a:
                obs[ri[region[a]], ri[region[b]]] += 1
                E += 1

    # expected under random mixing: E * (n_i/N) * (n_j/N)
    n = np.array([sum(1 for r in region.values() if r == reg) for reg in REGIONS], dtype=float)
    N = n.sum()
    exp = E * np.outer(n / N, n / N)
    ratio = np.divide(obs, exp, out=np.zeros_like(obs), where=exp > 0)

    fig, ax = plt.subplots(figsize=(8, 7))
    norm = TwoSlopeNorm(vmin=0, vcenter=1.0, vmax=max(2.0, ratio.max()))
    im = ax.imshow(ratio, cmap="RdBu_r", norm=norm)
    ax.set_xticks(range(5)); ax.set_xticklabels(REGIONS, rotation=30, ha="right")
    ax.set_yticks(range(5)); ax.set_yticklabels(REGIONS)
    ax.set_xlabel("friend's region (followed)")
    ax.set_ylabel("agent's region (follower)")
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{ratio[i, j]:.1f}×\n(n={int(obs[i, j])})",
                    ha="center", va="center", fontsize=9,
                    color="white" if abs(ratio[i, j] - 1) > 0.8 else "black")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("observed / expected  (1.0 = random mixing)")
    ax.set_title("Region friendship mixing — observed ÷ expected\n"
                 "diagonal > 1 = geographic homophily (within-region over-represented)",
                 fontsize=12)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")

    diag = float(np.mean([ratio[i, i] for i in range(5)]))
    offdiag = float(np.mean([ratio[i, j] for i in range(5) for j in range(5) if i != j]))
    print(f"Saved -> {args.out}")
    print(f"mean diagonal (within-region) ratio: {diag:.2f}×  |  "
          f"mean off-diagonal (cross-region): {offdiag:.2f}×  |  edges={E}")


if __name__ == "__main__":
    main()
