"""Draw the social network as a PNG so geographic clustering is visible.

Force-directed (spring) layout pulls connected agents together, so if friends
cluster by location the same-coloured nodes form visible blobs. Nodes are
coloured by Singapore URA region (26 planning areas grouped into 5 regions for
legibility) and sized by in-degree (followers). Read-only.

Run from src/:
    ../.venv/Scripts/python validation/plot_network.py --network ../outputs/networks/social_network_qwen.json
"""

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")  # headless: write a file, don't open a window
import matplotlib.pyplot as plt
import networkx as nx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sandbox.agent import load_agents          # noqa: E402
from utils.network_utils import load_network    # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")

# 26 planning areas -> 5 URA regions (for a readable 5-colour legend)
AREA_REGION = {
    "Ang Mo Kio": "North-East", "Hougang": "North-East", "Punggol": "North-East",
    "Sengkang": "North-East", "Serangoon": "North-East",
    "Bedok": "East", "Pasir Ris": "East", "Tampines": "East",
    "Sembawang": "North", "Woodlands": "North", "Yishun": "North",
    "Bukit Batok": "West", "Bukit Panjang": "West", "Choa Chu Kang": "West",
    "Clementi": "West", "Jurong East": "West", "Jurong West": "West",
    "Bishan": "Central", "Bukit Merah": "Central", "Bukit Timah": "Central",
    "Geylang": "Central", "Kallang": "Central", "Newton": "Central",
    "Novena": "Central", "Tanglin": "Central", "Toa Payoh": "Central",
}
REGION_COLOR = {  # ColorBrewer Set1 — clearly distinguishable
    "Central": "#e41a1c", "East": "#377eb8", "North": "#4daf4a",
    "North-East": "#984ea3", "West": "#ff7f00",
}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--network", default=os.path.join(ROOT, "outputs", "networks", "social_network_qwen.json"))
    ap.add_argument("--agents", default=os.path.join(ROOT, "agents_final_100.json"))
    ap.add_argument("--out", default=os.path.join(ROOT, "outputs", "analysis", "network_validation", "network_plot.png"))
    args = ap.parse_args()

    net = load_network(args.network)
    agents = {a.agent_id: a for a in load_agents(args.agents)}

    G = nx.DiGraph()
    G.add_nodes_from(agents)
    for a, followed in net.items():
        for b in followed:
            if b in agents and b != a:
                G.add_edge(a, b)

    region = {a: AREA_REGION.get(ag.planning_area, "Central") for a, ag in agents.items()}
    indeg = dict(G.in_degree())
    node_color = [REGION_COLOR[region[a]] for a in G.nodes()]
    node_size = [60 + 45 * indeg[a] for a in G.nodes()]  # bigger = more followers

    # reproducible force-directed layout; k spreads nodes out a bit
    pos = nx.spring_layout(G, seed=42, k=0.9, iterations=200)

    plt.figure(figsize=(15, 13))
    nx.draw_networkx_edges(G, pos, alpha=0.10, edge_color="#555555",
                           arrows=False, width=0.6)
    nx.draw_networkx_nodes(G, pos, node_color=node_color, node_size=node_size,
                           linewidths=0.5, edgecolors="white")

    handles = [plt.Line2D([0], [0], marker="o", color="w", label=r,
                          markerfacecolor=c, markersize=12)
               for r, c in REGION_COLOR.items()]
    plt.legend(handles=handles, title="URA region", loc="upper left", fontsize=11)
    plt.title("Qwen social network (100 agents) — node colour = region, size = followers\n"
              "spring layout: clustering of same colours = geographic homophily "
              "(planning-area lift 6.2x)", fontsize=13)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"Saved -> {args.out}  ({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)")


if __name__ == "__main__":
    main()
