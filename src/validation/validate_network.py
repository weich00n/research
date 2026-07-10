"""Structural validation of a generated social network.

Checks the LLM-generated follow graph (see generate_social_network.py) for
plausibility before it is used in C1/C3:

  1. Degree — out-degree (follows) and in-degree (followers) distributions.
  2. Connectivity — weak components, isolated readers/broadcasters.
  3. Reciprocity — share of directed edges with a reverse edge.
  4. Clustering — mean local clustering on the undirected projection.
  5. Homophily — observed edge-level similarity on age / gender /
     relationship status / planning area / education vs a node-label
     permutation null (attributes shuffled over nodes, graph fixed), so the
     degree sequence is held constant.

Usage (from src/):
    python validation/validate_network.py \
        --network ../outputs/networks/social_network_qwen.json \
        --agents ../agents_final_100.json

Writes a text report + degree histogram PNG to outputs/analysis/.
"""

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent

N_PERMUTATIONS = 2000
RANDOM_STATE = 42


# ---------------------------------------------------------------- loading


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def edge_list(network):
    return [(a, b) for a, follows in network.items() for b in follows]


# ---------------------------------------------------------------- structure


def degree_stats(network):
    out_deg = {a: len(v) for a, v in network.items()}
    in_deg = Counter()
    for a, v in network.items():
        in_deg.update(v)
    in_deg = {a: in_deg.get(a, 0) for a in network}
    return out_deg, in_deg


def describe(values):
    vals = sorted(values)
    n = len(vals)
    mean = sum(vals) / n
    median = vals[n // 2] if n % 2 else (vals[n // 2 - 1] + vals[n // 2]) / 2
    return {"min": vals[0], "median": median, "mean": mean, "max": vals[-1]}


def reciprocity(network):
    edges = set(edge_list(network))
    if not edges:
        return 0.0
    recip = sum(1 for (a, b) in edges if (b, a) in edges)
    return recip / len(edges)


def weak_components(network):
    neigh = {a: set() for a in network}
    for a, b in edge_list(network):
        neigh[a].add(b)
        neigh.setdefault(b, set()).add(a)
    seen, comps = set(), []
    for start in neigh:
        if start in seen:
            continue
        comp, stack = set(), [start]
        while stack:
            node = stack.pop()
            if node in comp:
                continue
            comp.add(node)
            stack.extend(neigh[node] - comp)
        seen |= comp
        comps.append(comp)
    return sorted(comps, key=len, reverse=True)


def mean_clustering(network):
    """Mean local clustering coefficient on the undirected projection."""
    neigh = {a: set() for a in network}
    for a, b in edge_list(network):
        neigh[a].add(b)
        neigh.setdefault(b, set()).add(a)
    coeffs = []
    for node, nb in neigh.items():
        k = len(nb)
        if k < 2:
            coeffs.append(0.0)
            continue
        links = sum(1 for u in nb for v in nb if u < v and v in neigh[u])
        coeffs.append(2 * links / (k * (k - 1)))
    return sum(coeffs) / len(coeffs)


# ---------------------------------------------------------------- homophily


def edge_metric(edges, attr, kind):
    """Mean edge-level similarity: same-category share, or |age gap| for age."""
    if kind == "same":
        return sum(1 for a, b in edges if attr[a] == attr[b]) / len(edges)
    return sum(abs(attr[a] - attr[b]) for a, b in edges) / len(edges)


def permutation_test(edges, attr, kind, rng):
    """Permute attribute labels over nodes (graph fixed) -> null distribution."""
    observed = edge_metric(edges, attr, kind)
    nodes = list(attr)
    values = list(attr.values())
    null = []
    for _ in range(N_PERMUTATIONS):
        rng.shuffle(values)
        shuffled = dict(zip(nodes, values))
        null.append(edge_metric(edges, shuffled, kind))
    null.sort()
    mean_null = sum(null) / len(null)
    var = sum((x - mean_null) ** 2 for x in null) / (len(null) - 1)
    sd = var**0.5
    z = (observed - mean_null) / sd if sd > 0 else 0.0
    # two-sided p from the empirical null
    more_extreme = sum(1 for x in null if abs(x - mean_null) >= abs(observed - mean_null))
    p = (more_extreme + 1) / (N_PERMUTATIONS + 1)
    return observed, mean_null, z, p


HOMOPHILY_ATTRS = [
    # (label, agent key, kind)  kind: "same" = same-category share, "gap" = |diff|
    ("age (|gap| in years)", "age", "gap"),
    ("gender", "gender", "same"),
    ("relationship_status", "relationship_status", "same"),
    ("planning_area", "planning_area", "same"),
    ("education", "education", "same"),
]


# ---------------------------------------------------------------- report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--network", default=str(ROOT / "outputs/networks/social_network_qwen.json"))
    ap.add_argument("--agents", default=str(ROOT / "agents_final_100.json"))
    ap.add_argument("--out-dir", default=str(ROOT / "outputs/analysis"))
    args = ap.parse_args()

    network = load_json(args.network)
    agents = {a["agent_id"]: a for a in load_json(args.agents)}
    missing = set(network) - set(agents)
    if missing:
        raise SystemExit(f"network references unknown agents: {sorted(missing)[:5]} ...")

    edges = edge_list(network)
    out_deg, in_deg = degree_stats(network)
    comps = weak_components(network)
    rng = random.Random(RANDOM_STATE)

    name = Path(args.network).stem
    lines = [f"# Structural validation: {name}", ""]
    lines.append(f"nodes: {len(network)}   directed edges: {len(edges)}   "
                 f"density: {len(edges) / (len(network) * (len(network) - 1)):.3f}")
    o, i = describe(out_deg.values()), describe(in_deg.values())
    lines.append(f"out-degree (follows):   min {o['min']}  median {o['median']:.0f}  "
                 f"mean {o['mean']:.1f}  max {o['max']}")
    lines.append(f"in-degree (followers):  min {i['min']}  median {i['median']:.0f}  "
                 f"mean {i['mean']:.1f}  max {i['max']}")
    zero_in = sorted(a for a, d in in_deg.items() if d == 0)
    zero_out = sorted(a for a, d in out_deg.items() if d == 0)
    lines.append(f"agents nobody follows (posts unread): {len(zero_in)}"
                 + (f" -> {', '.join(zero_in)}" if zero_in else ""))
    lines.append(f"agents following nobody (read nothing): {len(zero_out)}"
                 + (f" -> {', '.join(zero_out)}" if zero_out else ""))
    lines.append(f"weak components: {len(comps)} (largest = {len(comps[0])} nodes)")
    lines.append(f"reciprocity (edge has reverse edge): {reciprocity(network):.2f}")
    lines.append(f"mean local clustering (undirected): {mean_clustering(network):.2f}")
    lines.append("")
    lines.append(f"## Homophily vs node-label permutation null (n={N_PERMUTATIONS}, seed {RANDOM_STATE})")
    lines.append(f"{'attribute':28s} {'observed':>9s} {'null':>9s} {'z':>7s} {'p':>8s}")
    for label, key, kind in HOMOPHILY_ATTRS:
        attr = {a: agents[a][key] for a in network}
        obs, null_mean, z, p = permutation_test(edges, attr, kind, rng)
        note = ""
        if kind == "gap":
            note = "  (negative z = assortative: smaller age gaps than chance)"
        lines.append(f"{label:28s} {obs:9.3f} {null_mean:9.3f} {z:7.2f} {p:8.4f}{note}")
    lines.append("")
    lines.append("Interpretation: 'same'-kind rows are the share of edges joining "
                 "same-category agents (positive z = homophily); the age row is the "
                 "mean absolute age gap across edges (negative z = homophily).")

    report = "\n".join(lines)
    print(report)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / f"network_validation_{name}.md"
    report_path.write_text(report + "\n", encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, deg, title in [(axes[0], out_deg, "out-degree (follows)"),
                           (axes[1], in_deg, "in-degree (followers)")]:
        vals = list(deg.values())
        ax.hist(vals, bins=range(0, max(vals) + 2), edgecolor="white")
        ax.set_title(title)
        ax.set_xlabel("degree")
        ax.set_ylabel("agents")
    fig.suptitle(f"Degree distributions: {name}")
    fig.tight_layout()
    png_path = out_dir / f"network_validation_{name}.png"
    fig.savefig(png_path, dpi=150)

    print(f"\nsaved -> {report_path}")
    print(f"saved -> {png_path}")


if __name__ == "__main__":
    main()
