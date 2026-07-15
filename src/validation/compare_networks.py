"""Side-by-side structural comparison of two (or more) social networks.

Built for the with/without-planning-area experiment: does hiding the residence
field from the friendship prompt remove the implausibly strong geographic
homophily (~31% same-area ties vs ~5% chance) without degrading the rest of the
structure (degree, reciprocity, clustering, other homophily dimensions)?

Usage (from src/):
    python validation/compare_networks.py \
        --networks ../outputs/networks/social_network_qwen.json \
                   ../outputs/networks/social_network_qwen_noarea.json \
        --labels with_area no_area

Writes a text report to outputs/analysis/.
"""

import argparse
import random
from pathlib import Path

from validate_network import (HOMOPHILY_ATTRS, N_PERMUTATIONS, RANDOM_STATE,
                              describe, degree_stats, edge_list, load_json,
                              mean_clustering, permutation_test, reciprocity,
                              weak_components)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent


def analyse(network, agents):
    edges = edge_list(network)
    out_deg, in_deg = degree_stats(network)
    rng = random.Random(RANDOM_STATE)
    row = {
        "nodes": len(network),
        "edges": len(edges),
        "out-deg mean": f"{describe(out_deg.values())['mean']:.1f}",
        "in-deg max": describe(in_deg.values())["max"],
        "zero in-degree": sum(1 for d in in_deg.values() if d == 0),
        "weak components": len(weak_components(network)),
        "reciprocity": f"{reciprocity(network):.2f}",
        "clustering": f"{mean_clustering(network):.2f}",
    }
    for label, key, kind in HOMOPHILY_ATTRS:
        attr = {a: agents[a][key] for a in network}
        obs, _, z, _ = permutation_test(edges, attr, kind, rng)
        row[f"{label}"] = f"{obs:.3f} (z {z:+.1f})"
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--networks", nargs="+", required=True)
    ap.add_argument("--labels", nargs="+", default=None)
    ap.add_argument("--agents", default=str(ROOT / "agents_final_100.json"))
    ap.add_argument("--out-dir", default=str(ROOT / "outputs/analysis/network_validation"))
    args = ap.parse_args()

    labels = args.labels or [Path(p).stem for p in args.networks]
    if len(labels) != len(args.networks):
        raise SystemExit("--labels must match --networks in length")

    agents = {a["agent_id"]: a for a in load_json(args.agents)}
    rows = {lab: analyse(load_json(p), agents) for lab, p in zip(labels, args.networks)}

    metrics = list(next(iter(rows.values())))
    width = max(len(m) for m in metrics) + 2
    col = max(max(len(str(r[m])) for r in rows.values() for m in metrics),
              max(len(l) for l in labels)) + 2

    lines = [f"# Network comparison (homophily null: n={N_PERMUTATIONS} permutations, "
             f"seed {RANDOM_STATE})", ""]
    lines.append(" " * width + "".join(f"{l:>{col}}" for l in labels))
    for m in metrics:
        lines.append(f"{m:<{width}}" + "".join(f"{str(rows[l][m]):>{col}}" for l in labels))
    lines.append("")
    lines.append("Homophily rows: observed value (z vs permutation null). "
                 "'same'-share rows: higher = more homophilous; the age row is the "
                 "mean |age gap| in years: lower = more homophilous (negative z).")

    report = "\n".join(lines)
    print(report)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"network_comparison_{'_vs_'.join(labels)}.md"
    path.write_text(report + "\n", encoding="utf-8")
    print(f"\nsaved -> {path}")


if __name__ == "__main__":
    main()
