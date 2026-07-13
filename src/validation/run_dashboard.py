"""One-page interactive HTML dashboard for a single simulation run.

Reads a run JSON (engine.save output) plus its sibling .log (same path,
.json -> .log) and writes a fully self-contained HTML file (plotly inlined,
opens offline) to outputs/analysis/dashboard_<run_name>.html with six panels:

  1. population mean TPB trajectories (mean +/- std band) + E[intention]
  2. per-agent spaghetti (all agents, hover = agent_id) per series
  3. saturation monitor — share of agents >= 4.8 per construct per week
     (the ratchet-drift metric from the C2 drift fix)
  4. intention distribution evolution — stacked mean p1..p5 per week
  5. agent inspector — dropdown -> that agent's trajectories, memories, posts
  6. log summary — per-week wall time, LLM retries, WARNING/ERROR lines

Read-only. Reuses load_run / expected_intention / tidy from the existing
validation scripts and their palette (dataviz reference, light mode).

Run from src/:
    python validation/run_dashboard.py --run ../outputs/runs/c2_smoke_ratchet_fix.json
    python validation/run_dashboard.py --run ../outputs/runs/C0_baseline_100.json \
        --output ../outputs/analysis/my_dashboard.html
"""

import argparse
import html as html_mod
import json
import os
import re
import sys
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

# this file lives in src/validation/; put src/ on the path so imports work
# the same way they do for driver.py (which is run from src/).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from validation.compare_runs import expected_intention, load_run
from validation.plot_trajectories import tidy

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "..", "outputs")  # src/validation/ -> repo root
ANALYSIS = os.path.join(OUT, "analysis")

SATURATION_THRESHOLD = 4.8  # matches plot_trajectories' ratchet ceiling check

# ── palette (dataviz reference, light mode — same as plot_trajectories) ──────
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
# fixed per-series hues (categorical, validated): the same series keeps the
# same hue in every panel.
COLORS = {"attitude": "#2a78d6", "norm": "#1baf7a", "pbc": "#eda100",
          "E_intention": "#4a3aa7"}
NAMES = {"attitude": "Attitude", "norm": "Subjective norm", "pbc": "PBC",
         "E_intention": "E[intention]"}
# ordinal blue ramp (5 ordered intention levels, light -> dark)
ORDINAL5 = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]
LEVEL_NAMES = ["1 no intention", "2 weak", "3 uncertain", "4 likely", "5 strong"]

FONT = "Segoe UI, DejaVu Sans, sans-serif"


# ── figure helpers ───────────────────────────────────────────────────────────

def _hex_rgba(hex_color, alpha):
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
    return f"rgba({r},{g},{b},{alpha})"


def _layout(title, tmax, ylab, yrange=None, height=420):
    return go.Layout(
        title=dict(text=title, font=dict(color=INK, size=15), x=0),
        font=dict(family=FONT, color=INK2, size=12),
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        xaxis=dict(title="week", gridcolor=GRID, linecolor=AXIS, zeroline=False,
                   dtick=1 if tmax <= 15 else 2, range=[-0.3, tmax + 0.3]),
        yaxis=dict(title=ylab, gridcolor=GRID, linecolor=AXIS, zeroline=False,
                   range=yrange),
        height=height, margin=dict(l=60, r=30, t=50, b=45),
        legend=dict(orientation="h", y=1.06, x=1, xanchor="right"),
        hovermode="x unified",
    )


def _to_div(fig, include_js=False):
    return pio.to_html(
        fig, full_html=False, include_plotlyjs="inline" if include_js else False,
        default_width="100%", config={"displaylogo": False})


def fig_population(df, tmax):
    """Panel 1: mean +/- std band per series, all on the shared 1-5 scale."""
    fig = go.Figure(layout=_layout(
        "Population mean trajectories (band = ±1 std)", tmax, "score (1–5)",
        yrange=[1, 5]))
    g = df.groupby("timestep")
    for col in ("attitude", "norm", "pbc", "E_intention"):
        mean, std = g[col].mean(), g[col].std().fillna(0.0)
        t = list(mean.index)
        band = _hex_rgba(COLORS[col], 0.10)
        fig.add_trace(go.Scatter(  # band (drawn as one closed shape)
            x=t + t[::-1], y=list(mean + std) + list((mean - std))[::-1],
            fill="toself", fillcolor=band, line=dict(width=0),
            hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(
            x=t, y=list(mean), name=NAMES[col], mode="lines+markers",
            line=dict(color=COLORS[col], width=2), marker=dict(size=6),
            hovertemplate="%{y:.2f}<extra>" + NAMES[col] + "</extra>"))
    return fig


def fig_spaghetti(df, tmax):
    """Panel 2: every agent as a faint line (hover = agent_id), bold mean.

    One figure per series, stacked — plotly subplots with 400 traces get
    sluggish, so four independent figures keep each one light.
    """
    divs = []
    faint = _hex_rgba(MUTED, 0.25)
    for col in ("attitude", "norm", "pbc", "E_intention"):
        fig = go.Figure(layout=_layout(NAMES[col] + " — all agents", tmax,
                                       "score (1–5)", yrange=[0.9, 5.1],
                                       height=340))
        fig.update_layout(hovermode="closest", showlegend=False)
        wide = df.pivot(index="timestep", columns="agent_id", values=col)
        # a single-timestep run (e.g. the C0 frozen baseline) has nothing to
        # draw as a line — fall back to markers so the panel isn't blank
        mode = "lines" if len(wide.index) > 1 else "markers"
        for aid in wide.columns:
            fig.add_trace(go.Scatter(
                x=list(wide.index), y=list(wide[aid]), mode=mode,
                line=dict(color=faint, width=1), marker=dict(color=faint, size=6),
                hovertemplate=aid + ": %{y:.2f}<extra></extra>"))
        fig.add_trace(go.Scatter(
            x=list(wide.index), y=list(wide.mean(axis=1)), mode=mode,
            line=dict(color=COLORS[col], width=3),
            marker=dict(color=COLORS[col], size=10),
            hovertemplate="mean: %{y:.2f}<extra></extra>"))
        divs.append(_to_div(fig))
    return "\n".join(divs)


def fig_saturation(df, tmax):
    """Panel 3: share of agents at >= SATURATION_THRESHOLD per construct."""
    fig = go.Figure(layout=_layout(
        f"Saturation — share of agents ≥ {SATURATION_THRESHOLD} "
        "(ratchet-drift monitor)", tmax, "share of agents"))
    fig.update_layout(yaxis=dict(range=[0, 1.02], tickformat=".0%",
                                 gridcolor=GRID, linecolor=AXIS))
    for col in ("attitude", "norm", "pbc"):
        share = (df.pivot(index="timestep", columns="agent_id", values=col)
                 >= SATURATION_THRESHOLD).mean(axis=1)
        fig.add_trace(go.Scatter(
            x=list(share.index), y=list(share), name=NAMES[col],
            mode="lines+markers", line=dict(color=COLORS[col], width=2),
            marker=dict(size=6),
            hovertemplate="%{y:.0%}<extra>" + NAMES[col] + "</extra>"))
    return fig


def fig_intention_dist(run, tmax):
    """Panel 4: stacked mean p1..p5 per week (stacked bars; sums to 1)."""
    per_t = {}  # timestep -> list of dists
    for agent in run.values():
        for e in agent["belief_history"]:
            d = e.get("fertility_intention_dist")
            if d:
                per_t.setdefault(e["timestep"], []).append(d)
    if not per_t:
        return None
    ts = sorted(per_t)
    means = {t: [sum(d[i] for d in per_t[t]) / len(per_t[t]) for i in range(5)]
             for t in ts}
    fig = go.Figure(layout=_layout(
        "Intention distribution — population mean p1..p5", tmax,
        "mean probability"))
    fig.update_layout(barmode="stack", yaxis=dict(range=[0, 1.0],
                      gridcolor=GRID, linecolor=AXIS))
    for i in range(5):
        fig.add_trace(go.Bar(
            x=ts, y=[means[t][i] for t in ts], name=LEVEL_NAMES[i],
            marker=dict(color=ORDINAL5[i],
                        line=dict(color=SURFACE, width=1)),
            hovertemplate="%{y:.2f}<extra>" + LEVEL_NAMES[i] + "</extra>"))
    return fig


# ── panel 5: agent inspector (inline JS on embedded JSON) ────────────────────

def inspector_data(run):
    data = {}
    for aid in sorted(run):
        a = run[aid]
        hist = sorted(a["belief_history"], key=lambda e: e["timestep"])
        data[aid] = {
            "profile": (f"{a['age']} {a['gender']}, {a['relationship_status']}, "
                        f"{a['education']}, {a['occupation']} ({a['industry']}), "
                        f"{a['planning_area']}, financial security "
                        f"{a['financial_security_score']}/5"),
            "persona": a.get("general_persona") or "",
            "t": [e["timestep"] for e in hist],
            "attitude": [e["attitude_score"] for e in hist],
            "norm": [e["subjective_norm_score"] for e in hist],
            "pbc": [e["pbc_score"] for e in hist],
            "E_intention": [expected_intention(e.get("fertility_intention_dist"))
                            for e in hist],
            "mem": [[m["created_timestep"], m["source_type"],
                     round(m["importance"], 2), m["memory_text"]]
                    for m in sorted(a["memory_stream"],
                                    key=lambda m: m["created_timestep"])],
            "tweets": [[t.get("timestep", t.get("created_timestep", "?")),
                        t.get("text", t.get("tweet_text", ""))]
                       for t in a["tweets"]],
        }
    return data


INSPECTOR_JS = """
const DATA = %(data)s;
const COLORS = %(colors)s;
const NAMES = %(names)s;
const LAYOUT = {
  font: {family: %(font)s, color: "%(ink2)s", size: 12},
  paper_bgcolor: "%(surface)s", plot_bgcolor: "%(surface)s",
  xaxis: {title: "week", gridcolor: "%(grid)s", linecolor: "%(axis)s",
          zeroline: false, dtick: 1},
  yaxis: {title: "score (1\\u20135)", gridcolor: "%(grid)s",
          linecolor: "%(axis)s", zeroline: false, range: [0.9, 5.1]},
  height: 360, margin: {l: 60, r: 30, t: 20, b: 45},
  legend: {orientation: "h", y: 1.12, x: 1, xanchor: "right"},
  hovermode: "x unified",
};
function fillTable(id, rows) {
  const tb = document.getElementById(id);
  tb.innerHTML = "";
  if (!rows.length) {
    const tr = tb.insertRow();
    const td = tr.insertCell();
    td.colSpan = 4; td.textContent = "(none)"; td.className = "muted";
    return;
  }
  rows.forEach(r => {
    const tr = tb.insertRow();
    r.forEach(v => { tr.insertCell().textContent = v; });
  });
}
function show(aid) {
  const d = DATA[aid];
  const isNan = v => v !== v;
  const traces = ["attitude", "norm", "pbc", "E_intention"].map(k => ({
    x: d.t, y: d[k].map(v => isNan(v) ? null : v), name: NAMES[k],
    mode: "lines+markers", line: {color: COLORS[k], width: 2},
    marker: {size: 6},
  }));
  Plotly.react("insp-plot", traces, LAYOUT, {displaylogo: false});
  document.getElementById("insp-profile").textContent = d.profile;
  document.getElementById("insp-persona").textContent = d.persona;
  fillTable("insp-mem", d.mem);
  fillTable("insp-tweets", d.tweets);
}
const sel = document.getElementById("insp-select");
Object.keys(DATA).forEach(aid => {
  const o = document.createElement("option");
  o.value = aid; o.textContent = aid; sel.appendChild(o);
});
sel.onchange = () => show(sel.value);
show(sel.value);
"""


def inspector_html(run):
    data = inspector_data(run)
    js = INSPECTOR_JS % {
        # "</" -> "<\\/" so memory text can never close the <script> block
        "data": json.dumps(data, ensure_ascii=False).replace("</", "<\\/"),
        "colors": json.dumps(COLORS), "names": json.dumps(NAMES),
        "font": json.dumps(FONT), "ink2": INK2, "surface": SURFACE,
        "grid": GRID, "axis": AXIS,
    }
    return f"""
<label for="insp-select"><b>Agent:</b></label>
<select id="insp-select"></select>
<p id="insp-profile" class="profile"></p>
<p id="insp-persona" class="muted"></p>
<div id="insp-plot"></div>
<h3>Memory stream</h3>
<div class="tablewrap"><table>
  <thead><tr><th>week</th><th>source</th><th>importance</th><th>memory</th></tr></thead>
  <tbody id="insp-mem"></tbody>
</table></div>
<h3>Posts</h3>
<div class="tablewrap"><table>
  <thead><tr><th>week</th><th>post</th></tr></thead>
  <tbody id="insp-tweets"></tbody>
</table></div>
<script>{js}</script>
"""


# ── panel 6: log summary ─────────────────────────────────────────────────────

RE_WEEK_DONE = re.compile(r"Week (\d+) done in (\d+)s")
RE_ATTEMPT = re.compile(r"\(attempt (\d+),")


def parse_log(log_path):
    """Heuristic parse of the run .log (engine + fark.llm line formats)."""
    if not log_path or not os.path.exists(log_path):
        return None
    info = {"weeks": [], "llm_calls": 0, "retried_calls": 0, "problems": [],
            "header": []}
    with open(log_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            m = RE_WEEK_DONE.search(line)
            if m:
                info["weeks"].append((int(m.group(1)), int(m.group(2))))
            m = RE_ATTEMPT.search(line)
            if m:
                info["llm_calls"] += 1
                if int(m.group(1)) > 1:
                    info["retried_calls"] += 1
            if " WARNING " in line or " ERROR " in line:
                info["problems"].append(line)
            if "Simulation:" in line or "Logging to" in line:
                info["header"].append(line.split("[fark] ")[-1])
    return info


def log_html(info, log_path):
    esc = html_mod.escape
    if info is None:
        return (f'<p class="muted">No log file found at '
                f'<code>{esc(str(log_path))}</code>.</p>')
    parts = ["".join(f"<p class='muted'>{esc(h)}</p>" for h in info["header"])]
    parts.append(
        f"<p><b>{info['llm_calls']}</b> LLM calls, "
        f"<b>{info['retried_calls']}</b> needed a retry, "
        f"<b>{len(info['problems'])}</b> WARNING/ERROR lines.</p>")
    if info["weeks"]:
        secs = [s for _, s in info["weeks"]]
        rows = "".join(f"<tr><td>{w}</td><td>{s}</td></tr>"
                       for w, s in info["weeks"])
        parts.append(
            f"<p>Per-week wall time (total "
            f"{sum(secs) // 60} min {sum(secs) % 60}s, "
            f"mean {sum(secs) / len(secs):.0f}s):</p>"
            f"<div class='tablewrap'><table>"
            f"<thead><tr><th>week</th><th>seconds</th></tr></thead>"
            f"<tbody>{rows}</tbody></table></div>")
    if info["problems"]:
        items = "".join(f"<li><code>{esc(p)}</code></li>"
                        for p in info["problems"][:100])
        more = ("" if len(info["problems"]) <= 100 else
                f"<p class='muted'>... and {len(info['problems']) - 100} more</p>")
        parts.append(f"<h3>WARNING / ERROR lines</h3><ul>{items}</ul>{more}")
    else:
        parts.append("<p>No WARNING or ERROR lines — clean run.</p>")
    return "\n".join(parts)


# ── page assembly ────────────────────────────────────────────────────────────

CSS = f"""
body {{ font-family: {FONT}; background: {SURFACE}; color: {INK2};
       max-width: 1100px; margin: 0 auto; padding: 24px 32px 64px; }}
h1 {{ color: {INK}; font-size: 24px; margin-bottom: 4px; }}
h2 {{ color: {INK}; font-size: 18px; border-bottom: 1px solid {GRID};
     padding-bottom: 6px; margin-top: 40px; }}
h3 {{ color: {INK}; font-size: 14px; margin-bottom: 6px; }}
p.muted, .muted {{ color: {MUTED}; font-size: 12px; }}
p.profile {{ color: {INK}; }}
code {{ font-size: 11px; }}
select {{ font-family: inherit; font-size: 14px; padding: 2px 6px;
         margin-left: 8px; }}
.meta {{ color: {INK2}; margin-bottom: 24px; }}
.meta b {{ color: {INK}; }}
.tablewrap {{ max-height: 420px; overflow-y: auto; overflow-x: auto;
             border: 1px solid {GRID}; }}
table {{ border-collapse: collapse; width: 100%; font-size: 12px; }}
th {{ text-align: left; color: {INK}; background: {SURFACE};
     position: sticky; top: 0; border-bottom: 1px solid {AXIS};
     padding: 6px 10px; }}
td {{ border-bottom: 1px solid {GRID}; padding: 5px 10px;
     vertical-align: top; }}
"""


def build_page(run_path, log_path, state):
    run = {a["agent_id"]: a for a in state["agents"]}
    df = tidy(run)
    tmax = int(df["timestep"].max())
    tmin = int(df["timestep"].min())
    n_mem = sum(len(a["memory_stream"]) for a in run.values())
    n_tweets = sum(len(a["tweets"]) for a in run.values())
    run_name = os.path.splitext(os.path.basename(run_path))[0]

    dist_fig = fig_intention_dist(run, tmax)
    sections = [
        ("Population trajectories", _to_div(fig_population(df, tmax),
                                            include_js=True)),
        ("Per-agent trajectories", fig_spaghetti(df, tmax)),
        ("Saturation monitor", _to_div(fig_saturation(df, tmax))),
        ("Intention distribution", _to_div(dist_fig) if dist_fig else
         '<p class="muted">No fertility_intention_dist recorded yet.</p>'),
        ("Agent inspector", inspector_html(run)),
        ("Log summary", log_html(parse_log(log_path), log_path)),
    ]
    body = "\n".join(f"<h2>{i + 1}. {title}</h2>\n{content}"
                     for i, (title, content) in enumerate(sections))
    esc = html_mod.escape
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>Run dashboard — {esc(run_name)}</title>
<style>{CSS}</style></head><body>
<h1>Run dashboard — {esc(run_name)}</h1>
<p class="meta">condition <b>{esc(str(state.get('condition')))}</b> ·
<b>{len(run)}</b> agents · weeks <b>{tmin}–{tmax}</b> ·
<b>{n_mem}</b> memories · <b>{n_tweets}</b> posts ·
source <code>{esc(os.path.basename(run_path))}</code> ·
generated {datetime.now():%Y-%m-%d %H:%M}</p>
{body}
</body></html>
"""


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", required=True, help="run JSON (engine.save output)")
    ap.add_argument("--output", default=None,
                    help="output HTML path (default: "
                         "outputs/analysis/dashboard_<run_name>.html)")
    args = ap.parse_args()

    with open(args.run, encoding="utf-8") as f:
        state = json.load(f)
    log_path = os.path.splitext(args.run)[0] + ".log"
    run_name = os.path.splitext(os.path.basename(args.run))[0]
    out_path = args.output or os.path.join(ANALYSIS, f"dashboard_{run_name}.html")

    page = build_page(args.run, log_path, state)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(page)
    print(f"agents: {len(state['agents'])}, condition: {state.get('condition')}, "
          f"current_timestep: {state.get('current_timestep')}")
    print(f"saved -> {out_path}  ({os.path.getsize(out_path) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
