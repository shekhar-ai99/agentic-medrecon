"""Generate the three figures for the medication-reconciliation paper.

Style matches the reference RAG paper: clean IEEE look, matplotlib defaults
with light grid, colourblind-friendly palette. Numbers come from the paper's
tables (placeholder values) so the figures are internally consistent with the
text. Replace with real MIMIC-III output when available.

  Fig 1: Reconciliation-score distribution (violin, mirrors RAG Fig 3)
  Fig 2: Discrepancy-detection performance across systems (grouped bars +
         a Precision@k-style panel, mirrors RAG Fig 4)
  Fig 3: Stage-attributed error breakdown + per-agent confidence heatmap
         (mirrors RAG Fig 2 heatmap idea)
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams.update({
    "font.size": 11,
    "font.family": "serif",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "axes.axisbelow": True,
    "figure.dpi": 150,
})

# Palette (colourblind-safe, echoing the RAG paper's orange/green/gray)
ORANGE = "#E8834E"
GREEN = "#4CA98F"
GRAY = "#9AA0A6"
BLUE = "#5B8BD0"
DARKBLUE = "#3A5A8C"

rng = np.random.default_rng(42)


# ============================================================
# FIG 1: Reconciliation-score distribution (violin)
# Mirrors RAG paper Fig 3. Mean score ~0.85 (abstract), rescaled 1-5 like
# the reference for interpretability. Two modes = single-source vs
# multi-source admissions, echoing the RAG paper's narrative.
# ============================================================
def fig_score_distribution(path):
    # Build a distribution centred to give mean ~3.4 on a 1-5 scale
    # (0.85 * 4 + 1 = 4.4 max anchor; we keep the RAG paper's 3.39 mean look).
    n = 750
    mode_hi = rng.normal(3.7, 0.28, int(n * 0.55))   # single-source admissions
    mode_lo = rng.normal(3.05, 0.30, int(n * 0.45))  # multi-source syntheses
    data = np.clip(np.concatenate([mode_hi, mode_lo]), 1.0, 5.0)
    mean, median = data.mean(), np.median(data)

    fig, ax = plt.subplots(figsize=(7, 5.2))
    parts = ax.violinplot(data, positions=[1], widths=0.8,
                          showmeans=False, showmedians=False,
                          showextrema=False)
    for pc in parts["bodies"]:
        pc.set_facecolor(BLUE)
        pc.set_alpha(0.35)
        pc.set_edgecolor(DARKBLUE)
        pc.set_linewidth(1.2)

    # jittered scatter of the points
    x = rng.normal(1.0, 0.05, len(data))
    ax.scatter(x, data, s=14, color=DARKBLUE, alpha=0.45, zorder=3,
               edgecolors="none")

    # quartile whisker
    q1, q3 = np.percentile(data, [25, 75])
    ax.vlines(1, q1, q3, color=DARKBLUE, lw=6, alpha=0.5, zorder=2)
    ax.vlines(1, data.min(), data.max(), color=DARKBLUE, lw=1.2, zorder=2)

    ax.axhline(mean, color="#C0392B", lw=2, ls="--",
               label=f"Mean = {mean:.2f}")
    ax.axhline(median, color=GREEN, lw=2, ls=":",
               label=f"Median = {median:.2f}")

    ax.set_ylim(1, 5)
    ax.set_xlim(0.3, 1.7)
    ax.set_xticks([1])
    ax.set_xticklabels(["All admissions"])
    ax.set_ylabel("Reconciliation Score $\\mathcal{R}$ (1–5 scale)")
    ax.set_title("Reconciliation Score Distribution (n = 750)",
                 fontweight="bold", pad=12)
    ax.legend(loc="upper right", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}  (mean={mean:.2f}, median={median:.2f})")


# ============================================================
# FIG 2: Discrepancy-detection performance across systems
# Mirrors RAG paper Fig 4 (Precision@k). Left: grouped P/R/F1 bars from
# Table (Overall performance). Right: a Precision@k-style curve showing how
# F1 holds as the reconciliation task scales in list length.
# ============================================================
def fig_performance(path):
    systems = ["Rule-based", "Monolithic\nLLM", "Pipeline\n(no orch.)", "MARS\n(ours)"]
    precision = [0.68, 0.79, 0.81, 0.87]   # using F1 col from Table as proxy row
    # From Table (Overall performance): Discrepancy F1 column
    f1 = [0.68, 0.79, 0.81, 0.87]
    norm_acc = [0.79, 0.84, 0.88, 0.91]
    r_corr = [np.nan, 0.68, 0.74, 0.83]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # --- Left: grouped bars ---
    x = np.arange(len(systems))
    w = 0.26
    ax1.bar(x - w, f1, w, label="Discrepancy F1", color=ORANGE, edgecolor="black", lw=0.6)
    ax1.bar(x, norm_acc, w, label="Normalization Acc.", color=GREEN, edgecolor="black", lw=0.6)
    r_plot = [0 if np.isnan(v) else v for v in r_corr]
    bars = ax1.bar(x + w, r_plot, w, label="Score corr. $r$", color=BLUE, edgecolor="black", lw=0.6)
    # mark the N/A for rule-based
    ax1.text(0 + w, 0.02, "N/A", ha="center", va="bottom", fontsize=8, rotation=90, color=GRAY)

    for i, v in enumerate(f1):
        ax1.text(i - w, v + 0.012, f"{v:.2f}", ha="center", fontsize=8)

    ax1.set_xticks(x)
    ax1.set_xticklabels(systems, fontsize=9)
    ax1.set_ylabel("Score")
    ax1.set_ylim(0, 1.0)
    ax1.set_title("Reconciliation Performance by System", fontweight="bold")
    ax1.legend(loc="upper left", fontsize=9, framealpha=0.9)

    # --- Right: F1 vs medication-list length (Precision@k analogue) ---
    list_len = [3, 5, 8, 12, 20]
    mars = [0.91, 0.89, 0.87, 0.84, 0.79]
    noorch = [0.86, 0.83, 0.81, 0.77, 0.71]
    rule = [0.58, 0.52, 0.47, 0.42, 0.36]

    ax2.plot(list_len, mars, "-o", color=ORANGE, lw=2.2, ms=7,
             label="MARS (ours)")
    ax2.plot(list_len, noorch, "--s", color=GREEN, lw=1.8, ms=6,
             label="Pipeline (no orch.)")
    ax2.plot(list_len, rule, ":^", color=GRAY, lw=1.8, ms=6,
             label="Rule-based")
    for xv, yv in zip(list_len, mars):
        ax2.annotate(f"{yv:.2f}", (xv, yv), textcoords="offset points",
                     xytext=(0, 9), ha="center", fontsize=8, color=ORANGE)

    ax2.set_xlabel("Medications per admission")
    ax2.set_ylabel("Discrepancy-detection F1")
    ax2.set_ylim(0.3, 1.0)
    ax2.set_xticks(list_len)
    ax2.set_title("F1 vs. Medication-List Length", fontweight="bold")
    ax2.legend(loc="lower left", fontsize=9, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


# ============================================================
# FIG 3: Stage-attributed error breakdown + per-agent confidence heatmap
# Mirrors RAG paper Fig 2 (heatmap). Left: horizontal error-type bars from
# the error taxonomy table. Right: a per-agent x error-type confidence
# heatmap showing which agent is implicated in which error class.
# ============================================================
def fig_error_and_heatmap(path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2),
                                   gridspec_kw={"width_ratios": [1, 1.25]})

    # --- Left: error taxonomy bars (from Table: stage-attributed errors) ---
    stages = ["Extraction", "Normalization", "Discrepancy", "Interaction", "Orchestration"]
    counts = [31, 27, 19, 14, 7]
    pcts = [4.1, 3.6, 2.5, 1.9, 0.9]
    colors = [ORANGE, GREEN, BLUE, "#B084C7", GRAY]

    y = np.arange(len(stages))[::-1]
    ax1.barh(y, counts, color=colors, edgecolor="black", lw=0.6)
    for yi, c, p in zip(y, counts, pcts):
        ax1.text(c + 0.5, yi, f"{c}  ({p}%)", va="center", fontsize=9)
    ax1.set_yticks(y)
    ax1.set_yticklabels(stages)
    ax1.set_xlabel("Error count (of 98 total, on 750 admissions)")
    ax1.set_xlim(0, 38)
    ax1.set_title("Stage-Attributed Error Taxonomy", fontweight="bold")
    ax1.grid(axis="y", alpha=0)

    # --- Right: agent x error-type confidence/attribution heatmap ---
    # Rows = agents, Cols = error classes. Value = share of that error class
    # attributable to that agent (illustrative, sums ~1 down each column).
    agents = ["Extraction", "Normalization", "Discrepancy", "Interaction", "Orchestrator"]
    err_classes = ["Missed\nmention", "Wrong\nRxNorm", "Misaligned\ndiff",
                   "Bad DDI\nflag", "Wrong\nconflict-res"]
    M = np.array([
        [0.78, 0.10, 0.06, 0.02, 0.04],   # extraction agent
        [0.08, 0.80, 0.12, 0.05, 0.06],   # normalization agent
        [0.06, 0.06, 0.74, 0.03, 0.10],   # discrepancy agent
        [0.02, 0.02, 0.03, 0.86, 0.05],   # interaction agent
        [0.06, 0.02, 0.05, 0.04, 0.75],   # orchestrator
    ])
    im = ax2.imshow(M, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")
    ax2.set_xticks(np.arange(len(err_classes)))
    ax2.set_xticklabels(err_classes, fontsize=8)
    ax2.set_yticks(np.arange(len(agents)))
    ax2.set_yticklabels(agents, fontsize=9)
    ax2.set_xlabel("Error class")
    ax2.set_ylabel("Responsible agent")
    ax2.set_title("Error Attribution by Agent", fontweight="bold")
    for i in range(len(agents)):
        for j in range(len(err_classes)):
            v = M[i, j]
            ax2.text(j, i, f"{v:.2f}", ha="center", va="center",
                     color="white" if v > 0.5 else "black", fontsize=8)
    cbar = fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    cbar.set_label("Share of error class", fontsize=9)
    ax2.grid(False)

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


if __name__ == "__main__":
    import os
    out = "/home/claude/figs"
    os.makedirs(out, exist_ok=True)
    print("Generating figures...")
    fig_score_distribution(f"{out}/fig1_score_distribution.pdf")
    fig_performance(f"{out}/fig2_performance.pdf")
    fig_error_and_heatmap(f"{out}/fig3_error_attribution.pdf")
    # also PNG previews
    fig_score_distribution(f"{out}/fig1_score_distribution.png")
    fig_performance(f"{out}/fig2_performance.png")
    fig_error_and_heatmap(f"{out}/fig3_error_attribution.png")
    print("Done.")
