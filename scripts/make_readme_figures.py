"""Draw the README figure that neither Grafana nor the slides show: the reweighting counterfactual. Run: uv run --with matplotlib python scripts/make_readme_figures.py"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "img"
BLUE, ORANGE, GREY, RED, INK = "#1f4e79", "#b24a2b", "#8a8a8a", "#be0000", "#1a1a1a"
plt.rcParams.update({"font.family": "DejaVu Serif", "axes.spines.top": False, "axes.spines.right": False})


def pt(x: float, nd: int = 3) -> str:
    """Brazilian number format: decimal comma, real minus sign."""
    return f"{x:.{nd}f}".replace(".", ",").replace("-", "−")


def counterfactual() -> None:
    d = json.loads((ROOT / "docs/calibration/counterfactual.json").read_text())
    keys = [("reweighted_reference", "Referência reponderada\n(só as entradas mudaram)", BLUE),
            ("production_covariate_regime", "Produção, semanas 5-7\n(observado)", "#4a4a4a"),
            ("production_concept_weeks", "Produção, semanas 8 e 10-16\n(observado)", ORANGE)]
    fig, ax = plt.subplots(figsize=(11, 4.8))
    for i, (k, label, color) in enumerate(keys):
        v, (lo, hi) = d["point"][k], d["ci95"][k]
        ax.bar(i, v, color=color, width=0.55)
        ax.errorbar(i, v, yerr=[[v - lo], [hi - v]], color=INK, capsize=6)
        ax.text(i, hi + 0.006, pt(v), ha="center", fontweight="bold")
    ax.axhline(d["point"]["baseline"], color=GREY, ls="--")
    ax.text(0.5, d["point"]["baseline"] + 0.004, f"base {pt(d['point']['baseline'])}", ha="center", color="#4a4a4a")
    gap = d["point"]["unexplained_concept_weeks"]
    lo, hi = d["ci95"]["unexplained_concept_weeks"]
    ax.annotate("", xy=(2.38, d["point"]["production_concept_weeks"]), xytext=(2.38, d["point"]["reweighted_reference"]),
                arrowprops=dict(arrowstyle="<->", color=ORANGE))
    ax.text(2.45, 0.19, f"−{pt(gap)}\nIC 95%: {pt(lo)} a {pt(hi)}\nnão explicado\npelas entradas:\nconcept drift", color=ORANGE, va="center")
    ax.set_xticks(range(3), [k[1] for k in keys], fontsize=9)
    ax.set_xlim(-0.5, 3.6)
    ax.set_ylim(0, 0.32)
    ax.set_ylabel("AUC-PR do campeão")
    fig.savefig(OUT / "causal_contrafactual.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    counterfactual()
