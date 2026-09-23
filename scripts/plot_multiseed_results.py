"""Regenerate multi-seed ablation curves from docs/results/multiseed_results.json.

The plotted uncertainty is the sample standard deviation across independent
training seeds. No external human benchmark is drawn because the report does not
use a protocol-matched human comparison.
"""
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA = json.loads((ROOT / "docs/results/multiseed_results.json").read_text(encoding="utf-8"))
OUT = ROOT / "figures/report"
OUT.mkdir(parents=True, exist_ok=True)

CHECKPOINTS = {
    "epsilon": [300000, 600000, 900000, 1200000, 1500000, 1800000],
    "rally": [600000, 900000, 1200000, 1500000, 1800000],
    "potential": [600000, 900000, 1200000, 1500000, 1800000],
    "contact": [600000, 900000, 1200000, 1500000, 1800000],
}
LABELS = {
    "epsilon": {"linear": "Lineare", "exponential": "Esponenziale", "constant": "Costante 0.10"},
    "rally": {"small": "Piccolo", "moderate": "Moderato", "aggressive": "Aggressivo"},
    "potential": {"small": r"$\kappa=0.02$", "moderate": r"$\kappa=0.10$", "aggressive": r"$\kappa=0.50$"},
    "contact": {"small": r"$c=0.02$", "moderate": r"$c=0.10$", "aggressive": r"$c=0.50$"},
}
TITLES = {
    "epsilon": "Ablation della schedule ε-greedy",
    "rally": "Reward shaping - durata del rally",
    "potential": "Potential-Based Reward Shaping",
    "contact": "Reward shaping - bonus di contatto",
}

REPORT_FILENAMES = {
    "epsilon": "10_epsilon_multiseed_evaluation.png",
    "rally": "11_rally_multiseed_evaluation.png",
    "potential": "12_pbrs_multiseed_evaluation.png",
    "contact": "13_contact_multiseed_evaluation.png",
}

for family, checkpoints in CHECKPOINTS.items():
    x = np.asarray(checkpoints, dtype=float) / 1e6
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    for variant, label in LABELS[family].items():
        rows = DATA[family]["variants"][variant]["checkpoints"]
        means = np.asarray([rows[str(step)]["mean"] for step in checkpoints], dtype=float)
        stds = np.asarray([rows[str(step)]["std_across_training_seeds"] for step in checkpoints], dtype=float)
        line, = ax.plot(x, means, marker="o", linewidth=2, label=label)
        ax.fill_between(x, means - stds, means + stds, alpha=0.15)
        for seed in DATA[family]["training_seeds"]:
            vals = [rows[str(step)]["per_seed"][str(seed)] for step in checkpoints]
            ax.plot(x, vals, linewidth=0.8, alpha=0.22, color=line.get_color())
    ax.set_xlabel("Environment steps (milioni)")
    ax.set_ylabel("Return Atari greedy")
    ax.set_title(TITLES[family] + " — media ± std tra 3 training seed")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / REPORT_FILENAMES[family], dpi=220)
    plt.close(fig)

print("Figure salvate in", OUT)
