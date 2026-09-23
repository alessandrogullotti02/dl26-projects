"""Regenerate report-ready DQN/DDQN evaluation figures.

The numerical source is ``docs/results/checkpoint_metrics.json``.  Primary
comparisons use the common training seeds {42, 2, 3}; the script selects only
rows marked with ``variant == 'seeded'``.

The script writes canonical reproducible figures directly to
``figures/report``. It also creates a paired final-checkpoint comparison that
makes the seed-by-seed DQN -> DDQN change explicit.
"""

from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "docs" / "results" / "checkpoint_metrics.json"
REPORT_ROOT = ROOT / "figures" / "report"
COMMON_SEEDS = [42, 2, 3]


def _load_metrics():
    return json.loads(DATA_PATH.read_text(encoding="utf-8"))["metrics"]


def collect(experiment: str):
    metrics = _load_metrics()
    seeded = [
        row
        for row in metrics
        if row.get("experiment") == experiment and row.get("variant") == "seeded"
    ]

    available_seeds = {int(row["training_seed"]) for row in seeded}
    if available_seeds != set(COMMON_SEEDS):
        raise ValueError(
            f"{experiment}: expected primary seeds {COMMON_SEEDS}, "
            f"found {sorted(available_seeds)}"
        )

    # Only checkpoints shared by all three seeds are plotted.  This matters for
    # DDQN, whose stored evaluation cadence is coarser than the baseline one.
    per_seed_steps = {
        seed: {int(row["step"]) for row in seeded if int(row["training_seed"]) == seed}
        for seed in COMMON_SEEDS
    }
    common_steps = sorted(set.intersection(*(per_seed_steps[s] for s in COMMON_SEEDS)))
    if not common_steps:
        raise ValueError(f"{experiment}: no common checkpoints across seeds")

    values = np.asarray(
        [
            [
                next(
                    float(row["mean"])
                    for row in seeded
                    if int(row["training_seed"]) == seed and int(row["step"]) == step
                )
                for step in common_steps
            ]
            for seed in COMMON_SEEDS
        ],
        dtype=float,
    )
    return COMMON_SEEDS, common_steps, values


def plot_learning_curve(experiment: str, title: str, report_name: str):
    seeds, steps, values = collect(experiment)
    mean = values.mean(axis=0)
    std = values.std(axis=0, ddof=1)
    x = np.asarray(steps, dtype=float) / 1e6

    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    line, = ax.plot(x, mean, marker="o", linewidth=2.2, label="Media su 3 training seed")
    ax.fill_between(x, mean - std, mean + std, alpha=0.16, label="± 1 std tra seed")
    for index, seed in enumerate(seeds):
        ax.plot(
            x,
            values[index],
            linestyle="--",
            linewidth=1.0,
            alpha=0.6,
            color=line.get_color(),
            label=f"Seed {seed}",
        )

    ax.set_xlabel("Environment steps (milioni)")
    ax.set_ylabel("Return Atari greedy")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()

    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    out = REPORT_ROOT / report_name
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return out


def _final_per_seed(experiment: str):
    metrics = _load_metrics()
    result = {}
    for seed in COMMON_SEEDS:
        rows = [
            row
            for row in metrics
            if row.get("experiment") == experiment
            and row.get("variant") == "seeded"
            and int(row.get("training_seed")) == seed
        ]
        if not rows:
            raise ValueError(f"{experiment}: no rows for seed {seed}")
        final = max(rows, key=lambda row: int(row["step"]))
        result[seed] = float(final["mean"])
    return result


def plot_paired_final():
    baseline = _final_per_seed("baseline")
    ddqn = _final_per_seed("ddqn")

    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    x = np.asarray([0.0, 1.0])
    for seed in COMMON_SEEDS:
        ax.plot(
            x,
            [baseline[seed], ddqn[seed]],
            marker="o",
            linewidth=1.8,
            label=f"Seed {seed} (Δ {ddqn[seed] - baseline[seed]:+.2f})",
        )

    baseline_mean = np.mean([baseline[s] for s in COMMON_SEEDS])
    ddqn_mean = np.mean([ddqn[s] for s in COMMON_SEEDS])
    ax.plot(
        x,
        [baseline_mean, ddqn_mean],
        marker="D",
        linewidth=3.0,
        linestyle="--",
        label=f"Media (Δ {ddqn_mean - baseline_mean:+.2f})",
    )

    ax.set_xticks(x, ["Vanilla DQN", "Double DQN"])
    ax.set_ylabel("Return Atari greedy a 1.8M step")
    ax.set_title("Confronto paired DQN vs Double DQN")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()

    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    out = REPORT_ROOT / "03_dqn_ddqn_paired_final_comparison.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return out


def main():
    generated = [
        plot_learning_curve(
            "baseline",
            "Vanilla DQN — evaluation multi-seed",
            "01_baseline_multiseed_evaluation.png",
        )
    ]
    generated.append(
        plot_learning_curve(
            "ddqn",
            "Double DQN — evaluation multi-seed",
            "02_ddqn_multiseed_evaluation.png",
        )
    )
    generated.append(plot_paired_final())
    for path in generated:
        print(path)


if __name__ == "__main__":
    main()
