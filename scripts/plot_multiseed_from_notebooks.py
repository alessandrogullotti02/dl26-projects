"""Rebuild the four multi-seed ablation figures directly from executed notebooks.

The script intentionally aggregates the *mean greedy Atari return* of each trained
network.  The uncertainty band is the sample standard deviation across independent
training seeds (ddof=1); it is not the average of per-episode standard deviations.

Run from anywhere with:
    python scripts/plot_multiseed_from_notebooks.py

Outputs:
    figures/report/{10,11,12,13}_*_multiseed_evaluation.png
    docs/results/summary_final_scores.csv
"""

from __future__ import annotations

from pathlib import Path
import csv
import json
import re

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = PROJECT_ROOT / "notebooks"
FIGURES = PROJECT_ROOT / "figures" / "report"
RESULTS = PROJECT_ROOT / "docs" / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

FILES = {
    "epsilon": {
        42: NOTEBOOKS / "epsilon_analysis.ipynb",
        2: NOTEBOOKS / "replicates" / "epsilon_seed2.ipynb",
        3: NOTEBOOKS / "replicates" / "epsilon_seed3.ipynb",
    },
    "rally": {
        42: NOTEBOOKS / "rally_analysis.ipynb",
        2: NOTEBOOKS / "replicates" / "rally_seed2.ipynb",
        3: NOTEBOOKS / "replicates" / "rally_seed3.ipynb",
    },
    "potential": {
        42: NOTEBOOKS / "potential_analysis.ipynb",
        2: NOTEBOOKS / "replicates" / "potential_seed2.ipynb",
        3: NOTEBOOKS / "replicates" / "potential_seed3.ipynb",
    },
    "contact": {
        42: NOTEBOOKS / "contact_analysis.ipynb",
        2: NOTEBOOKS / "replicates" / "contact_seed2.ipynb",
        3: NOTEBOOKS / "replicates" / "contact_seed3.ipynb",
    },
}

LABELS = {
    "epsilon": ["Lineare", "Esponenziale", "Costante 0.10"],
    "rally": ["Piccolo", "Moderato", "Aggressivo"],
    "potential": [r"$\kappa=0.02$", r"$\kappa=0.10$", r"$\kappa=0.50$"],
    "contact": [r"$c=0.02$", r"$c=0.10$", r"$c=0.50$"],
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


def notebook_text(path: Path) -> str:
    nb = json.loads(path.read_text(encoding="utf-8"))
    parts: list[str] = []
    for cell in nb.get("cells", []):
        for output in cell.get("outputs", []):
            if "text" in output:
                value = output["text"]
                parts.append("".join(value) if isinstance(value, list) else value)
            data = output.get("data", {})
            if "text/plain" in data:
                value = data["text/plain"]
                parts.append("".join(value) if isinstance(value, list) else value)
    return "\n".join(parts)


def extract_seed(path: Path) -> int | None:
    nb = json.loads(path.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in nb.get("cells", [])
    )
    match = re.search(r"(?m)^\s*SEED\s*=\s*(\d+)", source)
    return int(match.group(1)) if match else None


def parse_epsilon(path: Path) -> dict[str, dict[int, float]]:
    text = notebook_text(path)
    aliases = {
        "Linear decay": "Lineare",
        "Exponential decay": "Esponenziale",
        "Constant epsilon=0.1": "Costante 0.10",
    }
    result = {label: {} for label in aliases.values()}
    pattern = re.compile(
        r"---\s*(.*?)\s*\|\s*Checkpoint\s*(\d+):\s*"
        r"([+-]?\d+(?:\.\d+)?)\s*\+/-\s*"
        r"([+-]?\d+(?:\.\d+)?)"
    )
    for name, step, mean, _episode_std in pattern.findall(text):
        if name in aliases:
            # First occurrence is the current executed run if stale legacy cells exist.
            result[aliases[name]].setdefault(int(step), float(mean))

    for label, values in result.items():
        if len(values) != 6:
            raise ValueError(
                f"{path.name}: {label}: attesi 6 checkpoint, trovati {len(values)}"
            )
    return result


def parse_three_variant_table(path: Path) -> dict[int, list[float]]:
    """Parse the executed compact checkpoint comparison for shaping notebooks."""
    text = notebook_text(path)
    rows: dict[int, list[float]] = {}
    pattern = re.compile(
        r"(?m)^\s*(600|900|1200|1500|1800)k\s*\|\s*"
        r"([+-]?\d+(?:\.\d+)?)\s*\+/-\s*[+-]?\d+(?:\.\d+)?\s*\|\s*"
        r"([+-]?\d+(?:\.\d+)?)\s*\+/-\s*[+-]?\d+(?:\.\d+)?\s*\|\s*"
        r"([+-]?\d+(?:\.\d+)?)\s*\+/-\s*[+-]?\d+(?:\.\d+)?"
    )
    for step_k, first, second, third in pattern.findall(text):
        # Keep the first matching current table and ignore later stale/archival output.
        rows.setdefault(
            int(step_k) * 1000,
            [float(first), float(second), float(third)],
        )

    if len(rows) != 5:
        raise ValueError(
            f"{path.name}: attese 5 righe checkpoint, trovate {len(rows)}"
        )
    return rows


def collect() -> dict:
    data: dict = {}

    for family, seed_files in FILES.items():
        for expected_seed, path in seed_files.items():
            if not path.exists():
                raise FileNotFoundError(path)
            actual_seed = extract_seed(path)
            if actual_seed != expected_seed:
                raise ValueError(
                    f"{family}: {path.name} dovrebbe avere seed {expected_seed}, "
                    f"ma contiene SEED={actual_seed}"
                )

    data["epsilon"] = {
        seed: parse_epsilon(path) for seed, path in FILES["epsilon"].items()
    }

    for family in ("rally", "potential", "contact"):
        data[family] = {}
        labels = LABELS[family]
        for seed, path in FILES[family].items():
            rows = parse_three_variant_table(path)
            data[family][seed] = {
                labels[index]: {
                    step: values[index] for step, values in rows.items()
                }
                for index in range(3)
            }

    return data


def aggregate_family(data: dict, family: str):
    seeds = list(FILES[family])
    labels = LABELS[family]
    steps = sorted(data[family][seeds[0]][labels[0]])

    aggregate = {}
    for label in labels:
        values = np.asarray(
            [
                [data[family][seed][label][step] for step in steps]
                for seed in seeds
            ],
            dtype=float,
        )
        aggregate[label] = {
            "values": values,
            "mean": values.mean(axis=0),
            "std": values.std(axis=0, ddof=1),
        }
    return seeds, steps, aggregate


def main() -> None:
    data = collect()
    summary_rows: list[dict] = []

    for family in ("epsilon", "rally", "potential", "contact"):
        seeds, steps, aggregate = aggregate_family(data, family)
        x = np.asarray(steps) / 1e6
        FIGURES.mkdir(parents=True, exist_ok=True)

        # Learning curve: mean and sample standard deviation across training seeds.
        fig, ax = plt.subplots(figsize=(8.6, 5.2))
        for label in LABELS[family]:
            mean = aggregate[label]["mean"]
            std = aggregate[label]["std"]
            line, = ax.plot(x, mean, marker="o", linewidth=2, label=label)
            ax.fill_between(x, mean - std, mean + std, alpha=0.15)
            for seed_values in aggregate[label]["values"]:
                ax.plot(
                    x,
                    seed_values,
                    linewidth=0.8,
                    alpha=0.22,
                    color=line.get_color(),
                )

        ax.set_xlabel("Environment steps (milioni)")
        ax.set_ylabel("Return Atari greedy")
        ax.set_title(TITLES[family] + " — media ± std tra 3 training seed")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIGURES / REPORT_FILENAMES[family], dpi=220)
        plt.close(fig)

        for label in LABELS[family]:
            row = {
                "family": family,
                "variant": label,
                "final_mean": float(aggregate[label]["mean"][-1]),
                "final_std_across_training_seeds": float(aggregate[label]["std"][-1]),
            }
            for index, seed in enumerate(seeds):
                row[f"seed_{seed}"] = float(aggregate[label]["values"][index, -1])
            summary_rows.append(row)

    csv_path = RESULTS / "summary_final_scores.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "family",
                "variant",
                "final_mean",
                "final_std_across_training_seeds",
                "seed_42",
                "seed_2",
                "seed_3",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Figure salvate in {FIGURES}")
    print(f"Tabella finale salvata in {csv_path}")


if __name__ == "__main__":
    main()
