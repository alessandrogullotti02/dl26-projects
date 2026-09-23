import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import importlib
from pathlib import Path
import time
import numpy as np
from src.utils.results import save_json
from src.utils.configuration import load_config, resolve_config


def validate(config):
    resolve_config(config)
    family = config["experiment"]
    if family not in {"baseline", "ddqn", "epsilon", "rally", "potential", "contact"}:
        raise ValueError(f"Esperimento sconosciuto: {family}")
    p = config["parameters"]
    if p["TOTAL_STEPS"] not in p["CHECKPOINTS"]:
        raise ValueError("TOTAL_STEPS deve essere un checkpoint")
    if p["BATCH_SIZE"] > p["BUFFER_CAPACITY"] or p["BATCH_SIZE"] > p["LEARNING_STARTS"]:
        raise ValueError("Batch incompatibile con buffer/warm-up")
    if config["checkpoint_policy"] not in {"original", "final", "none"}:
        raise ValueError("checkpoint_policy non valida")


def _aggregate_seeded_runs(runs, checkpoints):
    ordered = sorted(runs, key=lambda x: int(x.split("_")[-1]))
    matrix = np.asarray([
        [runs[name]["checkpoints"][step]["mean"] for step in checkpoints]
        for name in ordered
    ], dtype=np.float64)
    std = matrix.std(axis=0, ddof=1).tolist() if len(ordered) > 1 else [None] * len(checkpoints)
    return {
        "training_seeds": [int(name.split("_")[-1]) for name in ordered],
        "checkpoint_steps": list(checkpoints),
        "mean_across_training_seeds": matrix.mean(axis=0).tolist(),
        "std_across_training_seeds": std,
        "individual_means": matrix.tolist(),
    }


def _checkpoint_mean(run, step):
    value = run["checkpoints"][step]
    if isinstance(value, dict):
        return float(value["mean"])
    return float(value[0])


def _aggregate_variants(runs, checkpoints, variants, seeds):
    out = {}
    for variant in variants:
        matrix = np.asarray([
            [_checkpoint_mean(runs[f"{variant}_seed_{seed}"], step) for step in checkpoints]
            for seed in seeds
        ], dtype=np.float64)
        std = matrix.std(axis=0, ddof=1).tolist() if len(seeds) > 1 else [None] * len(checkpoints)
        out[variant] = {
            "training_seeds": list(seeds),
            "checkpoint_steps": list(checkpoints),
            "mean_across_training_seeds": matrix.mean(axis=0).tolist(),
            "std_across_training_seeds": std,
            "individual_means": matrix.tolist(),
        }
    return out


def run(config: dict, device, seed_filter=None):
    from src.datasets.environment import make_env
    from src.evaluation.policy import evaluate
    from src.training.common import set_all_seeds

    validate(config)
    output = Path(config["output_dir"])
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"{output} non è vuota; usa --output-dir con una nuova cartella.")
    output.mkdir(parents=True, exist_ok=True)
    save_json(output / "config.json", config)

    family, p = config["experiment"], config["parameters"]
    configured_seeds = list(p.get("SEEDS", [p.get("SEED")]))
    configured_seeds = [int(x) for x in configured_seeds if x is not None]
    if seed_filter is not None:
        if int(seed_filter) not in configured_seeds:
            raise ValueError(f"Seed {seed_filter} non presente nella configurazione: {configured_seeds}")
        active_seeds = [int(seed_filter)]
    else:
        active_seeds = configured_seeds
    train = importlib.import_module(f"src.training.{family}").train
    result = {
        "experiment": family,
        "config": config,
        "runs": {},
        "random": None,
        "aggregate": None,
        "complete": False,
    }
    start = time.time()

    def random_baseline():
        base_seed = p.get("EVAL_BASE_SEED", 1000)
        env = make_env(config, action_seed=base_seed)
        try:
            return list(evaluate(
                env,
                episodes=20 if family == "epsilon" else p.get("EVAL_EPISODES", p.get("N_EVAL_EPISODES", 20)),
                base_seed=base_seed,
                device=device,
                seed_random_actions=True,
            ))
        finally:
            env.close()

    if family in {"baseline", "ddqn"}:
        result["random"] = random_baseline()
        variants = [(f"seed_{seed}", {"seed": seed}) for seed in active_seeds]
    elif family == "epsilon":
        result["random"] = random_baseline()
        variants = [
            (f"{strategy}_seed_{seed}", {"strategy": strategy, "seed": seed})
            for seed in active_seeds
            for strategy in p["STRATEGIES"]
        ]
    elif family == "rally":
        variants = []
        for seed in active_seeds:
            variants.extend([
                (f"small_seed_{seed}", dict(label="Reward piccolo", step_bonus=p["SMALL_STEP_BONUS"], max_rally_bonus=p["SMALL_MAX_RALLY_BONUS"], seed=seed)),
                (f"moderate_seed_{seed}", dict(label="Reward moderato", step_bonus=p["MODERATE_STEP_BONUS"], max_rally_bonus=p["MODERATE_MAX_RALLY_BONUS"], seed=seed)),
                (f"aggressive_seed_{seed}", dict(label=f"Reward aggressivo (cap {p['AGGRESSIVE_MAX_RALLY_BONUS']:.2f})", step_bonus=p["AGGRESSIVE_STEP_BONUS"], max_rally_bonus=p["AGGRESSIVE_MAX_RALLY_BONUS"], seed=seed)),
            ])
    elif family == "potential":
        variants = []
        for seed in active_seeds:
            variants.extend([
                (f"small_seed_{seed}", dict(label="PBRS scala piccola", potential_scale=p["SMALL_POTENTIAL_SCALE"], seed=seed)),
                (f"moderate_seed_{seed}", dict(label="PBRS scala moderata", potential_scale=p["MODERATE_POTENTIAL_SCALE"], seed=seed)),
                (f"aggressive_seed_{seed}", dict(label="PBRS scala aggressiva", potential_scale=p["AGGRESSIVE_POTENTIAL_SCALE"], seed=seed)),
            ])
    elif family == "contact":
        variants = []
        for seed in active_seeds:
            variants.extend([
                (f"small_seed_{seed}", dict(label="Contatto piccolo", contact_bonus=p["SMALL_CONTACT_BONUS"], seed=seed)),
                (f"moderate_seed_{seed}", dict(label="Contatto moderato", contact_bonus=p["MODERATE_CONTACT_BONUS"], seed=seed)),
                (f"aggressive_seed_{seed}", dict(label="Contatto aggressivo", contact_bonus=p["AGGRESSIVE_CONTACT_BONUS"], seed=seed)),
            ])
    else:
        raise AssertionError(family)

    for name, kwargs in variants:
        result["runs"][name] = train(config, device, output, **kwargs)
        save_json(output / "results.json", result)

    if family in {"rally", "potential", "contact"}:
        result["random"] = random_baseline()
    if family in {"baseline", "ddqn"}:
        result["aggregate"] = _aggregate_seeded_runs(result["runs"], p["CHECKPOINTS"])
    elif family == "epsilon":
        result["aggregate"] = _aggregate_variants(result["runs"], p["CHECKPOINTS"], p["STRATEGIES"], active_seeds)
    elif family in {"rally", "potential", "contact"}:
        result["aggregate"] = _aggregate_variants(result["runs"], p["CHECKPOINTS"], ["small", "moderate", "aggressive"], active_seeds)

    result["elapsed_minutes"] = (time.time() - start) / 60.0
    result["complete"] = True
    save_json(output / "results.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--checkpoint-policy", choices=["original", "final", "none"])
    parser.add_argument("--device", default="auto", help="auto, cpu oppure cuda")
    parser.add_argument("--seed", type=int, help="esegue soltanto il training seed indicato")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.output_dir:
        config["output_dir"] = args.output_dir
    if args.checkpoint_policy:
        config["checkpoint_policy"] = args.checkpoint_policy
    validate(config)

    if args.dry_run:
        import json
        print(json.dumps(config, indent=2, ensure_ascii=False))
        return

    import torch
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto" else torch.device(args.device)
    )
    print("Dispositivo:", device, flush=True)
    run(config, device, seed_filter=args.seed)


if __name__ == "__main__":
    main()
