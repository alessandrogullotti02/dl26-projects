import argparse
from pathlib import Path
from src.utils.results import save_json, load_checkpoint
from src.utils.configuration import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--base-seed", type=int, default=1000)
    parser.add_argument("--output")
    args = parser.parse_args()
    import torch
    from src.models.dqn import DQN
    from src.datasets.environment import make_env
    from src.evaluation.policy import evaluate

    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = make_env(config)
    try:
        model = DQN(env.action_space.n).to(device)
        payload = load_checkpoint(Path(args.weights), device)
        model.load_state_dict(payload.get("policy_state_dict", payload))
        model.eval()
        mean, std = evaluate(env, model, args.episodes, args.base_seed, device=device)
        metrics = {
            "mean": mean,
            "std": std,
            "episodes": args.episodes,
            "base_seed": args.base_seed,
        }
        print(metrics)
        if args.output:
            save_json(args.output, metrics)
    finally:
        env.close()


if __name__ == "__main__":
    main()
