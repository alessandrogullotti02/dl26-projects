import gc
import time
from functools import partial
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from src.models.dqn import DQN
from src.datasets.replay_buffer import ReplayBuffer as ReplayBufferBase
from src.datasets.environment import make_env as build_env
from src.evaluation.policy import evaluate as evaluate_policy
from src.training.common import set_all_seeds, configure_determinism, select_action, optimize
from src.utils.results import save_checkpoint


def train(config, device, output_dir, seed):
    """Una replica Double DQN; il protocollo primario usa i seed 42,2,3 separatamente."""
    p = config["parameters"]
    configure_determinism(True)
    set_all_seeds(seed)

    make_env = partial(build_env, config)
    ReplayBuffer = partial(ReplayBufferBase, device=device)
    env = make_env(action_seed=seed)
    eval_env = make_env(action_seed=p["EVAL_BASE_SEED"])
    evaluate = partial(
        evaluate_policy,
        device=device,
        seed_random_actions=True,
        base_seed=p["EVAL_BASE_SEED"],
    )

    n_actions = env.action_space.n
    policy_net = DQN(n_actions).to(device)
    target_net = DQN(n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    optimizer = optim.Adam(policy_net.parameters(), lr=p["LR"])
    loss_fn = nn.SmoothL1Loss()
    buffer = ReplayBuffer(capacity=p["BUFFER_CAPACITY"])

    state, _ = env.reset(seed=seed)
    ep_reward = 0.0
    rewards_history, step_history, loss_history, target_gap_history = [], [], [], []
    checkpoint_results = {}
    policy_net.train()
    start_time = time.time()

    for step in range(1, p["TOTAL_STEPS"] + 1):
        eps = max(
            p["EPS_END"],
            p["EPS_START"]
            - (p["EPS_START"] - p["EPS_END"]) * (step / p["DECAY_STEPS"]),
        )
        action = select_action(state, policy_net, env, eps, device)
        next_state, reward, term, trunc, _ = env.step(action)
        buffer.push(state, action, reward, next_state, term)
        state = next_state
        ep_reward += float(reward)

        if term or trunc:
            rewards_history.append(ep_reward)
            step_history.append(step)
            if len(rewards_history) % 20 == 0:
                elapsed = (time.time() - start_time) / 60.0
                print(
                    f"[seed {seed}] Step {step:7d}/{p['TOTAL_STEPS']} | "
                    f"Episodio {len(rewards_history):4d} | "
                    f"Media ultimi 20: {np.mean(rewards_history[-20:]):6.2f} | "
                    f"eps: {eps:.3f} | {elapsed:.1f} min",
                    flush=True,
                )
            state, _ = env.reset()
            ep_reward = 0.0

        if step > p["LEARNING_STARTS"] and step % p["TRAIN_FREQ"] == 0:
            loss_value, gap = optimize(
                policy_net, target_net, optimizer, loss_fn, buffer,
                p["BATCH_SIZE"], p["GAMMA"], double=True,
            )
            loss_history.append(loss_value)
            target_gap_history.append(float(gap))

        if step % p["TARGET_UPDATE"] == 0:
            target_net.load_state_dict(policy_net.state_dict())

        if step % 100000 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if step in p["CHECKPOINTS"]:
            policy_net.eval()
            mean, episode_std, scores = evaluate(
                eval_env,
                model=policy_net,
                episodes=p["N_EVAL_EPISODES"],
                return_scores=True,
            )
            policy_net.train()
            checkpoint_results[step] = {
                "mean": mean,
                "episode_std": episode_std,
                "scores": scores.tolist(),
            }
            save_checkpoint(
                policy_net.state_dict(),
                f"double_dqn_pong_seed{seed}_{step}.pth",
                step, config, output_dir,
            )
            print(
                f"\n--- seed {seed} | checkpoint {step}: "
                f"{mean:.2f} +/- {episode_std:.2f} ---\n",
                flush=True,
            )

    env.close(); eval_env.close()
    result = {
        "seed": int(seed),
        "steps": step_history,
        "rewards": rewards_history,
        "losses": loss_history,
        "target_gap": target_gap_history,
        "checkpoints": checkpoint_results,
        "n_episodes": len(rewards_history),
        "n_updates": len(loss_history),
    }
    del buffer, optimizer, target_net, policy_net
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result
