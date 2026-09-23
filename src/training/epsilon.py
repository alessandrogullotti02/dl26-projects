import gc
import time
import re
from functools import partial
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from src.models.dqn import DQN
from src.datasets.replay_buffer import ReplayBuffer as ReplayBufferBase
from src.datasets.environment import make_env as build_env
from src.evaluation.policy import evaluate as evaluate_policy
from src.training.common import set_all_seeds, reset_seeds, select_action, optimize
from src.utils.results import save_checkpoint
from src.training.schedules import epsilon_value


def train(config, device, output_dir, strategy, seed):

    make_env = partial(build_env, config)
    evaluate = partial(evaluate_policy, device=device, seed_random_actions=True)
    ReplayBuffer = partial(ReplayBufferBase, device=device)
    reset_seeds(seed)
    env = make_env()
    eval_env = make_env()
    env.action_space.seed(seed)
    eval_env.action_space.seed(seed + 1)
    n_actions = env.action_space.n
    policy_net = DQN(n_actions).to(device)
    target_net = DQN(n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    optimizer = optim.Adam(policy_net.parameters(), lr=config["parameters"]["LR"])
    loss_fn = nn.SmoothL1Loss()
    replay_buffer = ReplayBuffer(capacity=config["parameters"]["BUFFER_CAPACITY"])
    state, _ = env.reset(seed=seed)
    episode_reward = 0.0
    rewards_history = []
    step_history = []
    loss_history = []
    checkpoint_results = {}
    policy_net.train()
    start_time = time.time()
    label = config["parameters"]["STRATEGY_LABELS"][strategy]
    print(f"\n===== INIZIO: {label} =====", flush=True)
    for step in range(1, config["parameters"]["TOTAL_STEPS"] + 1):
        epsilon = epsilon_value(strategy, step, config)
        action = select_action(state, policy_net, env, epsilon, device)
        next_state, reward, term, trunc, _ = env.step(action)
        replay_buffer.push(state, action, reward, next_state, term)
        state = next_state
        episode_reward += reward
        if term or trunc:
            rewards_history.append(episode_reward)
            step_history.append(step)
            if len(rewards_history) % 20 == 0:
                avg_recent = np.mean(rewards_history[-20:])
                elapsed = (time.time() - start_time) / 60.0
                print(
                    f"{label:24s} | Step {step:7d}/{config['parameters']['TOTAL_STEPS']} | Episodio {len(rewards_history):4d} | Media ultimi 20: {avg_recent:6.2f} | eps: {epsilon:.3f} | Tempo: {elapsed:.1f}m",
                    flush=True,
                )
            state, _ = env.reset()
            episode_reward = 0.0
        if (
            step > config["parameters"]["LEARNING_STARTS"]
            and step % config["parameters"]["TRAIN_FREQ"] == 0
        ):
            loss_value, gap = optimize(
                policy_net,
                target_net,
                optimizer,
                loss_fn,
                replay_buffer,
                config["parameters"]["BATCH_SIZE"],
                config["parameters"]["GAMMA"],
                double=False,
            )
            loss_history.append(loss_value)
        if step % config["parameters"]["TARGET_UPDATE"] == 0:
            target_net.load_state_dict(policy_net.state_dict())
        if step % 100000 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        if step in config["parameters"]["CHECKPOINTS"]:
            episodes = (
                config["parameters"]["FINAL_EVAL_EPISODES"]
                if step == config["parameters"]["TOTAL_STEPS"]
                else config["parameters"]["EVAL_EPISODES"]
            )
            policy_net.eval()
            mean_score, std_score = evaluate(
                eval_env, model=policy_net, episodes=episodes, base_seed=1000
            )
            policy_net.train()
            checkpoint_results[step] = {
                "mean": mean_score,
                "std": std_score,
                "episodes": episodes,
            }
            print(
                f"\n--- {label} | Checkpoint {step}: {mean_score:.2f} +/- {std_score:.2f} ({episodes} episodi) ---\n",
                flush=True,
            )
    elapsed_minutes = (time.time() - start_time) / 60.0
    save_checkpoint(
        {
            "strategy": strategy,
            "step": config["parameters"]["TOTAL_STEPS"],
            "policy_state_dict": policy_net.state_dict(),
            "target_state_dict": target_net.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epsilon_final": epsilon_value(
                strategy, config["parameters"]["TOTAL_STEPS"], config
            ),
        },
        f"dqn_pong_epsilon_{strategy}_seed{seed}_{config['parameters']['TOTAL_STEPS']}.pth",
        step,
        config,
        output_dir,
    )
    result = {
        "strategy": strategy,
        "seed": int(seed),
        "label": label,
        "steps": step_history,
        "rewards": rewards_history,
        "losses": loss_history,
        "checkpoints": checkpoint_results,
        "elapsed_minutes": elapsed_minutes,
    }
    env.close()
    eval_env.close()
    replay_buffer.clear()
    del replay_buffer, policy_net, target_net, optimizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"===== FINE: {label} | {elapsed_minutes:.1f} minuti =====", flush=True)
    return result
