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


def train(config, device, output_dir, label, step_bonus, max_rally_bonus, seed):

    make_env = partial(build_env, config)
    evaluate = partial(evaluate_policy, device=device, seed_random_actions=True)
    ReplayBuffer = partial(ReplayBufferBase, device=device)
    "Esegue una run completa e restituisce checkpoint e diagnostiche."
    set_all_seeds(seed)
    train_env = make_env(step_bonus, max_rally_bonus)
    eval_env = make_env()
    train_env.action_space.seed(seed)
    eval_env.action_space.seed(seed + 1)
    n_actions = train_env.action_space.n
    policy_net = DQN(n_actions).to(device)
    target_net = DQN(n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    optimizer = optim.Adam(policy_net.parameters(), lr=config["parameters"]["LR"])
    loss_fn = nn.SmoothL1Loss()
    replay_buffer = ReplayBuffer(capacity=config["parameters"]["BUFFER_CAPACITY"])
    state, _ = train_env.reset(seed=seed)
    raw_ep_reward = 0.0
    shaped_ep_reward = 0.0
    ep_bonus = 0.0
    current_rally_lengths = []
    history = {
        "step": [],
        "raw_return": [],
        "shaped_return": [],
        "bonus": [],
        "mean_rally": [],
        "loss": [],
    }
    checkpoint_results = {}
    policy_net.train()
    start_time = time.time()
    print(
        f"\n=== {label}: bonus={step_bonus:g}, cap={max_rally_bonus:.2f} ===",
        flush=True,
    )
    for step in range(1, config["parameters"]["TOTAL_STEPS"] + 1):
        epsilon = max(
            config["parameters"]["EPS_END"],
            config["parameters"]["EPS_START"]
            - (config["parameters"]["EPS_START"] - config["parameters"]["EPS_END"])
            * (step / config["parameters"]["DECAY_STEPS"]),
        )
        action = select_action(state, policy_net, train_env, epsilon, device)
        next_state, shaped_reward, term, trunc, info = train_env.step(action)
        raw_reward = info["raw_reward"]
        shaping_bonus = info["shaping_bonus"]
        completed_rally_steps = info["completed_rally_steps"]
        replay_buffer.push(state, action, shaped_reward, next_state, term)
        state = next_state
        raw_ep_reward += raw_reward
        shaped_ep_reward += shaped_reward
        ep_bonus += shaping_bonus
        if completed_rally_steps is not None:
            current_rally_lengths.append(completed_rally_steps)
        if term or trunc:
            history["step"].append(step)
            history["raw_return"].append(raw_ep_reward)
            history["shaped_return"].append(shaped_ep_reward)
            history["bonus"].append(ep_bonus)
            history["mean_rally"].append(
                float(np.mean(current_rally_lengths)) if current_rally_lengths else 0.0
            )
            if len(history["raw_return"]) % 20 == 0:
                elapsed = (time.time() - start_time) / 60.0
                print(
                    f"{label} | step {step:7d}/{config['parameters']['TOTAL_STEPS']} | ep {len(history['raw_return']):4d} | raw20 {np.mean(history['raw_return'][-20:]):6.2f} | shaped20 {np.mean(history['shaped_return'][-20:]):6.2f} | bonus20 {np.mean(history['bonus'][-20:]):6.2f} | rally20 {np.mean(history['mean_rally'][-20:]):6.1f} | eps {epsilon:.3f} | {elapsed:.1f}m",
                    flush=True,
                )
            state, _ = train_env.reset()
            raw_ep_reward = 0.0
            shaped_ep_reward = 0.0
            ep_bonus = 0.0
            current_rally_lengths = []
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
            history["loss"].append(loss_value)
        if step % config["parameters"]["TARGET_UPDATE"] == 0:
            target_net.load_state_dict(policy_net.state_dict())
        if step % 100000 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        if step in config["parameters"]["CHECKPOINTS"]:
            policy_net.eval()
            mean_score, std_score = evaluate(
                eval_env,
                model=policy_net,
                episodes=config["parameters"]["EVAL_EPISODES"],
            )
            policy_net.train()
            checkpoint_results[step] = (mean_score, std_score)
            print(
                f"--- {label} @ {step}: Atari raw {mean_score:.2f} +/- {std_score:.2f} ---",
                flush=True,
            )
            safe_label = re.sub("[^a-z0-9]+", "_", label.lower()).strip("_")
            save_checkpoint(
                {
                    "step": step,
                    "label": label,
                    "policy_state_dict": policy_net.state_dict(),
                    "target_state_dict": target_net.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "rally_step_bonus": step_bonus,
                    "max_rally_bonus": max_rally_bonus,
                },
                f"dqn_pong_{safe_label}_{step}.pth",
                step,
                config,
                output_dir,
            )
    train_env.close()
    eval_env.close()
    result = {
        "label": label,
        "step_bonus": step_bonus,
        "max_rally_bonus": max_rally_bonus,
        "checkpoints": checkpoint_results,
        "history": history,
    }
    del replay_buffer, policy_net, target_net, optimizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result
