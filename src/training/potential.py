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


def train(config, device, output_dir, label, potential_scale, seed):

    make_env = partial(build_env, config)
    evaluate = partial(evaluate_policy, device=device, seed_random_actions=True)
    ReplayBuffer = partial(ReplayBufferBase, device=device)
    set_all_seeds(seed)
    env = make_env(use_potential=True, potential_scale=potential_scale)
    eval_env = make_env(use_potential=False)
    env.action_space.seed(seed)
    eval_env.action_space.seed(seed + 1)
    n_actions = env.action_space.n
    policy_net = DQN(n_actions).to(device)
    target_net = DQN(n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    optimizer = optim.Adam(policy_net.parameters(), lr=config["parameters"]["LR"])
    loss_fn = nn.SmoothL1Loss()
    buffer = ReplayBuffer(capacity=config["parameters"]["BUFFER_CAPACITY"])
    state, reset_info = env.reset(seed=seed)
    initial_potential = float(reset_info["potential"])
    raw_ep_return = 0.0
    shaped_ep_return = 0.0
    shaping_ep_sum = 0.0
    discounted_shaping_sum = 0.0
    discount_power = 1.0
    potential_sum = initial_potential
    abs_shaping_sum = 0.0
    episode_steps = 0
    history = {
        "step": [],
        "raw_return": [],
        "shaped_return": [],
        "shaping_sum": [],
        "mean_potential": [],
        "mean_abs_shaping": [],
        "telescoping_error": [],
        "loss": [],
    }
    checkpoint_results = {}
    policy_net.train()
    start_time = time.time()
    for step in range(1, config["parameters"]["TOTAL_STEPS"] + 1):
        epsilon = max(
            config["parameters"]["EPS_END"],
            config["parameters"]["EPS_START"]
            - (config["parameters"]["EPS_START"] - config["parameters"]["EPS_END"])
            * (step / config["parameters"]["DECAY_STEPS"]),
        )
        action = select_action(state, policy_net, env, epsilon, device)
        next_state, shaped_reward, term, trunc, info = env.step(action)
        raw_reward = info["raw_reward"]
        shaping_reward = info["shaping_reward"]
        next_potential = info["potential_next"]
        buffer.push(state, action, shaped_reward, next_state, term)
        state = next_state
        raw_ep_return += raw_reward
        shaped_ep_return += shaped_reward
        shaping_ep_sum += shaping_reward
        discounted_shaping_sum += discount_power * shaping_reward
        discount_power *= config["parameters"]["GAMMA"]
        potential_sum += next_potential
        abs_shaping_sum += abs(shaping_reward)
        episode_steps += 1
        if term or trunc:
            telescoping_target = -initial_potential + discount_power * next_potential
            telescoping_error = discounted_shaping_sum - telescoping_target
            history["step"].append(step)
            history["raw_return"].append(raw_ep_return)
            history["shaped_return"].append(shaped_ep_return)
            history["shaping_sum"].append(shaping_ep_sum)
            history["mean_potential"].append(potential_sum / max(1, episode_steps + 1))
            history["mean_abs_shaping"].append(abs_shaping_sum / max(1, episode_steps))
            history["telescoping_error"].append(telescoping_error)
            if len(history["raw_return"]) % 20 == 0:
                elapsed = (time.time() - start_time) / 60.0
                print(
                    f"{label} | Step {step:7d}/{config['parameters']['TOTAL_STEPS']} | Episodio {len(history['raw_return']):4d} | Raw20 {np.mean(history['raw_return'][-20:]):6.2f} | Shaped20 {np.mean(history['shaped_return'][-20:]):6.2f} | |F| medio20 {np.mean(history['mean_abs_shaping'][-20:]):.4f} | eps {epsilon:.3f} | {elapsed:.1f}m",
                    flush=True,
                )
            state, reset_info = env.reset()
            initial_potential = float(reset_info["potential"])
            raw_ep_return = 0.0
            shaped_ep_return = 0.0
            shaping_ep_sum = 0.0
            discounted_shaping_sum = 0.0
            discount_power = 1.0
            potential_sum = initial_potential
            abs_shaping_sum = 0.0
            episode_steps = 0
        if (
            step > config["parameters"]["LEARNING_STARTS"]
            and step % config["parameters"]["TRAIN_FREQ"] == 0
        ):
            loss_value, gap = optimize(
                policy_net,
                target_net,
                optimizer,
                loss_fn,
                buffer,
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
                f"\n--- Checkpoint {step} passi ({label}, scala {potential_scale:g}, reward Atari): {mean_score:.2f} +/- {std_score:.2f} ---\n",
                flush=True,
            )
            save_checkpoint(
                {
                    "step": step,
                    "policy_state_dict": policy_net.state_dict(),
                    "target_state_dict": target_net.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "gamma": config["parameters"]["GAMMA"],
                    "potential_scale": potential_scale,
                    "potential_kwargs": dict(config["parameters"]["POTENTIAL_KWARGS"], scale=potential_scale),
                },
                f"dqn_pong_potential_{potential_scale:g}_{step}.pth",
                step,
                config,
                output_dir,
            )
    env.close()
    eval_env.close()
    return {
        "label": label,
        "potential_scale": float(potential_scale),
        "history": history,
        "checkpoints": checkpoint_results,
    }
