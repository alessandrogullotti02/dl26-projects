import random
import numpy as np
import torch
import torch.nn as nn


def set_all_seeds(seed: int, cuda=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda and torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


reset_seeds = set_all_seeds


def configure_determinism(enabled=True):
    """Impostazioni usate dalle nuove baseline/DDQN multi-seed."""
    torch.backends.cudnn.benchmark = not enabled
    torch.backends.cudnn.deterministic = bool(enabled)
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(bool(enabled))


def select_action(state, model, env, epsilon: float, device):
    if random.random() < epsilon:
        return env.action_space.sample()
    tensor = (
        torch.as_tensor(np.asarray(state), dtype=torch.uint8, device=device)
        .float()
        .unsqueeze(0)
    )
    with torch.inference_mode():
        return model(tensor).argmax(dim=1).item()


def optimize(
    policy_net,
    target_net,
    optimizer,
    loss_fn,
    buffer,
    batch_size: int,
    gamma: float,
    double=False,
):
    states, actions, rewards, next_states, terminated = buffer.sample(batch_size)
    with torch.no_grad():
        if double:
            best_actions = policy_net(next_states).argmax(dim=1, keepdim=True)
            next_q = target_net(next_states).gather(1, best_actions).squeeze(1)
            dqn_next_q = target_net(next_states).max(dim=1).values
            gap = (gamma * (1.0 - terminated) * (dqn_next_q - next_q)).mean().item()
        else:
            next_q = target_net(next_states).max(dim=1).values
            gap = None
        target_q = rewards + (1.0 - terminated) * gamma * next_q

    current_q = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    loss = loss_fn(current_q, target_q)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
    optimizer.step()
    return float(loss.item()), gap
