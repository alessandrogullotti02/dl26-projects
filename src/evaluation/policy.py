import numpy as np
import torch


def evaluate(
    eval_env,
    model=None,
    episodes=20,
    base_seed=1000,
    *,
    device="cpu",
    seed_random_actions=True,
    return_scores=False,
):
    """Greedy evaluation con gli stessi seed episodici per tutti i modelli."""
    scores = []
    for episode in range(episodes):
        episode_seed = base_seed + episode
        state, _ = eval_env.reset(seed=episode_seed)
        if model is None and seed_random_actions:
            eval_env.action_space.seed(episode_seed)

        done = False
        total_reward = 0.0
        while not done:
            if model is None:
                action = eval_env.action_space.sample()
            else:
                state_t = (
                    torch.as_tensor(np.asarray(state), dtype=torch.uint8, device=device)
                    .float()
                    .unsqueeze(0)
                )
                with torch.inference_mode():
                    action = model(state_t).argmax(dim=1).item()

            state, reward, term, trunc, _ = eval_env.step(action)
            done = term or trunc
            total_reward += float(reward)

        scores.append(total_reward)

    scores = np.asarray(scores, dtype=np.float32)
    result = (float(scores.mean()), float(scores.std(ddof=0)))
    return (*result, scores) if return_scores else result
