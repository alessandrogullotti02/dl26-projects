from functools import partial
import numpy as np
import matplotlib.pyplot as plt
from src.datasets.environment import make_env as build_env
from src.rewards.contact import PongBallContactDetector


def inspect_contact_detector(config, n_samples=4, max_steps=3000, seed=None):
    seed = config["parameters"]["SEED"] if seed is None else seed
    make_env = partial(build_env, config)
    detector = PongBallContactDetector(
        **config["parameters"]["CONTACT_DETECTOR_KWARGS"]
    )
    diagnostic_env = make_env(contact_bonus=0.0)
    diagnostic_env.action_space.seed(seed)
    state, _ = diagnostic_env.reset(seed=seed)
    samples = []
    candidate_count = 0
    for _ in range(max_steps):
        action = diagnostic_env.action_space.sample()
        state, _, terminated, truncated, _ = diagnostic_env.step(action)
        details = detector.analyze(state)
        if details["contact"]:
            candidate_count += 1
            if len(samples) < n_samples:
                samples.append((np.asarray(state)[-1].copy(), details))
        if terminated or truncated:
            state, _ = diagnostic_env.reset()
    diagnostic_env.close()
    print(f"Contatti candidati: {candidate_count} in {max_steps} passi.")
    if not samples:
        print(
            "Nessun contatto visualizzabile: aumentare max_steps oppure correggere regioni e soglie."
        )
        return
    fig, axes = plt.subplots(1, len(samples), figsize=(4 * len(samples), 4))
    axes = np.atleast_1d(axes)
    for axis, (frame, details) in zip(axes, samples):
        axis.imshow(frame, cmap="gray", vmin=0, vmax=255)
        for index, position in enumerate(details["positions"]):
            axis.scatter(
                position["x"], position["y"], c="red", s=35 + 15 * index, marker="x"
            )
        axis.axhline(details["paddle_y"], color="cyan", linewidth=1.5)
        axis.set_title(f"dx: {details['dx_before']:.1f} -> {details['dx_after']:.1f}")
        axis.axis("off")
    plt.tight_layout()
    plt.show()
