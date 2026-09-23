from functools import partial
import numpy as np
import matplotlib.pyplot as plt
from src.datasets.environment import make_env as build_env
from src.rewards.potential import PongAlignmentPotential


def inspect_potential_detector(config, n_samples=6, max_steps=500, seed=None):
    seed = config["parameters"]["SEED"] if seed is None else seed
    make_env = partial(build_env, config)
    detector = PongAlignmentPotential(**config["parameters"]["POTENTIAL_KWARGS"])
    diagnostic_env = make_env(use_potential=False)
    diagnostic_env.action_space.seed(seed)
    state, _ = diagnostic_env.reset(seed=seed)
    samples = []
    detected_steps = 0
    for _ in range(max_steps):
        action = diagnostic_env.action_space.sample()
        state, _, term, trunc, _ = diagnostic_env.step(action)
        objects = detector.estimate(state)
        detected_steps += int(objects["detected"])
        if objects["detected"] and len(samples) < n_samples:
            samples.append((np.asarray(state)[-1].copy(), objects, detector(state)))
        if term or trunc:
            state, _ = diagnostic_env.reset()
    diagnostic_env.close()
    print(
        f"Rilevamento completo in {detected_steps}/{max_steps} passi ({100 * detected_steps / max_steps:.1f}%)."
    )
    if not samples:
        print("Nessun campione valido: correggere regioni o soglie.")
        return
    fig, axes = plt.subplots(1, len(samples), figsize=(3 * len(samples), 4))
    axes = np.atleast_1d(axes)
    for axis, (frame, objects, potential) in zip(axes, samples):
        axis.imshow(frame, cmap="gray", vmin=0, vmax=255)
        axis.scatter(
            objects["ball_x"],
            objects["ball_y"],
            c="red",
            s=45,
            marker="x",
            label="pallina",
        )
        axis.axhline(
            objects["paddle_y"], color="cyan", linewidth=1.5, label="racchetta"
        )
        axis.set_title(f"Phi={potential:.3f}")
        axis.axis("off")
    axes[0].legend(loc="lower left", fontsize=8)
    plt.tight_layout()
    plt.show()
