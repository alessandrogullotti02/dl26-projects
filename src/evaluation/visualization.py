"""Post-hoc diagnostics used by the baseline Visualization extra.

The executed reference implementation, including plotting, is preserved in
`notebooks/baseline_analysis.ipynb`.  This module exposes the reusable numerical
parts without touching the training loop.
"""

import numpy as np
import torch
from scipy import ndimage

ACTION_NAMES = ("NOOP", "UP", "DOWN")


def _components(mask):
    labels, _ = ndimage.label(mask)
    found = []
    for idx, box in enumerate(ndimage.find_objects(labels), start=1):
        if box is None:
            continue
        ys, xs = np.where(labels[box] == idx)
        h = box[0].stop - box[0].start
        w = box[1].stop - box[1].start
        found.append((
            xs.mean() + box[1].start,
            ys.mean() + box[0].start,
            w,
            h,
            len(xs),
        ))
    return found


def detect_ball_and_right_paddle(rgb):
    """Euristic detector used only to interpret the learned policy."""
    rgb = np.asarray(rgb)
    if rgb.shape != (210, 160, 3):
        return None

    ball_mask = np.all(rgb > 180, axis=-1)
    ball_mask[:34] = False
    ball_mask[194:] = False
    ball_mask[:, :10] = False
    ball_mask[:, 150:] = False
    balls = [
        c for c in _components(ball_mask)
        if 1 <= c[2] <= 5 and 1 <= c[3] <= 6 and 2 <= c[4] <= 24
    ]

    background = np.median(rgb[40:190, 20:135].reshape(-1, 3), axis=0)
    foreground = np.max(
        np.abs(rgb.astype(np.float32) - background), axis=-1
    ) > 30
    foreground[:34] = False
    foreground[194:] = False
    foreground[:, :136] = False
    foreground[:, 151:] = False
    paddles = [
        c for c in _components(foreground)
        if 1 <= c[2] <= 6 and 7 <= c[3] <= 25 and c[4] >= 10
    ]

    if len(balls) != 1 or len(paddles) != 1:
        return None
    bx, by = balls[0][:2]
    px, py = paddles[0][:2]
    return np.asarray([bx, by, px, py], dtype=np.float32)


def score_model_on_probe(model, target_model, probe, gamma, device, batch_size=128):
    """Compute TD errors, Q estimates, greedy actions and action gap.

    `probe` must contain `states`, `next_last`, `actions`, `rewards`, and
    `terminated`, matching the dataset built in the reference notebook.
    """
    q_chunks, td_chunks = [], []
    n = len(probe["actions"])

    model.eval(); target_model.eval()
    for start in range(0, n, batch_size):
        stop = min(start + batch_size, n)
        sl = slice(start, stop)
        states = torch.as_tensor(
            probe["states"][sl], dtype=torch.uint8, device=device
        ).float()
        next_states_np = np.concatenate(
            [probe["states"][sl, 1:], probe["next_last"][sl, None]], axis=1
        )
        next_states = torch.as_tensor(
            next_states_np, dtype=torch.uint8, device=device
        ).float()

        with torch.inference_mode():
            q = model(states).cpu().numpy()
            q_next = target_model(next_states).cpu().numpy()

        actions = probe["actions"][sl]
        rewards = probe["rewards"][sl]
        terminated = probe["terminated"][sl].astype(np.float32)
        target = rewards + gamma * (1.0 - terminated) * q_next.max(axis=1)
        td = target - q[np.arange(len(q)), actions]
        q_chunks.append(q)
        td_chunks.append(td)

    q = np.concatenate(q_chunks)
    td = np.concatenate(td_chunks)
    sorted_q = np.sort(q, axis=1)
    return {
        "q": q,
        "td": td,
        "action_gap": sorted_q[:, -1] - sorted_q[:, -2],
        "greedy_actions": q.argmax(axis=1),
    }


def summarize_diagnostics(diagnostics):
    q = diagnostics["q"]
    td = diagnostics["td"]
    greedy = diagnostics["greedy_actions"]
    gap = diagnostics["action_gap"]
    return {
        "td_mean": float(np.mean(td)),
        "td_mae": float(np.mean(np.abs(td))),
        "td_std": float(np.std(td, ddof=0)),
        "td_p95_abs": float(np.quantile(np.abs(td), 0.95)),
        "q_mean_by_action": {
            ACTION_NAMES[a]: float(q[:, a].mean()) for a in range(3)
        },
        "q_std_by_action": {
            ACTION_NAMES[a]: float(q[:, a].std(ddof=0)) for a in range(3)
        },
        "max_q_mean": float(q.max(axis=1).mean()),
        "action_gap_mean": float(gap.mean()),
        "greedy_action_frequency": {
            ACTION_NAMES[a]: float(np.mean(greedy == a)) for a in range(3)
        },
    }


def policy_heatmap(features, greedy_actions, direction, min_bin=5, min_purity=0.60):
    """Project a greedy policy onto ball position / paddle misalignment bins."""
    x_edges = np.linspace(0, 160, 9)
    y_edges = np.linspace(-160, 160, 11)
    features = np.asarray(features)
    greedy_actions = np.asarray(greedy_actions)

    valid = np.isfinite(features).all(axis=1) & (direction * features[:, 2] > 0)
    f = features[valid]
    a = greedy_actions[valid]
    counts = np.stack([
        np.histogram2d(
            f[a == action, 1],
            f[a == action, 0],
            bins=[y_edges, x_edges],
        )[0]
        for action in range(3)
    ])
    occupancy = counts.sum(axis=0)
    modal_action = counts.argmax(axis=0).astype(float)
    purity = counts.max(axis=0) / np.maximum(occupancy, 1)
    modal_action[(occupancy < min_bin) | (purity < min_purity)] = np.nan
    return modal_action, occupancy, purity, x_edges, y_edges


def plot_policy_heatmaps(
    probe_features,
    diagnostics_by_seed,
    seeds,
    output_path,
    min_bin=5,
    min_purity=0.60,
):
    """Render the report-friendly 3x2 policy heatmap layout.

    Titles are shared by column and axis labels are global, preventing the
    overlap present in the original notebook rendering.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap

    cmap = ListedColormap(["#7f7f7f", "#1f77b4", "#d62728"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)
    fig, axes = plt.subplots(
        len(seeds), 2, figsize=(12.5, 10.5), sharex=True, sharey=True,
        layout="constrained", squeeze=False,
    )

    for row, seed in enumerate(seeds):
        actions = diagnostics_by_seed[seed]["greedy_actions"]
        for col, direction in enumerate((-1, +1)):
            heat, _, _, x_edges, y_edges = policy_heatmap(
                probe_features, actions, direction, min_bin, min_purity
            )
            ax = axes[row, col]
            image = ax.imshow(
                heat,
                origin="lower",
                aspect="auto",
                extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
                cmap=cmap,
                norm=norm,
                interpolation="nearest",
            )
            if row == 0:
                ax.set_title(
                    "Pallina verso sinistra"
                    if direction < 0
                    else "Pallina verso la racchetta destra",
                    fontsize=12,
                    pad=8,
                )
            if col == 0:
                ax.annotate(
                    f"Seed {seed}", xy=(-0.16, 0.5), xycoords="axes fraction",
                    rotation=90, va="center", ha="center", fontsize=11,
                    fontweight="bold",
                )
            ax.tick_params(labelsize=9)

    fig.supxlabel("Posizione orizzontale della pallina [pixel RGB]", fontsize=11)
    fig.supylabel(
        "Disallineamento verticale: y pallina - y racchetta [pixel RGB]",
        fontsize=11,
    )
    colorbar = fig.colorbar(image, ax=axes, ticks=[0, 1, 2], shrink=0.74, pad=0.025)
    colorbar.ax.set_yticklabels(ACTION_NAMES)
    colorbar.set_label("Azione greedy", fontsize=10)
    fig.suptitle(
        "Vanilla DQN - proiezione empirica della policy su stati osservati\n"
        f"Bin visualizzato se n >= {min_bin} e purezza >= {min_purity:.0%}",
        fontsize=13,
    )
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
