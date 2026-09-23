import json
from pathlib import Path


def _numeric_keys(value):
    if isinstance(value, dict):
        return {
            int(k) if k.isdigit() else k: _numeric_keys(v) for k, v in value.items()
        }
    if isinstance(value, list):
        return [_numeric_keys(v) for v in value]
    return value


def load_json(path):

    with Path(path).open(encoding="utf-8") as stream:
        return _numeric_keys(json.load(stream))


def save_json(path, value):

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    temporary.replace(path)


def save_checkpoint(payload, filename, step, config, output_dir):

    import torch

    policy = config["checkpoint_policy"]
    if policy == "none" or (
        policy == "final" and step != config["parameters"]["TOTAL_STEPS"]
    ):
        return
    destination = Path(output_dir) / "weights"
    destination.mkdir(parents=True, exist_ok=True)

    if isinstance(payload, dict) and "epsilon_final" in payload:
        payload = dict(payload, epsilon_final=float(payload["epsilon_final"]))
    torch.save(payload, destination / filename)


def load_checkpoint(path, device="cpu"):

    import numpy as np
    import torch

    with torch.serialization.safe_globals(
        [
            np._core.multiarray.scalar,
            np.dtype,
            np.dtypes.Float64DType,
        ]
    ):
        return torch.load(path, map_location=device, weights_only=True)
