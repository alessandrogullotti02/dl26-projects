from .results import load_json


def resolve_config(config: dict) -> dict:
    p = config["parameters"]
    if config["experiment"] == "epsilon":
        p["STRATEGY_LABELS"]["constant"] = f"Constant epsilon={p['CONSTANT_EPS']}"
    return config


def load_config(path) -> dict:
    return resolve_config(load_json(path))
