import numpy as np


def epsilon_value(strategy, step, config):

    if step <= config["parameters"]["LEARNING_STARTS"]:
        return config["parameters"]["EPS_START"]
    schedule_step = step - config["parameters"]["LEARNING_STARTS"]
    if strategy == "constant":
        return config["parameters"]["CONSTANT_EPS"]
    progress = min(schedule_step / config["parameters"]["DECAY_STEPS"], 1.0)
    if strategy == "linear":
        return config["parameters"]["EPS_START"] + progress * (
            config["parameters"]["EPS_END"] - config["parameters"]["EPS_START"]
        )
    if strategy == "exponential":
        exp_end = np.exp(-config["parameters"]["EXPONENTIAL_K"])
        normalized = (
            np.exp(-config["parameters"]["EXPONENTIAL_K"] * progress) - exp_end
        ) / (1.0 - exp_end)
        return (
            config["parameters"]["EPS_END"]
            + (config["parameters"]["EPS_START"] - config["parameters"]["EPS_END"])
            * normalized
        )
    raise ValueError(f"Strategia sconosciuta: {strategy}")
