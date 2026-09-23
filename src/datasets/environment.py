import gymnasium as gym
import ale_py
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation
from .wrappers import FireResetEnv, PongThreeActions
from src.rewards.rally import RallyRewardShaping
from src.rewards.potential import PongAlignmentPotential, PotentialBasedRewardShaping
from src.rewards.contact import PongBallContactDetector, BallContactRewardShaping


def make_env(
    config: dict,
    step_bonus=0.0,
    max_rally_bonus=0.0,
    use_potential=False,
    potential_scale=None,
    contact_bonus=0.0,
    action_seed=None,
):
    """Costruisce l'ambiente con la stessa catena usata nei notebook DL2."""

    gym.register_envs(ale_py)
    settings = dict(config["environment"])
    env_id = settings.pop("id")
    frameskip = settings.pop("frameskip")
    stack_size = settings.pop("stack_size")
    repeat_action_probability = settings.pop("repeat_action_probability", 0.25)

    env = gym.make(
        env_id,
        frameskip=frameskip,
        repeat_action_probability=repeat_action_probability,
    )
    env = AtariPreprocessing(env, **settings)
    env = FireResetEnv(env)
    env = PongThreeActions(env, action_seed=action_seed)

    # Rally shaping opera sul reward Atari prima del frame stack.
    if step_bonus > 0.0 and max_rally_bonus > 0.0:
        env = RallyRewardShaping(env, step_bonus, max_rally_bonus)

    env = FrameStackObservation(env, stack_size=stack_size)
    p = config["parameters"]

    # PBRS e contact detector devono vedere lo stesso stack 4x84x84 della DQN.
    if use_potential:
        kwargs = dict(p["POTENTIAL_KWARGS"])
        kwargs.pop("scale", None)
        scale = (
            p.get("MODERATE_POTENTIAL_SCALE", p.get("POTENTIAL_SCALE", 0.10))
            if potential_scale is None
            else potential_scale
        )
        env = PotentialBasedRewardShaping(
            env,
            PongAlignmentPotential(scale=scale, **kwargs),
            p["GAMMA"],
        )

    if contact_bonus > 0.0:
        env = BallContactRewardShaping(
            env,
            contact_bonus,
            PongBallContactDetector(**p["CONTACT_DETECTOR_KWARGS"]),
            p["CONTACT_COOLDOWN_STEPS"],
        )

    return env
