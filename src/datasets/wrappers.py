import gymnasium as gym


class FireResetEnv(gym.Wrapper):
    """Premere FIRE al reset per servire la pallina."""

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs, _, term, trunc, info = self.env.step(1)
        if term or trunc:
            obs, info = self.env.reset(**kwargs)
        return obs, info


class PongThreeActions(gym.ActionWrapper):
    """Espone tre azioni mappate su ALE 0, 2 e 3.

    In Pong i nomi ALE ufficiali sono NOOP, RIGHT e LEFT; nel progetto le
    azioni 2/3 corrispondono semanticamente al movimento verticale della
    racchetta controllata.
    """

    def __init__(self, env, action_seed=None):
        super().__init__(env)
        self.action_mapping = [0, 2, 3]
        self.action_space = gym.spaces.Discrete(3)
        if action_seed is not None:
            self.action_space.seed(int(action_seed))

    def action(self, action):
        return self.action_mapping[int(action)]
