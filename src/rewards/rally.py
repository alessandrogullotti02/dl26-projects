import numpy as np
import gymnasium as gym


class RallyRewardShaping(gym.Wrapper):


    def __init__(self, env, step_bonus, max_rally_bonus):
        super().__init__(env)
        if step_bonus < 0.0:
            raise ValueError("step_bonus deve essere non negativo")
        if max_rally_bonus < 0.0:
            raise ValueError("max_rally_bonus deve essere non negativo")

        self.step_bonus = float(step_bonus)
        self.max_rally_bonus = float(max_rally_bonus)
        self.rally_steps = 0
        self.rally_bonus = 0.0

    def reset(self, **kwargs):
        self.rally_steps = 0
        self.rally_bonus = 0.0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, raw_reward, term, trunc, info = self.env.step(action)
        raw_reward = float(raw_reward)
        completed_rally_steps = None
        shaping_bonus = 0.0

        if raw_reward == 0.0 and not (term or trunc):
            self.rally_steps += 1
            remaining = max(0.0, self.max_rally_bonus - self.rally_bonus)
            shaping_bonus = min(self.step_bonus, remaining)
            self.rally_bonus += shaping_bonus
        else:
            if raw_reward != 0.0:
                completed_rally_steps = self.rally_steps
            self.rally_steps = 0
            self.rally_bonus = 0.0

        shaped_reward = raw_reward + shaping_bonus
        info = dict(info)
        info["raw_reward"] = raw_reward
        info["shaping_bonus"] = shaping_bonus
        info["rally_steps"] = self.rally_steps
        info["completed_rally_steps"] = completed_rally_steps
        return obs, shaped_reward, term, trunc, info
