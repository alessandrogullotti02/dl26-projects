import numpy as np
import gymnasium as gym


class PongAlignmentPotential:


    def __init__(
        self,
        scale=0.10,
        play_y=(8, 80),
        ball_x=(10, 70),
        player_x=(70, 82),
        contrast_threshold=35,
        motion_threshold=10,
        max_ball_pixels=40,
    ):
        if scale < 0.0:
            raise ValueError("scale deve essere non negativo")
        self.scale = float(scale)
        self.play_y = tuple(play_y)
        self.ball_x = tuple(ball_x)
        self.player_x = tuple(player_x)
        self.contrast_threshold = float(contrast_threshold)
        self.motion_threshold = float(motion_threshold)
        self.max_ball_pixels = int(max_ball_pixels)

    @staticmethod
    def _last_two_frames(observation):
        frames = np.asarray(observation)
        if frames.ndim == 2:
            return frames, frames
        if frames.ndim != 3 or frames.shape[0] < 2:
            raise ValueError(
                "Il potenziale richiede un frame 84x84 o uno stack (N, 84, 84)"
            )
        return frames[-2], frames[-1]

    def estimate(self, observation):
        previous, current = self._last_two_frames(observation)
        previous = previous.astype(np.int16, copy=False)
        current = current.astype(np.int16, copy=False)
        height, width = current.shape

        y0, y1 = self.play_y
        bx0, bx1 = self.ball_x
        px0, px1 = self.player_x

        if not (0 <= y0 < y1 <= height and 0 <= bx0 < bx1 <= width):
            raise ValueError("Regione della pallina non compatibile con il frame")
        if not (0 <= px0 < px1 <= width):
            raise ValueError("Regione della racchetta non compatibile con il frame")


        background = float(np.median(current[y0:y1, bx0:bx1]))


        paddle_roi = current[y0:y1, px0:px1]
        paddle_mask = np.abs(paddle_roi - background) >= self.contrast_threshold
        paddle_y_local, _ = np.nonzero(paddle_mask)
        paddle_y = (
            float(np.median(paddle_y_local + y0)) if paddle_y_local.size >= 2 else None
        )


        ball_current = current[y0:y1, bx0:bx1]
        ball_previous = previous[y0:y1, bx0:bx1]
        object_mask = np.abs(ball_current - background) >= self.contrast_threshold
        moving_mask = np.abs(ball_current - ball_previous) >= self.motion_threshold
        ball_mask = object_mask & moving_mask
        ball_y_local, ball_x_local = np.nonzero(ball_mask)

        valid_ball = 1 <= ball_y_local.size <= self.max_ball_pixels
        ball_y = float(np.median(ball_y_local + y0)) if valid_ball else None
        ball_x = float(np.median(ball_x_local + bx0)) if valid_ball else None

        return {
            "ball_y": ball_y,
            "ball_x": ball_x,
            "paddle_y": paddle_y,
            "background": background,
            "detected": ball_y is not None and paddle_y is not None,
        }

    def __call__(self, observation):
        objects = self.estimate(observation)
        if not objects["detected"]:
            return 0.0

        height = np.asarray(observation).shape[-2]
        vertical_error = abs(objects["ball_y"] - objects["paddle_y"])
        alignment = 1.0 - vertical_error / max(1, height - 1)
        return self.scale * float(np.clip(alignment, 0.0, 1.0))


class PotentialBasedRewardShaping(gym.Wrapper):


    def __init__(self, env, potential_fn, gamma):
        super().__init__(env)
        if not 0.0 <= gamma <= 1.0:
            raise ValueError("gamma deve appartenere a [0, 1]")
        self.potential_fn = potential_fn
        self.gamma = float(gamma)
        self.current_potential = 0.0

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self.current_potential = float(self.potential_fn(observation))
        info = dict(info)
        info["potential"] = self.current_potential
        return observation, info

    def step(self, action):
        observation, raw_reward, terminated, truncated, info = self.env.step(action)
        raw_reward = float(raw_reward)


        next_potential = 0.0 if terminated else float(self.potential_fn(observation))
        shaping_reward = self.gamma * next_potential - self.current_potential
        shaped_reward = raw_reward + shaping_reward

        info = dict(info)
        info["raw_reward"] = raw_reward
        info["shaping_reward"] = shaping_reward
        info["potential_s"] = self.current_potential
        info["potential_next"] = next_potential

        self.current_potential = next_potential
        return observation, shaped_reward, terminated, truncated, info
