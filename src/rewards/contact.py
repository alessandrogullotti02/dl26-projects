import numpy as np
import gymnasium as gym


class PongBallContactDetector:


    def __init__(
        self,
        play_y=(8, 80),
        ball_x=(10, 72),
        player_x=(72, 82),
        contrast_threshold=35,
        motion_threshold=10,
        max_ball_pixels=40,
        min_horizontal_speed=0.5,
        min_contact_x=60.0,
        max_vertical_gap=14.0,
    ):
        self.play_y = tuple(play_y)
        self.ball_x = tuple(ball_x)
        self.player_x = tuple(player_x)
        self.contrast_threshold = float(contrast_threshold)
        self.motion_threshold = float(motion_threshold)
        self.max_ball_pixels = int(max_ball_pixels)
        self.min_horizontal_speed = float(min_horizontal_speed)
        self.min_contact_x = float(min_contact_x)
        self.max_vertical_gap = float(max_vertical_gap)

    def _background(self, frame):
        y0, y1 = self.play_y
        x0, x1 = self.ball_x
        return float(np.median(frame[y0:y1, x0:x1]))

    def _locate_current_ball(self, previous, current):
        previous = previous.astype(np.int16, copy=False)
        current = current.astype(np.int16, copy=False)
        y0, y1 = self.play_y
        x0, x1 = self.ball_x
        background = self._background(current)

        current_roi = current[y0:y1, x0:x1]
        previous_roi = previous[y0:y1, x0:x1]
        object_mask = np.abs(current_roi - background) >= self.contrast_threshold
        moving_mask = np.abs(current_roi - previous_roi) >= self.motion_threshold
        y_local, x_local = np.nonzero(object_mask & moving_mask)

        if not 1 <= y_local.size <= self.max_ball_pixels:
            return None
        return {
            "x": float(np.median(x_local + x0)),
            "y": float(np.median(y_local + y0)),
        }

    def _locate_player_paddle(self, frame):
        frame = frame.astype(np.int16, copy=False)
        y0, y1 = self.play_y
        x0, x1 = self.player_x
        background = self._background(frame)
        roi = frame[y0:y1, x0:x1]
        y_local, _ = np.nonzero(np.abs(roi - background) >= self.contrast_threshold)
        if y_local.size < 2:
            return None
        return float(np.median(y_local + y0))

    def analyze(self, observation):
        frames = np.asarray(observation)
        if frames.ndim != 3 or frames.shape[0] < 4:
            raise ValueError("Il rilevatore richiede uno stack di almeno 4 frame")

        frames = frames[-4:]
        positions = [
            self._locate_current_ball(frames[i - 1], frames[i]) for i in range(1, 4)
        ]
        paddle_y = self._locate_player_paddle(frames[-1])

        details = {
            "contact": False,
            "positions": positions,
            "paddle_y": paddle_y,
            "dx_before": None,
            "dx_after": None,
            "turning_x": None,
            "turning_y": None,
        }
        if paddle_y is None or any(position is None for position in positions):
            return details

        dx_before = positions[1]["x"] - positions[0]["x"]
        dx_after = positions[2]["x"] - positions[1]["x"]
        turning_x = positions[1]["x"]
        turning_y = positions[1]["y"]

        moving_toward_player = dx_before >= self.min_horizontal_speed
        moving_away_after = dx_after <= -self.min_horizontal_speed
        near_player_side = turning_x >= self.min_contact_x
        vertically_compatible = abs(turning_y - paddle_y) <= self.max_vertical_gap

        details.update(
            {
                "contact": bool(
                    moving_toward_player
                    and moving_away_after
                    and near_player_side
                    and vertically_compatible
                ),
                "dx_before": dx_before,
                "dx_after": dx_after,
                "turning_x": turning_x,
                "turning_y": turning_y,
            }
        )
        return details

    def __call__(self, observation):
        return self.analyze(observation)["contact"]


class BallContactRewardShaping(gym.Wrapper):


    def __init__(self, env, contact_bonus, detector, cooldown_steps=3):
        super().__init__(env)
        if contact_bonus < 0.0:
            raise ValueError("contact_bonus deve essere non negativo")
        if cooldown_steps < 0:
            raise ValueError("cooldown_steps deve essere non negativo")
        self.contact_bonus = float(contact_bonus)
        self.detector = detector
        self.cooldown_steps = int(cooldown_steps)
        self.cooldown = 0

    def reset(self, **kwargs):
        self.cooldown = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        observation, raw_reward, terminated, truncated, info = self.env.step(action)
        raw_reward = float(raw_reward)
        candidate_contact = bool(self.detector(observation))

        if self.cooldown > 0:
            self.cooldown -= 1

        ball_contact = candidate_contact and self.cooldown == 0
        bonus = self.contact_bonus if ball_contact else 0.0
        if ball_contact:
            self.cooldown = self.cooldown_steps

        shaped_reward = raw_reward + bonus
        info = dict(info)
        info["raw_reward"] = raw_reward
        info["ball_contact"] = ball_contact
        info["candidate_contact"] = candidate_contact
        info["contact_bonus"] = bonus
        return observation, shaped_reward, terminated, truncated, info
