"""The reaching task as a Gymnasium environment.

An RL environment answers three questions, every step:
  1. What can the agent see?      -> observation
  2. What can the agent do?       -> action
  3. How well did it just do?     -> reward

Stable-Baselines3 (the training library) only talks to the arm through
reset() and step(), so anything that follows this interface can be trained.
"""

import gymnasium as gym
import mujoco
import numpy as np

from arm_model import make_arm_xml


class ReachEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 25}

    def __init__(self, link_lengths=(0.1, 0.1), episode_steps=100, frame_skip=2, render_mode=None):
        self.link_lengths = tuple(link_lengths)
        self.model = mujoco.MjModel.from_xml_string(make_arm_xml(self.link_lengths))
        self.data = mujoco.MjData(self.model)
        self.n_joints = self.model.nu
        self.reach = sum(self.link_lengths)
        self.episode_steps = episode_steps
        self.frame_skip = frame_skip  # physics steps per agent decision (0.02 s)
        self.render_mode = render_mode
        self._renderer = None

        self._tip = self.model.site("fingertip").id
        self._target_mocap = self.model.body("target").mocapid[0]

        # Action: one number per motor in [-1, 1] (full torque one way or the other).
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(self.n_joints,), dtype=np.float32)
        obs_size = 3 * self.n_joints + 4
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(obs_size,), dtype=np.float32)

    # --- what the agent sees -------------------------------------------------
    def _obs(self):
        q = self.data.qpos[: self.n_joints]
        qd = self.data.qvel[: self.n_joints]
        tip = self.data.site_xpos[self._tip][:2]
        target = self.data.mocap_pos[self._target_mocap][:2]
        # cos/sin instead of raw angles so 359 deg and 1 deg look close together.
        return np.concatenate([np.cos(q), np.sin(q), qd, target, target - tip]).astype(np.float32)

    def _distance(self):
        tip = self.data.site_xpos[self._tip][:2]
        target = self.data.mocap_pos[self._target_mocap][:2]
        return float(np.linalg.norm(tip - target))

    # --- start a new attempt -------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[: self.n_joints] = self.np_random.uniform(-0.1, 0.1, self.n_joints)
        self.data.qpos[0] = self.np_random.uniform(-np.pi, np.pi)

        # Pick a random target the arm can actually reach (not right on top of the base).
        radius = self.np_random.uniform(0.15, 0.95) * self.reach
        angle = self.np_random.uniform(-np.pi, np.pi)
        self.data.mocap_pos[self._target_mocap] = [radius * np.cos(angle), radius * np.sin(angle), 0.0]

        mujoco.mj_forward(self.model, self.data)
        self._t = 0
        return self._obs(), {}

    # --- one decision --------------------------------------------------------
    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        self.data.ctrl[:] = action
        mujoco.mj_step(self.model, self.data, nstep=self.frame_skip)
        self._t += 1

        dist = self._distance()
        # Reward: get close (main term), don't flail (small penalty on effort).
        # Distance is in arm-lengths so the same numbers work for any arm size.
        reward = -dist / self.reach - 0.05 * float(np.sum(np.square(action)))

        truncated = self._t >= self.episode_steps
        info = {"distance": dist, "success": dist < 0.02}
        return self._obs(), reward, False, truncated, info

    def render(self):
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, 480, 480)
        self._renderer.update_scene(self.data, "top")
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


if __name__ == "__main__":
    from gymnasium.utils.env_checker import check_env

    env = ReachEnv()
    check_env(env, skip_render_check=True)
    obs, _ = env.reset(seed=0)
    total = 0.0
    for _ in range(env.episode_steps):
        obs, r, term, trunc, info = env.step(env.action_space.sample())
        total += r
    print(f"env check passed. random policy: return {total:.1f}, final distance {info['distance'] * 100:.1f} cm")
