"""Train a reaching policy with SAC (Soft Actor-Critic).

Usage:
    uv run python train.py                      # 2-joint arm, 200k steps
    uv run python train.py --steps 100000 --links 0.1 0.1

SAC keeps a memory ("replay buffer") of everything the arm has tried and
keeps re-learning from it, which makes it very sample-efficient: small arm
tasks like this one learn in minutes on a laptop CPU.
"""

import argparse
import time

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from reach_env import ReachEnv


class ProgressLog(BaseCallback):
    """Prints how close the fingertip ends up, every few thousand steps."""

    def __init__(self, every=5000):
        super().__init__()
        self.every, self.final_dists, self.history = every, [], []

    def _on_step(self):
        for done, info in zip(self.locals["dones"], self.locals["infos"]):
            if done:
                self.final_dists.append(info["distance"])
        if self.num_timesteps % self.every == 0 and self.final_dists:
            recent = np.array(self.final_dists[-20:])
            self.history.append((self.num_timesteps, float(recent.mean())))
            print(f"step {self.num_timesteps:>7,}  |  avg final distance {recent.mean() * 100:5.1f} cm  "
                  f"|  success {np.mean(recent < 0.02) * 100:3.0f}%", flush=True)
        return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=200_000)
    p.add_argument("--links", type=float, nargs="+", default=[0.1, 0.1])
    p.add_argument("--out", default=None)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    env = Monitor(ReachEnv(link_lengths=args.links))
    out = args.out or f"models/sac_{len(args.links)}joint"

    model = SAC("MlpPolicy", env, seed=args.seed, verbose=0, learning_starts=2000,
                policy_kwargs=dict(net_arch=[128, 128]), device="cpu")
    log = ProgressLog()
    start = time.time()
    model.learn(total_timesteps=args.steps, callback=log)
    model.save(out)
    np.save(out + "_curve.npy", np.array(log.history))
    print(f"\ndone in {(time.time() - start) / 60:.1f} min -> saved {out}.zip")


if __name__ == "__main__":
    main()
