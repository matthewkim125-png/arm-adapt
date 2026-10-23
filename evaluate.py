"""Score a trained policy, compare it with random flailing, and make a GIF.

Usage:
    uv run python evaluate.py                    # uses models/sac_2joint.zip
    uv run python evaluate.py --model models/sac_2joint --links 0.1 0.1
"""

import argparse
import os

import imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import SAC

from reach_env import ReachEnv


def run_episodes(env, policy, n, seed=1000):
    """Final fingertip distance for n fresh targets (same targets for every policy)."""
    finals = []
    for i in range(n):
        obs, _ = env.reset(seed=seed + i)
        done = False
        while not done:
            obs, _, term, trunc, info = env.step(policy(obs))
            done = term or trunc
        finals.append(info["distance"])
    return np.array(finals)


def make_gif(env, policy, path, n_targets=6, seed=2000):
    frames = []
    for i in range(n_targets):
        obs, _ = env.reset(seed=seed + i)
        done = False
        while not done:
            obs, _, term, trunc, _ = env.step(policy(obs))
            done = term or trunc
            frames.append(env.render())
    imageio.mimsave(path, frames[::2], duration=0.04, loop=0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="models/sac_2joint")
    p.add_argument("--links", type=float, nargs="+", default=[0.1, 0.1])
    p.add_argument("--episodes", type=int, default=200)
    args = p.parse_args()

    os.makedirs("results", exist_ok=True)
    env = ReachEnv(link_lengths=args.links)
    model = SAC.load(args.model, device="cpu")
    trained = lambda obs: model.predict(obs, deterministic=True)[0]
    random_policy = lambda obs: env.action_space.sample()

    name = os.path.basename(args.model)
    for label, policy in [("random", random_policy), ("trained", trained)]:
        d = run_episodes(env, policy, args.episodes)
        print(f"{label:>8}: avg final distance {d.mean() * 100:5.2f} cm  |  "
              f"within 2 cm: {np.mean(d < 0.02) * 100:5.1f}%  |  worst {d.max() * 100:5.2f} cm")

    make_gif(env, trained, f"results/{name}.gif")
    print(f"saved results/{name}.gif")

    curve_path = args.model + "_curve.npy"
    if os.path.exists(curve_path):
        steps, dist = np.load(curve_path).T
        plt.figure(figsize=(6, 3.5))
        plt.plot(steps / 1000, dist * 100, marker="o", markersize=3)
        plt.axhline(2, ls="--", c="gray", lw=1, label="2 cm success line")
        plt.xlabel("training steps (thousands)")
        plt.ylabel("final distance to target (cm)")
        plt.title("Learning to reach: 2-joint arm")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"results/{name}_curve.png", dpi=150)
        print(f"saved results/{name}_curve.png")


if __name__ == "__main__":
    main()
