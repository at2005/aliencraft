# PPO + random-policy baselines; episode return = tech-tree rungs climbed
#   uv run python baselines/ppo_sb3.py --steps 1000000 --envs 8
import argparse

# SB3's NatureCNN needs >= 36px observations
ENV_KWARGS = dict(visual_field_size=64)

import gymnasium as gym
import numpy as np
import torch

import aliencraft  # noqa: F401  (registers AlienCraft-v0)
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv


class Uint8Obs(gym.ObservationWrapper):
    # SB3's CnnPolicy wants uint8 images; the env emits float [0, 1]
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            0, 255, env.observation_space.shape, np.uint8
        )

    def observation(self, obs):
        return (obs * 255).astype(np.uint8)


def make_env(seed):
    def thunk():
        torch.set_num_threads(1)
        env = Monitor(Uint8Obs(gym.make("AlienCraft-v0", **ENV_KWARGS)))
        env.reset(seed=seed)
        return env

    return thunk


def random_baseline(episodes):
    env = Uint8Obs(gym.make("AlienCraft-v0", **ENV_KWARGS))
    returns = []
    for ep in range(episodes):
        env.reset(seed=10_000 + ep)
        total, done = 0.0, False
        while not done:
            _, r, term, trunc, _ = env.step(env.action_space.sample())
            total += r
            done = term or trunc
        returns.append(total)
    return float(np.mean(returns)), float(np.std(returns))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1_000_000)
    parser.add_argument("--envs", type=int, default=8)
    parser.add_argument("--eval-episodes", type=int, default=20)
    args = parser.parse_args()

    mean, std = random_baseline(args.eval_episodes)
    print(f"random policy: {mean:.2f} +/- {std:.2f} types discovered/episode")

    vec = SubprocVecEnv([make_env(seed) for seed in range(args.envs)])
    model = PPO(
        "CnnPolicy",
        vec,
        n_steps=256,
        batch_size=512,
        ent_coef=0.01,
        verbose=1,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    model.learn(total_timesteps=args.steps, progress_bar=False)
    model.save("baselines/ppo_aliencraft")

    eval_env = Monitor(Uint8Obs(gym.make("AlienCraft-v0", **ENV_KWARGS)))
    eval_env.reset(seed=99_000)
    mean, std = evaluate_policy(model, eval_env, n_eval_episodes=args.eval_episodes)
    print(f"PPO ({args.steps} steps): {mean:.2f} +/- {std:.2f} types discovered/episode")


if __name__ == "__main__":
    main()
