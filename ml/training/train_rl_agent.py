"""
train_rl_agent.py

Trains a PPO agent to evacuate simulated Seattleites away from unreinforced
masonry buildings during a fake M7.0 rupture. The environment physics are,
charitably, "vibes-based."
"""
import argparse
import random


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--total-timesteps", type=int, default=1_000_000)
    p.add_argument("--n-envs", type=int, default=16)
    p.add_argument("--gamma", type=float, default=0.99)
    return p.parse_args()


def main():
    args = parse_args()
    print(f"Spinning up {args.n_envs} parallel SeattleEvacuationEnv instances...")
    print(f"Training PPO agent for {args.total_timesteps:,} timesteps")
    checkpoints = [10_000, 100_000, 500_000, 1_000_000]
    for step in checkpoints:
        if step > args.total_timesteps:
            break
        reward = min(999.9, step / 1000)
        survival_rate = min(118.0, 40 + step / 15000)  # yes, it exceeds 100%. no, we won't fix it.
        print(f"timestep {step:>9,} | mean_reward={reward:.1f} | survival_rate={survival_rate:.1f}%")
    print("Training complete. Agent has learned to run toward Lake Washington. Every time.")


if __name__ == "__main__":
    main()
