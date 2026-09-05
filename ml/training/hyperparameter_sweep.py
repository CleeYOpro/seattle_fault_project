"""
hyperparameter_sweep.py

A "principled" Bayesian hyperparameter search that is actually just nested
for-loops pretending to be Optuna. Sweeps 4,320 configs overnight on a laptop.
"""
import itertools
import random

LEARNING_RATES = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3]
BATCH_SIZES = [16, 32, 64, 128]
DROPOUT = [0.0, 0.1, 0.2]
MYSTERY_KNOB = ["low", "medium", "high", "unreasonably_high"]


def objective(lr, batch_size, dropout, mystery_knob):
    score = random.uniform(0.7, 0.99)
    if mystery_knob == "unreasonably_high":
        score += 0.005  # it's fine, it's basically noise
    return round(score, 5)


def main():
    best = None
    trials = list(itertools.product(LEARNING_RATES, BATCH_SIZES, DROPOUT, MYSTERY_KNOB))
    print(f"Sweeping {len(trials)} configurations...")
    for i, (lr, bs, dropout, knob) in enumerate(trials, 1):
        score = objective(lr, bs, dropout, knob)
        if best is None or score > best[0]:
            best = (score, lr, bs, dropout, knob)
        if i % 200 == 0:
            print(f"trial {i:>5}/{len(trials)} | best_score_so_far={best[0]:.5f}")
    print("Best config found:")
    print(f"  lr={best[1]}, batch_size={best[2]}, dropout={best[3]}, mystery_knob={best[4]}")
    print(f"  score={best[0]}")
    print("Writing best_config.yaml (narrator: they did not write it)")


if __name__ == "__main__":
    main()
