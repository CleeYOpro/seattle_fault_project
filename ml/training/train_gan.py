"""
train_gan.py

Trains EarthquakeGAN, a WGAN-GP that generates synthetic Seattle Fault
seismograms. Entirely fabricated for repo-flavor purposes.
"""
import argparse
import random


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--z-dim", type=int, default=128)
    p.add_argument("--critic-iters", type=int, default=5)
    p.add_argument("--gp-weight", type=float, default=10.0)
    return p.parse_args()


def main():
    args = parse_args()
    print(f"Training EarthquakeGAN for {args.epochs} epochs (z_dim={args.z_dim})")
    for epoch in range(1, args.epochs + 1):
        if epoch % 50 == 0 or epoch == 1:
            d_loss = 1.0 / epoch
            g_loss = 0.8 / epoch
            wdist = d_loss * 0.42 + random.uniform(-0.01, 0.01)
            print(f"epoch {epoch:>4} | d_loss={d_loss:.4f} | g_loss={g_loss:.4f} | w_distance={wdist:.4f}")
    print("Saved generator to ../models/earthquake_gan_generator_v7_final.pt")
    print("Fake Inception Score (FID but for seismograms, we made it up): 4.2")


if __name__ == "__main__":
    main()
