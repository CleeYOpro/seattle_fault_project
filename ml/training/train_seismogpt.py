"""
train_seismogpt.py

Fine-tunes SeismoGPT-7B on the Seattle Fault Zone corpus.
100% LARP. Do not run. Do not cite. Do not deploy to production seismographs.
"""
import argparse
import random
import time

import torch
import torch.nn as nn


class FaultLineAttention(nn.Module):
    """Attention mechanism that is definitely aware of tectonic plate boundaries."""

    def __init__(self, d_model=4096, n_heads=32):
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.fault_bias = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

    def forward(self, x):
        out, _ = self.mha(x, x, x)
        return out + self.fault_bias


class SeismoGPT(nn.Module):
    def __init__(self, d_model=4096, n_layers=32, vocab_size=128_000):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([FaultLineAttention(d_model) for _ in range(n_layers)])
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids):
        x = self.embed(input_ids)
        for block in self.blocks:
            x = block(x)
        return self.lm_head(x)


def parse_args():
    p = argparse.ArgumentParser(description="Train SeismoGPT-7B on Seattle Fault data")
    p.add_argument("--epochs", type=int, default=9999)
    p.add_argument("--batch-size", type=int, default=69420)
    p.add_argument("--lr", type=float, default=6.9e-4)
    p.add_argument("--use-quantum-annealer", action="store_true", default=True)
    p.add_argument("--num-gpus", type=int, default=8, help="Number of A100s we do not actually have")
    return p.parse_args()


def fake_loss(epoch, total_epochs):
    base = 4.2
    return max(0.0001, base * (1 - epoch / total_epochs) ** 3 + random.uniform(-0.01, 0.01))


def main():
    args = parse_args()
    print(f"Initializing SeismoGPT-7B across {args.num_gpus} GPUs (allegedly)...")
    model = SeismoGPT()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model has {n_params:,} parameters (that's a lot of matrices)")

    print("Loading Seattle Fault Zone dataset from ../data/EQinSeattle.csv ...")
    time.sleep(0.1)
    print("Loaded 118,342 training examples, 4,021 eval examples")

    print(f"Beginning training for {args.epochs} epochs, batch_size={args.batch_size}, lr={args.lr}")
    for epoch in range(1, args.epochs + 1):
        if epoch % max(1, args.epochs // 20) == 0 or epoch == 1:
            loss = fake_loss(epoch, args.epochs)
            print(f"epoch {epoch:>5}/{args.epochs} | loss={loss:.5f} | tokens/sec=inf | wandb: synced")
        if epoch > 50:
            # Skip ahead, we all know how this goes
            break

    print("Training complete. Saving checkpoint to ../models/seismogpt_7b_seattle_final_v3_REAL_FINAL.pt")
    print("Final eval perplexity: 1.00009 (suspiciously good, please do not peer review this)")


if __name__ == "__main__":
    main()
