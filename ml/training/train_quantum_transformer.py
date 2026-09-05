"""
train_quantum_transformer.py

Trains QuantumFaultNet, a "quantum-enhanced" transformer for fault rupture
prediction. Spoiler: there is no quantum computer involved, just a for-loop
and unwarranted confidence.
"""
import argparse
import random

SEGMENTS = [
    "Restoration Point",
    "Bainbridge Island",
    "Downtown Seattle",
    "Lake Washington",
    "Issaquah",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--n-qubits", type=int, default=12)
    p.add_argument("--simulate-decoherence", action="store_true", default=True)
    return p.parse_args()


def main():
    args = parse_args()
    print(f"Entangling {args.n_qubits} qubits with a 24-layer transformer...")
    print("(Note: 'entangling' here means calling random.choice(). Sorry.)")
    for epoch in range(1, args.epochs + 1):
        if epoch % 10 == 0 or epoch == 1:
            loss = 2.5 / epoch ** 1.3
            pred = random.choice(SEGMENTS)
            conf = min(0.999, 0.5 + epoch * 0.01)
            print(f"epoch {epoch:>4} | loss={loss:.4f} | prediction={pred!r} | confidence={conf:.3f}")
    print("Model believes it can predict earthquakes. Model is wrong. Model is very confident.")


if __name__ == "__main__":
    main()
