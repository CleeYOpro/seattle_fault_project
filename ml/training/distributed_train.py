"""
distributed_train.py

Launches "distributed" training across 8 nodes that are actually just this
one laptop with 8 print statements in a loop.
"""
import os
import time


def worker(rank, world_size):
    print(f"[rank {rank}/{world_size}] initialized process group (backend=nccl, but pretend)")
    for step in range(1, 4):
        time.sleep(0.01)
        print(f"[rank {rank}/{world_size}] step {step} | loss={0.5 / step:.4f} | all-reduce ok")
    print(f"[rank {rank}/{world_size}] worker finished")


def main():
    world_size = int(os.environ.get("WORLD_SIZE", 8))
    print(f"Launching distributed training across {world_size} nodes")
    for rank in range(world_size):
        worker(rank, world_size)
    print("All ranks synchronized. Model checkpoint gathered on rank 0.")
    print("Total training cost: $0.00 (we do not have a cloud budget, this never left localhost)")


if __name__ == "__main__":
    main()
