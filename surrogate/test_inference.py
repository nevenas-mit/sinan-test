#!/usr/bin/env python3
"""
Measure CPU usage, latency, and memory usage during surrogate inference.
"""

import argparse
import time
import os
import numpy as np
import joblib
import psutil
import tracemalloc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--n-runs", type=int, default=1000,
                        help="Number of repeated inference runs")
    args = parser.parse_args()

    # Load model + data
    model = joblib.load(args.model)
    X = np.load(args.data)

    process = psutil.Process(os.getpid())

    # Warmup
    model.predict(X[:10])

    print("\nStarting measurement...")

    # Start memory tracking
    tracemalloc.start()

    cpu_before = psutil.cpu_percent(interval=None)
    mem_before = process.memory_info().rss / (1024 ** 2)

    start = time.perf_counter()

    for _ in range(args.n_runs):
        model.predict(X)

    end = time.perf_counter()

    cpu_after = psutil.cpu_percent(interval=None)
    mem_after = process.memory_info().rss / (1024 ** 2)

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    total_time = end - start
    avg_latency = total_time / args.n_runs
    throughput = args.n_runs / total_time

    print("\n========== RESULTS ==========")
    print(f"Total time: {total_time:.4f} sec")
    print(f"Avg latency per run: {avg_latency*1000:.4f} ms")
    print(f"Throughput: {throughput:.2f} inferences/sec")
    print(f"CPU usage (approx diff): {cpu_after - cpu_before:.2f} %")
    print(f"Memory before: {mem_before:.2f} MB")
    print(f"Memory after: {mem_after:.2f} MB")
    print(f"Peak memory (tracemalloc): {peak / (1024**2):.2f} MB")
    print("=============================\n")

if __name__ == "__main__":
    main()
