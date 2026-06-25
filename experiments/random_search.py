"""
random_search.py

Runs a random search over GA hyperparameters by sampling random configurations
and evaluating each one. Better than grid search when the parameter space is large.

Each configuration is sampled randomly from defined ranges and run with multiple
seeds. Results are saved to experiments/logs/random_search_summary.csv.

Usage:
    cd Project
    python experiments/random_search.py
    python experiments/random_search.py --n-configs 50 --seeds 42 123
"""

import argparse
import csv
import os
import subprocess
import sys
import random


SEEDS = [42]

# Search space — one random value is sampled per parameter per config
SEARCH_SPACE = {
    "pop_size":         [30, 50, 100],
    "generations":      [200],
    "selection_method": ["tournament", "roulette", "rank"],
    "tournament_k":     [2, 3, 5, 10],
    "crossover_method": ["single_point", "uniform", "gene_single_point", "gene_uniform"],
    "crossover_rate":   [0.6, 0.7, 0.8, 0.9],
    "mutation_method":  ["gaussian", "adaptive", "vertex_jitter", "color_perturbation"],
    "mutation_rate":    [0.01, 0.05, 0.10, 0.20, 0.30],
    "sigma":            [5.0, 10.0, 20.0, 50.0],
    "elite_size":       [0, 1, 3, 5],
}


def sample_config(rng: random.Random) -> dict:
    return {param: rng.choice(values) for param, values in SEARCH_SPACE.items()}


def config_to_args(cfg: dict) -> list:
    return [
        "--pop-size",         str(cfg["pop_size"]),
        "--generations",      str(cfg["generations"]),
        "--selection-method", cfg["selection_method"],
        "--tournament-k",     str(cfg["tournament_k"]),
        "--crossover-method", cfg["crossover_method"],
        "--crossover-rate",   str(cfg["crossover_rate"]),
        "--mutation-method",  cfg["mutation_method"],
        "--mutation-rate",    str(cfg["mutation_rate"]),
        "--sigma",            str(cfg["sigma"]),
        "--elite-size",       str(cfg["elite_size"]),
    ]


def config_name(cfg: dict, idx: int) -> str:
    return (
        f"rs{idx:03d}"
        f"_pop{cfg['pop_size']}"
        f"_sel{cfg['selection_method']}"
        f"_cx{cfg['crossover_method']}"
        f"_mut{cfg['mutation_method']}"
        f"_rate{cfg['mutation_rate']}"
        f"_sigma{cfg['sigma']}"
    )


def run_random_search(n_configs: int = 30, seeds: list = None, log_dir: str = "experiments/logs"):
    if seeds is None:
        seeds = SEEDS

    os.makedirs(log_dir, exist_ok=True)
    summary_path = os.path.join(log_dir, "random_search_summary.csv")

    rng = random.Random(0)  # fixed seed for reproducible config sampling
    configs = [sample_config(rng) for _ in range(n_configs)]

    total = n_configs * len(seeds)
    count = 0

    with open(summary_path, "w", newline="") as summary_file:
        writer = csv.writer(summary_file)
        writer.writerow([
            "config_id", "seed", "run_id",
            *SEARCH_SPACE.keys(),
        ])

        for idx, cfg in enumerate(configs):
            name = config_name(cfg, idx)
            print(f"\nConfig [{idx+1}/{n_configs}]: {name}")
            print("  " + "  ".join(f"{k}={v}" for k, v in cfg.items()))

            for seed in seeds:
                count += 1
                run_id = f"{name}_seed{seed}"
                print(f"  [{count}/{total}] seed={seed} → {run_id}")

                cmd = [
                    sys.executable, "run.py",
                    *config_to_args(cfg),
                    "--seed", str(seed),
                    "--run-id", run_id,
                ]

                result = subprocess.run(cmd, cwd=os.path.join(os.path.dirname(__file__), ".."))

                if result.returncode != 0:
                    print(f"    ERROR in {run_id}")
                else:
                    writer.writerow([idx, seed, run_id, *cfg.values()])
                    summary_file.flush()

    print(f"\nRandom search complete. Summary: {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Random search over GA hyperparameters")
    parser.add_argument("--n-configs", type=int, default=70,
                        help="Number of random configurations to sample (default: 30)")
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS,
                        help="Seeds to run each config with (default: 42 123 456)")
    args = parser.parse_args()

    run_random_search(n_configs=args.n_configs, seeds=args.seeds)
