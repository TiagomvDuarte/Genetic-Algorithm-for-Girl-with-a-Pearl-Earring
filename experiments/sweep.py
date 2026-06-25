"""
sweep.py

Runs multiple GA experiments by calling run.py with different parameters.
Results are saved to experiments/logs/ as CSVs.

Usage:
    cd Project
    python experiments/sweep.py
"""

import csv
import os
import subprocess
import sys

SEEDS = [42, 123, 456]

BASE = [
    "--pop-size", "50",
    "--generations", "200",
    "--selection-method", "tournament",
    "--tournament-k", "3",
    "--crossover-method", "single_point",
    "--crossover-rate", "0.8",
    "--mutation-method", "gaussian",
    "--mutation-rate", "0.1",
    "--sigma", "10.0",
    "--elite-size", "1",
]

EXPERIMENTS = [
    # Selection
    {"name": "selection_tournament", "args": ["--selection-method", "tournament"]},
    {"name": "selection_roulette",   "args": ["--selection-method", "roulette"]},
    {"name": "selection_rank",       "args": ["--selection-method", "rank"]},

    # Crossover
    {"name": "crossover_single_point", "args": ["--crossover-method", "single_point"]},
    {"name": "crossover_uniform",      "args": ["--crossover-method", "uniform"]},
    {"name": "crossover_gene_single",  "args": ["--crossover-method", "gene_single_point"]},
    {"name": "crossover_gene_uniform", "args": ["--crossover-method", "gene_uniform"]},

    # Mutation method
    {"name": "mutation_gaussian",           "args": ["--mutation-method", "gaussian"]},
    {"name": "mutation_adaptive",           "args": ["--mutation-method", "adaptive"]},
    {"name": "mutation_vertex_jitter",      "args": ["--mutation-method", "vertex_jitter"]},
    {"name": "mutation_color_perturbation", "args": ["--mutation-method", "color_perturbation"]},

    # Mutation rate
    {"name": "mutation_rate_001", "args": ["--mutation-rate", "0.01"]},
    {"name": "mutation_rate_005", "args": ["--mutation-rate", "0.05"]},
    {"name": "mutation_rate_010", "args": ["--mutation-rate", "0.10"]},
    {"name": "mutation_rate_030", "args": ["--mutation-rate", "0.30"]},

    # Sigma
    {"name": "sigma_5",  "args": ["--sigma", "5.0"]},
    {"name": "sigma_10", "args": ["--sigma", "10.0"]},
    {"name": "sigma_20", "args": ["--sigma", "20.0"]},
    {"name": "sigma_50", "args": ["--sigma", "50.0"]},

    # Population size
    {"name": "pop_30",  "args": ["--pop-size", "30"]},
    {"name": "pop_50",  "args": ["--pop-size", "50"]},
    {"name": "pop_100", "args": ["--pop-size", "100"]},

    # Tournament k
    {"name": "tournament_k2",  "args": ["--selection-method", "tournament", "--tournament-k", "2"]},
    {"name": "tournament_k5",  "args": ["--selection-method", "tournament", "--tournament-k", "5"]},
    {"name": "tournament_k10", "args": ["--selection-method", "tournament", "--tournament-k", "10"]},

    # Elite size
    {"name": "elite_0", "args": ["--elite-size", "0"]},
    {"name": "elite_1", "args": ["--elite-size", "1"]},
    {"name": "elite_5", "args": ["--elite-size", "5"]},
]


def run_sweep(log_dir: str = "experiments/logs"):
    os.makedirs(log_dir, exist_ok=True)
    summary_path = os.path.join(log_dir, "sweep_summary.csv")

    with open(summary_path, "w", newline="") as summary_file:
        writer = csv.writer(summary_file)
        writer.writerow(["experiment", "seed", "run_id"])

        total = len(EXPERIMENTS) * len(SEEDS)
        count = 0

        for exp in EXPERIMENTS:
            for seed in SEEDS:
                count += 1
                run_id = f"{exp['name']}_seed{seed}"
                print(f"\n[{count}/{total}] {run_id}")

                cmd = [
                    sys.executable, "run.py",
                    *BASE,
                    *exp["args"],
                    "--seed", str(seed),
                    "--run-id", run_id,
                ]

                result = subprocess.run(cmd, cwd=os.path.join(os.path.dirname(__file__), ".."))

                if result.returncode != 0:
                    print(f"  ERROR in {run_id}")
                else:
                    writer.writerow([exp["name"], seed, run_id])
                    summary_file.flush()

    print(f"\nSweep complete. Summary: {summary_path}")


if __name__ == "__main__":
    run_sweep()