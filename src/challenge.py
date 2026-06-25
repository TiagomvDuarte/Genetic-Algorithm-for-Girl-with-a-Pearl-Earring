"""
challenge.py

Challenge #3 — Component Contribution Analysis (Ablation Study)

Runs 5 experiments × N seeds to isolate the contribution of each GA component:

    | Experiment       | Selection | Crossover | Mutation | Purpose                         |
    |------------------|-----------|-----------|----------|---------------------------------|
    | Full GA          | ✓         | ✓         | ✓        | Baseline — all components on    |
    | No crossover     | ✓         | ✗         | ✓        | Isolate mutation's contribution |
    | No mutation      | ✓         | ✓         | ✗        | Isolate crossover's contribution|
    | Random selection | Random    | ✓         | ✓        | Isolate selection pressure      |
    | Mutation only    | Random    | ✗         | ✓        | Pure random search + mutation   |
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from src.ga import run_ga
from src.fitness import load_target


# ---------------------------------------------------------------------------
# Experiment definitions
# ---------------------------------------------------------------------------

# Best configuration found via parameter search (notebook 05_improved_ga):
# pop_size=150, selection=rank, crossover=uniform, mutation=multi, mutation_rate=0.05, sigma=10, elite_size=2
BEST_CONFIG = {
    "pop_size":         150,
    "selection_method": "rank",
    "crossover_method": "uniform",
    "crossover_rate":   0.8,
    "mutation_method":  "multi",
    "mutation_rate":    0.05,
    "sigma":            10.0,
    "elite_size":       2,
    "fast_fitness":     True,
}

EXPERIMENTS = {
    "full_ga": {
        "label": "Full GA",
        **BEST_CONFIG,
    },
    "no_crossover": {
        "label": "No Crossover",
        **BEST_CONFIG,
        "crossover_rate": 0.0,
    },
    "no_mutation": {
        "label": "No Mutation",
        **BEST_CONFIG,
        "mutation_rate": 0.0,
    },
    "random_selection": {
        "label": "Random Selection",
        **BEST_CONFIG,
        "selection_method": "random",
    },
    "mutation_only": {
        "label": "Mutation Only",
        **BEST_CONFIG,
        "selection_method": "random",
        "crossover_rate":   0.0,
    },
}

DEFAULT_SEEDS = [42, 123, 456, 789, 1337]


# ---------------------------------------------------------------------------
# Full ablation study
# ---------------------------------------------------------------------------

def run_ablation_study(
    generations: int = 3000,
    pop_size: int | None = None,
    seeds: list[int] | None = None,
    experiments: list[str] | None = None,
    output_dir: str = "experiments",
) -> dict:
    """
    Run the full ablation study.

    Parameters
    ----------
    generations : int
        Number of GA generations per run.
    pop_size : int | None
        Override population size. If None, uses the value from BEST_CONFIG.
    seeds : list of int
        Random seeds (default: 5 seeds for statistical robustness).
    experiments : list of str
        Subset of experiment keys to run (default: all 5).
    output_dir : str
        Root directory to save logs and figures.

    Returns
    -------
    results : dict  {experiment_key: [run_result, ...]}
    """
    if seeds is None:
        seeds = DEFAULT_SEEDS
    if experiments is None:
        experiments = list(EXPERIMENTS.keys())

    for exp_key in experiments:
        if exp_key not in EXPERIMENTS:
            raise ValueError(f"Unknown experiment: '{exp_key}'. Choose from {list(EXPERIMENTS.keys())}")

    target_image = load_target()

    os.makedirs(os.path.join(output_dir, "logs"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "figures"), exist_ok=True)

    results: dict[str, list] = {key: [] for key in experiments}

    total = len(experiments) * len(seeds)
    done = 0

    for exp_key in experiments:
        for seed in seeds:
            done += 1
            config = EXPERIMENTS[exp_key]
            print(f"\n[{done}/{total}] Experiment: {config['label']} | Seed: {seed}")

            kwargs = {
                **{k: v for k, v in config.items() if k != "label"},
                "generations": generations,
                "seed":        seed,
            }
            if pop_size is not None:
                kwargs["pop_size"] = pop_size

            _, best_fit, history = run_ga(target_image, **kwargs)

            results[exp_key].append({
                "label":        config["label"],
                "seed":         seed,
                "best_fitness": float(best_fit),
                "best_per_gen": history["best_fitness"],
            })
            print(f"  -> Best fitness: {best_fit:.4f}")

    _save_logs(results, output_dir)
    _plot_convergence(results, output_dir)
    _print_summary(results)

    return results


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _save_logs(results: dict, output_dir: str):
    """Save per-experiment aggregated logs as JSON."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(output_dir, "logs", f"ablation_{timestamp}.json")

    with open(path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nLogs saved to {path}")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_convergence(results: dict, output_dir: str):
    """Plot mean convergence curve (best fitness vs generation) for each experiment.
    Shaded area = ±1 std across seeds.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.tab10.colors  # type: ignore

    for i, (key, runs) in enumerate(results.items()):
        if not runs:
            continue
        curves = np.array([r["best_per_gen"] for r in runs])
        mean = curves.mean(axis=0)
        std  = curves.std(axis=0)
        gens = np.arange(1, len(mean) + 1)
        color = colors[i % len(colors)]
        ax.plot(gens, mean, label=EXPERIMENTS[key]["label"], color=color, linewidth=1.8)
        ax.fill_between(gens, mean - std, mean + std, alpha=0.15, color=color)

    ax.set_xlabel("Generation")
    ax.set_ylabel("Best RMSE")
    ax.set_title("Ablation Study — Convergence Curves (mean ± std)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(output_dir, "figures", f"ablation_convergence_{timestamp}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Convergence plot saved to {path}")


def plot_final_fitness_comparison(results: dict, output_dir: str = "experiments/figures"):
    """Bar chart of mean final best fitness per experiment, with error bars (std)."""
    os.makedirs(output_dir, exist_ok=True)

    labels, means, stds = [], [], []
    for key, runs in results.items():
        if not runs:
            continue
        finals = [r["best_fitness"] for r in runs]
        labels.append(EXPERIMENTS[key]["label"])
        means.append(np.mean(finals))
        stds.append(np.std(finals))

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=stds, capsize=5, color=plt.cm.tab10.colors[:len(labels)])  # type: ignore
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Final Best RMSE")
    ax.set_title("Ablation Study — Final Fitness Comparison")
    ax.grid(axis="y", alpha=0.3)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(output_dir, f"ablation_bar_{timestamp}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Bar chart saved to {path}")
    return path


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def _print_summary(results: dict):
    print("\n" + "=" * 60)
    print("ABLATION STUDY — SUMMARY")
    print("=" * 60)
    print(f"{'Experiment':<22} {'Mean RMSE':>10} {'Std':>8} {'Best':>8} {'Worst':>8}")
    print("-" * 60)
    for key, runs in results.items():
        if not runs:
            continue
        finals = [r["best_fitness"] for r in runs]
        print(
            f"{EXPERIMENTS[key]['label']:<22} "
            f"{np.mean(finals):>10.4f} "
            f"{np.std(finals):>8.4f} "
            f"{np.min(finals):>8.4f} "
            f"{np.max(finals):>8.4f}"
        )
    print("=" * 60)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_ablation_study(generations=3000)
