import argparse
import os

import yaml
from PIL import Image

from src.ga import run_ga, run_ga_islands
from src.fitness import load_target
from src.rendering import render_individual_fast as render_individual


DEFAULTS = {
    "pop_size": 150,
    "generations": 5000,
    "selection_method": "rank",
    "tournament_k": 3,
    "crossover_method": "uniform",
    "crossover_rate": 0.8,
    "mutation_method": "multi",
    "mutation_rate": 0.05,
    "sigma": 10.0,
    "elite_size": 2,
    "seed": None,
    "early_stop_generations": 500,
    "snapshot_interval": 0,
    "snapshot_dir": "data/snapshots",
    "fast_fitness": False,
    "fitness_mode": "rmse",
    "local_search_iterations": 5,
    "island_model": False,
    "num_islands": 4,
    "migration_interval": 50,
    "migration_size": 2,
    "fitness_sharing": False,
    "restricted_mating": False,
    "restricted_mating_pool": 5,
}


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Run the triangle GA")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to a YAML config file (overrides defaults; CLI args override config)")
    parser.add_argument("--run-id", type=str, default=None,
                        help="Name for the log file (default: 'run')")
    parser.add_argument("--pop-size", type=int)
    parser.add_argument("--generations", type=int)
    parser.add_argument("--selection-method", type=str,
                        choices=["tournament", "roulette", "rank"])
    parser.add_argument("--tournament-k", type=int)
    parser.add_argument("--crossover-method", type=str,
                        choices=["single_point", "uniform", "gene_single_point", "gene_uniform", "blend", "fitness_weighted_blend"])
    parser.add_argument("--crossover-rate", type=float)
    parser.add_argument("--mutation-method", type=str,
                        choices=["gaussian", "vertex_jitter", "color_perturbation", "triangle_reset", "swap", "adaptive", "multi", "sort_by_area", "adaptive_vertex_jitter"])
    parser.add_argument("--mutation-rate", type=float)
    parser.add_argument("--sigma", type=float)
    parser.add_argument("--elite-size", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--early-stop", type=int)
    parser.add_argument("--snapshot-interval", type=int)
    parser.add_argument("--snapshot-dir", type=str)
    parser.add_argument("--fast-fitness", action="store_true", default=None,
                        help="Use half-resolution rendering for faster fitness evaluation")
    parser.add_argument("--fitness-mode", type=str,
                        choices=["rmse", "hybrid", "perceptual"])
    parser.add_argument("--local-search-iterations", type=int,
                        help="Hill climbing iterations on elite individuals per generation (0=disabled)")
    parser.add_argument("--island-model", action="store_true", default=None,
                        help="Use island model with multiple sub-populations")
    parser.add_argument("--num-islands", type=int,
                        help="Number of islands (default: 4)")
    parser.add_argument("--migration-interval", type=int,
                        help="Migrate best individuals every N generations (default: 50)")
    parser.add_argument("--migration-size", type=int,
                        help="Number of individuals to migrate between islands (default: 2)")
    parser.add_argument("--fitness-sharing", action="store_true", default=None,
                        help="Enable fitness sharing")
    parser.add_argument("--restricted-mating", action="store_true", default=None,
                        help="Enable restricted mating (best partial match)")
    parser.add_argument("--restricted-mating-pool", type=int,
                        help="Pool size for restricted mating candidate sampling (default: 5)")
    args = parser.parse_args()

    # Priority: defaults < config file < CLI args
    cfg = dict(DEFAULTS)
    if args.config:
        cfg.update(load_config(args.config))

    cli_overrides = {
        "run_id": args.run_id,
        "pop_size": args.pop_size,
        "generations": args.generations,
        "selection_method": args.selection_method,
        "tournament_k": args.tournament_k,
        "crossover_method": args.crossover_method,
        "crossover_rate": args.crossover_rate,
        "mutation_method": args.mutation_method,
        "mutation_rate": args.mutation_rate,
        "sigma": args.sigma,
        "elite_size": args.elite_size,
        "seed": args.seed,
        "early_stop_generations": args.early_stop,
        "snapshot_interval": args.snapshot_interval,
        "snapshot_dir": args.snapshot_dir,
        "fast_fitness": args.fast_fitness,
        "fitness_mode": args.fitness_mode,
        "local_search_iterations": args.local_search_iterations,
        "island_model": args.island_model,
        "num_islands": args.num_islands,
        "migration_interval": args.migration_interval,
        "migration_size": args.migration_size,
        "fitness_sharing": args.fitness_sharing,
        "restricted_mating": args.restricted_mating,
        "restricted_mating_pool": args.restricted_mating_pool,
    }
    cfg.update({k: v for k, v in cli_overrides.items() if v is not None})

    target = load_target()

    ga_params = dict(
        pop_size=cfg["pop_size"],
        generations=cfg["generations"],
        selection_method=cfg["selection_method"],
        tournament_k=cfg["tournament_k"],
        crossover_method=cfg["crossover_method"],
        crossover_rate=cfg["crossover_rate"],
        mutation_method=cfg["mutation_method"],
        mutation_rate=cfg["mutation_rate"],
        sigma=cfg["sigma"],
        elite_size=cfg["elite_size"],
        seed=cfg["seed"],
        early_stop_generations=cfg["early_stop_generations"],
        snapshot_interval=cfg["snapshot_interval"],
        snapshot_dir=cfg["snapshot_dir"],
        fast_fitness=cfg["fast_fitness"],
        fitness_mode=cfg["fitness_mode"],
        local_search_iterations=cfg["local_search_iterations"],
        fitness_sharing=cfg["fitness_sharing"],
        restricted_mating=cfg["restricted_mating"],
        restricted_mating_pool=cfg["restricted_mating_pool"],
    )

    if cfg["island_model"]:
        best_individual, best_fitness, history = run_ga_islands(
            target_image=target,
            num_islands=cfg["num_islands"],
            migration_interval=cfg["migration_interval"],
            migration_size=cfg["migration_size"],
            **ga_params,
        )
    else:
        best_individual, best_fitness, history = run_ga(
            target_image=target, **ga_params)

    print("Final best fitness:", best_fitness)
    os.makedirs("data/snapshots", exist_ok=True)

    img = render_individual(best_individual)
    Image.fromarray(img).save("data/snapshots/ga_result.png")

    # Save convergence log
    run_id = cfg.get("run_id", "run")
    log_dir = "experiments/logs"
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{run_id}.csv")

    import csv
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["generation", "best_fitness", "avg_fitness", "worst_fitness"])
        for i, (b, a, w) in enumerate(zip(
            history["best_fitness"], history["avg_fitness"], history["worst_fitness"]
        )):
            writer.writerow([i + 1, b, a, w])

    print(f"Log saved to {log_path}")


if __name__ == "__main__":
    main()
