"""
ga.py

Main genetic algorithm loop for the triangle-based image reconstruction problem.

    - initialize population
    - evaluate fitness
    - select parents
    - apply crossover
    - apply mutation
    - apply elitism
    - track fitness history
"""

import os

import numpy as np
from PIL import Image

from src.representation import random_individual, NUM_TRIANGLES
from src.selection import select_parent, select_parent_with_index
from src.crossover import crossover
from src.mutation import (
    gaussian_mutation,
    vertex_jitter,
    color_perturbation,
    triangle_reset,
    swap_mutation,
    adaptive_mutation,
    multi_mutation,
    sort_by_area_mutation,
    adaptive_vertex_jitter,
)
from src.fitness import downscale_target, evaluate, evaluate_fast
from src.rendering import render_individual_fast as render_individual
from src.diversity import apply_fitness_sharing, select_parent_restricted_mating


def initialize_population(
    pop_size: int,
    rng: np.random.Generator,
    num_triangles: int = NUM_TRIANGLES,
) -> np.ndarray:
    """
    Create the initial population with target-independent random triangles.
    """
    individuals = [random_individual(rng, num_triangles=num_triangles)
                   for _ in range(pop_size)]
    return np.array(individuals, dtype=np.float32)


def evaluate_population(
    population: np.ndarray,
    target_image: np.ndarray,
    fast_fitness: bool = False,
    target_downscaled: np.ndarray | None = None,
    fitness_mode: str = "rmse",
) -> np.ndarray:
    """
    Evaluate all individuals in the population.
    Lower fitness is better.
    """
    if fast_fitness:
        return np.array(
            [evaluate_fast(individual, target_downscaled, fitness_mode=fitness_mode)
             for individual in population],
            dtype=np.float32
        )
    return np.array(
        [evaluate(individual, target_image, fitness_mode=fitness_mode)
         for individual in population],
        dtype=np.float32
    )


def mutate_child(
    child: np.ndarray,
    rng: np.random.Generator,
    mutation_method: str,
    mutation_rate: float,
    generation: int,
    generations: int,
    sigma: float
) -> np.ndarray:
    """
    Apply the chosen mutation operator to one child.
    """
    if mutation_method == "gaussian":
        return gaussian_mutation(child, rng=rng, sigma=sigma, mutation_rate=mutation_rate)

    if mutation_method == "vertex_jitter":
        return vertex_jitter(child, rng=rng, sigma=sigma, mutation_rate=mutation_rate)

    if mutation_method == "color_perturbation":
        return color_perturbation(child, rng=rng, sigma=sigma, mutation_rate=mutation_rate)

    if mutation_method == "triangle_reset":
        return triangle_reset(child, rng=rng, mutation_rate=mutation_rate)

    if mutation_method == "swap":
        return swap_mutation(child, rng=rng, mutation_rate=mutation_rate)

    if mutation_method == "adaptive":
        return adaptive_mutation(
            child,
            rng=rng,
            generation=generation,
            max_generations=generations,
            sigma_initial=sigma,
            sigma_final=1.0,
            mutation_rate=mutation_rate
        )

    if mutation_method == "multi":
        return multi_mutation(child, rng=rng, sigma=sigma, mutation_rate=mutation_rate)

    if mutation_method == "sort_by_area":
        return sort_by_area_mutation(child, rng=rng, mutation_rate=mutation_rate)

    if mutation_method == "adaptive_vertex_jitter":
        return adaptive_vertex_jitter(
            child,
            rng=rng,
            generation=generation,
            max_generations=generations,
            sigma_initial=sigma,
            sigma_final=1.0,
            mutation_rate=mutation_rate
        )

    raise ValueError(
        "Unknown mutation method. Use 'gaussian', 'vertex_jitter', "
        "'color_perturbation', 'triangle_reset', 'swap', 'adaptive', "
        "'multi', 'sort_by_area', or 'adaptive_vertex_jitter'."
    )


def local_search(
    individual: np.ndarray,
    target_image: np.ndarray,
    rng: np.random.Generator,
    n_iterations: int = 5,
    sigma: float = 3.0,
    fast_fitness: bool = False,
    target_downscaled: np.ndarray | None = None,
    fitness_mode: str = "rmse",
) -> tuple:
    """
    Hill climbing: perturb one random triangle per iteration, keep if fitness improves.
    """
    from src.representation import clamp_individual

    current = individual.copy()
    if fast_fitness:
        current_fit = evaluate_fast(current, target_downscaled, fitness_mode=fitness_mode)
    else:
        current_fit = evaluate(current, target_image, fitness_mode=fitness_mode)

    for _ in range(n_iterations):
        candidate = current.copy()
        tri_idx = rng.integers(0, len(candidate))
        candidate[tri_idx, 0:6] += rng.normal(0, sigma, 6)
        candidate[tri_idx, 6:9] += rng.normal(0, sigma * 0.5, 3)
        candidate = clamp_individual(candidate)

        if fast_fitness:
            cand_fit = evaluate_fast(candidate, target_downscaled, fitness_mode=fitness_mode)
        else:
            cand_fit = evaluate(candidate, target_image, fitness_mode=fitness_mode)

        if cand_fit < current_fit:
            current = candidate
            current_fit = cand_fit

    return current, current_fit


def run_ga(
    target_image: np.ndarray,
    pop_size: int = 50,
    generations: int = 100,
    selection_method: str = "tournament",
    tournament_k: int = 3,
    crossover_method: str = "single_point",
    crossover_rate: float = 0.8,
    mutation_method: str = "gaussian",
    mutation_rate: float = 0.1,
    sigma: float = 10.0,
    elite_size: int = 1,
    seed: int | None = None,
    early_stop_generations: int = 0,
    snapshot_interval: int = 0,
    snapshot_dir: str = "data/snapshots",
    fast_fitness: bool = False,
    fitness_mode: str = "rmse",
    local_search_iterations: int = 0,
    fitness_sharing: bool = False,
    restricted_mating: bool = False,
    restricted_mating_pool: int = 5,
):
    """
    Run the genetic algorithm.

    Parameters
    ----------
    target_image : np.ndarray
        Target image to approximate.
    pop_size : int
        Population size.
    generations : int
        Number of generations.
    selection_method : str
        'tournament', 'roulette', or 'rank'
    tournament_k : int
        Tournament size for tournament selection.
    crossover_method : str
        'single_point' or 'uniform'
    crossover_rate : float
        Probability of applying crossover (otherwise child = copy of parent A).
    mutation_method : str
        'gaussian', 'uniform', or 'adaptive'
    mutation_rate : float
        Mutation probability.
    sigma : float
        Sigma for gaussian/adaptive mutation.
    elite_size : int
        Number of best individuals copied directly to next generation.
    seed : int | None
        Random seed.
    early_stop_generations : int
        Stop if best fitness doesn't improve for this many generations. 0 = disabled.
    snapshot_interval : int
        Save best individual image every N generations. 0 = disabled.
    snapshot_dir : str
        Directory for snapshot images.

    Returns
    -------
    best_individual : np.ndarray
    best_fitness : float
    history : dict
    """
    rng = np.random.default_rng(seed)

    population = initialize_population(pop_size, rng)
    history = {
        "best_fitness": [],
        "avg_fitness": [],
        "worst_fitness": []
    }

    best_individual = None
    best_fitness = float("inf")
    generations_without_improvement = 0

    target_downscaled = downscale_target(target_image) if fast_fitness else None

    if snapshot_interval > 0:
        os.makedirs(snapshot_dir, exist_ok=True)

    for generation in range(generations):
        fitness = evaluate_population(
            population, target_image,
            fast_fitness=fast_fitness,
            target_downscaled=target_downscaled,
            fitness_mode=fitness_mode,
        )

        selection_fitness = apply_fitness_sharing(fitness, population) if fitness_sharing else fitness

        sorted_indices = np.argsort(fitness)

        current_best_idx = sorted_indices[0]
        current_best_fitness = float(fitness[current_best_idx])

        if current_best_fitness < best_fitness:
            best_fitness = current_best_fitness
            best_individual = population[current_best_idx].copy()
            generations_without_improvement = 0
        else:
            generations_without_improvement += 1

        history["best_fitness"].append(float(fitness[sorted_indices[0]]))
        history["avg_fitness"].append(float(np.mean(fitness)))
        history["worst_fitness"].append(float(fitness[sorted_indices[-1]]))

        # Save snapshot of best individual every N generations
        if snapshot_interval > 0 and (generation + 1) % snapshot_interval == 0:
            img = render_individual(best_individual)
            path = os.path.join(snapshot_dir, f"gen_{generation + 1}.png")
            Image.fromarray(img).save(path)

        new_population = []

        # Elitism: keep top N individuals
        for idx in sorted_indices[:elite_size]:
            elite = population[idx].copy()
            if local_search_iterations > 0:
                elite, _ = local_search(
                    elite, target_image, rng,
                    n_iterations=local_search_iterations,
                    sigma=sigma * 0.3,
                    fast_fitness=fast_fitness,
                    target_downscaled=target_downscaled,
                    fitness_mode=fitness_mode,
                )
            new_population.append(elite)

        # Create the rest of the new population
        uses_fitness_crossover = crossover_method == "fitness_weighted_blend"

        while len(new_population) < pop_size:
            if uses_fitness_crossover:
                parent_a, idx_a = select_parent_with_index(
                    population, selection_fitness, method=selection_method, rng=rng, k=tournament_k)
                parent_b, idx_b = select_parent_with_index(
                    population, selection_fitness, method=selection_method, rng=rng, k=tournament_k)
            elif restricted_mating:
                parent_a = select_parent(
                    population, selection_fitness, method=selection_method, rng=rng, k=tournament_k)
                parent_b = select_parent_restricted_mating(
                    population, selection_fitness, anchor=parent_a,
                    rng=rng, pool_size=restricted_mating_pool)
            else:
                parent_a = select_parent(
                    population, selection_fitness, method=selection_method, rng=rng, k=tournament_k)
                parent_b = select_parent(
                    population, selection_fitness, method=selection_method, rng=rng, k=tournament_k)

            if rng.random() < crossover_rate:
                if uses_fitness_crossover:
                    child = crossover(
                        parent_a, parent_b, method=crossover_method, rng=rng,
                        fitness_a=float(fitness[idx_a]),
                        fitness_b=float(fitness[idx_b]),
                    )
                else:
                    child = crossover(
                        parent_a, parent_b, method=crossover_method, rng=rng)
            else:
                child = parent_a.copy()

            child = mutate_child(
                child,
                rng=rng,
                mutation_method=mutation_method,
                mutation_rate=mutation_rate,
                generation=generation,
                generations=generations,
                sigma=sigma
            )

            new_population.append(child)

        population = np.array(new_population, dtype=np.float32)

        print(
            f"Generation {generation + 1}/{generations} | "
            f"Best: {history['best_fitness'][-1]:.4f} | "
            f"Avg: {history['avg_fitness'][-1]:.4f}"
        )

        # Early stopping
        if early_stop_generations > 0 and generations_without_improvement >= early_stop_generations:
            print(f"Early stop: no improvement for {early_stop_generations} generations.")
            break

    return best_individual, best_fitness, history


def _run_island_generation(
    population, target_image, fitness_arr, rng, generation, target_downscaled, **kw
):
    """Run one generation for a single island. Returns (new_population, fitness_arr)."""
    fast_fitness = kw.get("fast_fitness", False)
    fitness_mode = kw.get("fitness_mode", "rmse")
    selection_method = kw.get("selection_method", "tournament")
    tournament_k = kw.get("tournament_k", 3)
    crossover_method = kw.get("crossover_method", "single_point")
    crossover_rate = kw.get("crossover_rate", 0.8)
    mutation_method = kw.get("mutation_method", "gaussian")
    mutation_rate = kw.get("mutation_rate", 0.1)
    sigma = kw.get("sigma", 10.0)
    elite_size = kw.get("elite_size", 1)
    generations = kw.get("generations", 5000)
    local_search_iterations = kw.get("local_search_iterations", 0)

    if fitness_arr is None:
        fitness_arr = evaluate_population(
            population, target_image, fast_fitness=fast_fitness,
            target_downscaled=target_downscaled, fitness_mode=fitness_mode)

    sorted_indices = np.argsort(fitness_arr)
    pop_size = len(population)
    new_population = []

    for idx in sorted_indices[:elite_size]:
        elite = population[idx].copy()
        if local_search_iterations > 0:
            elite, _ = local_search(
                elite, target_image, rng, n_iterations=local_search_iterations,
                sigma=sigma * 0.3, fast_fitness=fast_fitness,
                target_downscaled=target_downscaled, fitness_mode=fitness_mode)
        new_population.append(elite)

    uses_fitness_crossover = crossover_method == "fitness_weighted_blend"
    while len(new_population) < pop_size:
        if uses_fitness_crossover:
            pa, ia = select_parent_with_index(population, fitness_arr, method=selection_method, rng=rng, k=tournament_k)
            pb, ib = select_parent_with_index(population, fitness_arr, method=selection_method, rng=rng, k=tournament_k)
        else:
            pa = select_parent(population, fitness_arr, method=selection_method, rng=rng, k=tournament_k)
            pb = select_parent(population, fitness_arr, method=selection_method, rng=rng, k=tournament_k)

        if rng.random() < crossover_rate:
            if uses_fitness_crossover:
                child = crossover(pa, pb, method=crossover_method, rng=rng,
                                  fitness_a=float(fitness_arr[ia]), fitness_b=float(fitness_arr[ib]))
            else:
                child = crossover(pa, pb, method=crossover_method, rng=rng)
        else:
            child = pa.copy()

        child = mutate_child(child, rng=rng, mutation_method=mutation_method,
                             mutation_rate=mutation_rate, generation=generation,
                             generations=generations, sigma=sigma)
        new_population.append(child)

    new_pop = np.array(new_population, dtype=np.float32)
    new_fit = evaluate_population(
        new_pop, target_image, fast_fitness=fast_fitness,
        target_downscaled=target_downscaled, fitness_mode=fitness_mode)
    return new_pop, new_fit


def run_ga_islands(
    target_image: np.ndarray,
    num_islands: int = 4,
    migration_interval: int = 50,
    migration_size: int = 2,
    **ga_kwargs,
):
    """
    Run multiple sub-populations (islands) with periodic migration.
    Ring topology: best individuals from island i replace worst in island (i+1) % num_islands.
    """
    total_pop = ga_kwargs.get("pop_size", 150)
    total_gens = ga_kwargs.get("generations", 5000)
    island_pop = max(10, total_pop // num_islands)
    seed = ga_kwargs.get("seed", None)
    fast_fitness = ga_kwargs.get("fast_fitness", False)
    fitness_mode = ga_kwargs.get("fitness_mode", "rmse")

    rng = np.random.default_rng(seed)
    target_downscaled = downscale_target(target_image) if fast_fitness else None

    # Initialize islands with separate populations and RNGs
    islands = []
    for _ in range(num_islands):
        island_rng = np.random.default_rng(int(rng.integers(0, 2**31)))
        pop = initialize_population(island_pop, island_rng)
        fit = evaluate_population(pop, target_image, fast_fitness=fast_fitness,
                                  target_downscaled=target_downscaled, fitness_mode=fitness_mode)
        islands.append({"population": pop, "fitness": fit, "rng": island_rng})

    best_individual = None
    best_fitness = float("inf")
    history = {"best_fitness": [], "avg_fitness": [], "worst_fitness": []}

    for gen in range(total_gens):
        gen_best = float("inf")
        gen_avg_sum = 0.0
        gen_worst = float("-inf")

        for island in islands:
            island["population"], island["fitness"] = _run_island_generation(
                island["population"], target_image, island["fitness"],
                island["rng"], gen, target_downscaled, **ga_kwargs)

            island_best = float(np.min(island["fitness"]))
            island_worst = float(np.max(island["fitness"]))
            island_avg = float(np.mean(island["fitness"]))

            gen_best = min(gen_best, island_best)
            gen_worst = max(gen_worst, island_worst)
            gen_avg_sum += island_avg

            if island_best < best_fitness:
                best_idx = np.argmin(island["fitness"])
                best_fitness = island_best
                best_individual = island["population"][best_idx].copy()

        history["best_fitness"].append(gen_best)
        history["avg_fitness"].append(gen_avg_sum / num_islands)
        history["worst_fitness"].append(gen_worst)

        # Migration every migration_interval generations
        if migration_interval > 0 and (gen + 1) % migration_interval == 0:
            for i in range(num_islands):
                src = islands[i]
                dst = islands[(i + 1) % num_islands]
                src_sorted = np.argsort(src["fitness"])
                dst_sorted = np.argsort(dst["fitness"])
                for m in range(min(migration_size, island_pop)):
                    dst["population"][dst_sorted[-(m + 1)]] = src["population"][src_sorted[m]].copy()
                    dst["fitness"][dst_sorted[-(m + 1)]] = src["fitness"][src_sorted[m]]

        print(f"Generation {gen + 1}/{total_gens} | "
              f"Best: {gen_best:.4f} | Avg: {gen_avg_sum / num_islands:.4f}")

    return best_individual, best_fitness, history
