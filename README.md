# Girl with a Pearl Earring - Genetic Triangle Approximation

A genetic algorithm that recreates Vermeer's *Girl with a Pearl Earring* using 100 semi-transparent triangles. The algorithm evolves triangle positions, sizes, and colors to approximate the target image through selection, crossover, mutation, and elitism.

## Overview

This project demonstrates evolutionary image approximation by encoding an image as a collection of layered polygons. Each individual is scored against a target image using RMSE fitness, and the population evolves over generations to minimize visual difference.

**Key features:**
- Triangle-based representation (100 RGBA polygons per individual)
- Multiple selection, crossover, and mutation strategies
- Optional local search, island models, and diversity controls
- Configurable via YAML or command-line arguments

## Installation

```bash
git clone <repository-url>
cd project

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Requires Python 3.10+

## Quick Start

Run a quick test:

```bash
python3 run.py --pop-size 30 --generations 25 --run-id test
```

Run with default settings:

```bash
python3 run.py --config experiments/configs/default.yaml --run-id baseline
```

Run with custom parameters:

```bash
python3 run.py \
  --pop-size 100 \
  --generations 5000 \
  --selection-method rank \
  --mutation-rate 0.05 \
  --elite-size 2 \
  --seed 42 \
  --run-id custom
```

**Outputs:**
- `data/snapshots/ga_result.png` — final image
- `experiments/logs/<run-id>.csv` — convergence history

## Parameters

Configure via command-line flags or YAML config file:

- `pop_size` — population size (default: 150)
- `generations` — number of generations (default: 5000)
- `selection_method` — `tournament`, `roulette`, or `rank` (default: `rank`)
- `crossover_method` — `uniform`, `single_point`, `blend`, etc.
- `mutation_method` — `gaussian`, `adaptive`, `multi`, etc.
- `mutation_rate` — probability of mutation per gene (default: 0.05)
- `sigma` — Gaussian noise scale (default: 10.0)
- `elite_size` — number of best individuals to preserve (default: 2)
- `seed` — random seed for reproducibility

See `experiments/configs/default.yaml` for full configuration options.

## Project Structure

```
src/
  ga.py                 # Main genetic algorithm loop
  fitness.py            # Fitness evaluation
  rendering.py          # Image rendering from triangles
  selection.py          # Parent selection methods
  crossover.py          # Crossover operators
  mutation.py           # Mutation operators
  diversity.py          # Diversity mechanisms
  representation.py     # Individual representation
  challenge.py          # Problem definition

experiments/
  sweep.py              # Parameter sweep experiments
  random_search.py      # Random hyperparameter search
  configs/              # Configuration files

notebooks/
  01_eda_target.ipynb            # Exploratory data analysis
  03_parameter_sweep_analysis.ipynb   # Sweep results
  08_final_results.ipynb         # Final comparison
```

## Results

The algorithm successfully reconstructs the target image using only 100 triangles. Results improve significantly with more generations and optimized parameters. Pre-computed results and convergence logs are available in `data/` and `experiments/logs/`.
selection_method: rank
crossover_method: uniform
crossover_rate: 0.8
mutation_method: multi
mutation_rate: 0.05
sigma: 10.0
elite_size: 2
seed: 42
snapshot_interval: 500
fitness_mode: rmse
local_search_iterations: 5
```

Command-line arguments override values loaded from a config file.

Useful options:

- `--fast-fitness` evaluates downscaled renders for faster iteration.
- `--fitness-mode {rmse,hybrid,perceptual}` switches the scoring function.
- `--island-model --num-islands 4` enables multiple sub-populations.
- `--fitness-sharing` encourages population diversity.
- `--restricted-mating` pairs parents using partial-match similarity.
- `--local-search-iterations N` applies hill climbing to elites.

See all available options:

```bash
python3 run.py --help
```

## How It Works

Each individual is a NumPy array with shape `(100, 10)`. Each row encodes one
triangle:

| Genes | Meaning |
| --- | --- |
| `x1, y1, x2, y2, x3, y3` | Triangle vertices on a 300 x 400 canvas |
| `r, g, b` | Fill color channels in `[0, 255]` |
| `alpha` | Opacity in `[0.1, 0.8]` |

The genetic algorithm repeatedly:

1. Renders each individual as layered translucent triangles.
2. Computes fitness against `data/target.png`.
3. Selects parents using the configured selection method.
4. Applies crossover and mutation to produce children.
5. Preserves the best individuals through elitism.
6. Records best, average, and worst fitness per generation.

Lower fitness is better.

## Experiments

Run one-factor parameter sweeps:

```bash
python3 experiments/sweep.py
```

Run random hyperparameter search:

```bash
python3 experiments/random_search.py --n-configs 20 --seeds 42 123
```

Run the ablation study comparing GA components:

```bash
python3 -m src.challenge
```

These experiments can be long-running. Results are saved under
`experiments/logs/` and `experiments/figures/`.

## Project Layout

```text
.
|-- data/                  Target image, generated snapshots, saved arrays
|-- experiments/           Sweep scripts, random search, logs, and figures
|-- notebooks/             Exploratory analysis and result notebooks
|-- report/                Exported project report
|-- src/
|   |-- challenge.py       Ablation study helpers
|   |-- crossover.py       Crossover operators
|   |-- diversity.py       Fitness sharing and restricted mating
|   |-- fitness.py         Target loading and fitness functions
|   |-- ga.py              Main GA and island-model loops
|   |-- mutation.py        Mutation operators
|   |-- rendering.py       Triangle rasterization
|   |-- representation.py  Genome constants and random initialization
|   `-- selection.py       Parent selection operators
|-- run.py                 Command-line entry point
|-- requirements.txt       Runtime dependencies
`-- LICENSE                MIT license
```

## Reproducibility

Most commands accept `--seed` for deterministic runs. Reproducibility can still
depend on Python, NumPy, and dependency versions, so keep `requirements.txt`
fixed when comparing runs.

For a lightweight code sanity check:

```bash
python3 -m compileall run.py src experiments
```

## Contributing

Contributions are welcome. Useful areas include faster rendering, richer fitness
metrics, better experiment tooling, cleaner notebooks, and additional benchmark
targets.

Before opening a pull request:

- Keep changes focused and documented.
- Include a small run or compile check when possible.
- Avoid committing large generated artifacts unless they are needed to explain
  a benchmark or result.

## License

This project is released under the [MIT License](LICENSE).

The painting itself is in the public domain. Image-file reuse rights can depend
on the reproduction source, so verify asset terms before redistributing modified
image datasets outside this repository.
