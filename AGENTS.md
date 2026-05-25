# Repository Guidelines

## Project Structure & Module Organization

This repository is a Python 3.11 framework for LLM unlearning experiments. Core code lives in `src/`: `src/train.py` and `src/eval.py` are the main Hydra entry points, `src/trainer/unlearn/` contains unlearning methods, `src/evals/` contains benchmarks and metrics, `src/data/` contains datasets and collators, and `src/model/` contains model utilities. Hydra config groups live in `configs/` (`experiment`, `trainer`, `data`, `model`, `eval`, etc.). Use `scripts/` for reproducible experiment commands, `docs/` for longer guides, `community/` for contributed methods or benchmarks, and `assets/` for documentation media. Treat `results/`, `report/`, notebooks, and checkpoint-like files as generated or exploratory artifacts unless explicitly documenting a result.

## Build, Test, and Development Commands

- `pip install -e ".[dev]"`: install editable package plus development tools.
- `pip install -e ".[lm-eval]"`: include lm-evaluation-harness support.
- `python setup_data.py --eval`: download evaluation logs into `saves/eval`.
- `make quality`: run Ruff linting and format checks; this is the active CI gate.
- `make style`: apply Ruff fixes and formatting.
- `make test`: run `CUDA_VISIBLE_DEVICES= pytest tests/`; add tests under `tests/` before relying on this.

See `README.md` for full train/eval examples using `python src/train.py --config-name=unlearn.yaml ...` and `python src/eval.py --config-name=eval.yaml ...`.

## Coding Style & Naming Conventions

Use 4-space indentation, Python type-friendly code, and Ruff-compatible formatting. Name modules, functions, and variables in `snake_case`; name classes in `PascalCase`. Keep Hydra component names aligned with implementation and config files, for example `src/trainer/unlearn/grad_ascent.py` with `configs/trainer/GradAscent.yaml`. Prefer small helpers in existing package areas over new top-level scripts.

## Testing Guidelines

Use `pytest` for new automated tests, with files named `tests/test_*.py`. Keep tests lightweight: mock model downloads, use tiny fixtures, and isolate metrics, data loading, config composition, or trainer logic. For experiment scripts, include a smoke command and expected output location. Run `make quality` and focused tests before opening a PR.

## Commit & Pull Request Guidelines

Recent local commits are short and informal (for example `new exp`, `sku and flat`), but contributor-facing commits should be imperative and descriptive, such as `Add SatImp trainer config`. PRs should summarize the change, link issues, list reproduction or evaluation commands, mention hardware/GPU assumptions, and include screenshots or result tables for docs or benchmark updates. Prefix unfinished PRs with `[WIP]`.

## Security & Configuration Tips

Do not commit Hugging Face tokens, private dataset paths, model checkpoints, or large generated logs. Put machine-specific paths in local overrides or `configs/paths/`, and document any required environment variables in the PR.
