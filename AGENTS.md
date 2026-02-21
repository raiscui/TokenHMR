# Repository Guidelines

## proxy
- 运行网络命令前要注册代理: `export https_proxy=http://127.0.0.1:7897 http_proxy=http://127.0.0.1:7897 all_proxy=socks5://127.0.0.1:7897`

## Project Structure & Module Organization

- `tokenhmr/`: TokenHMR training, evaluation, and demos.
  - `tokenhmr/lib/models/`: model components (e.g., `tokenhmr/lib/models/tokenhmr.py`).
  - `tokenhmr/lib/datasets/`: datasets + Lightning datamodule.
  - `tokenhmr/lib/configs_hydra/`: Hydra configs and experiments.
- `tokenization/`: pose tokenization (PoseVQ) training code and utilities.
  - `tokenization/configs/`: tokenizer configs (YAML).
  - `tokenization/scripts/`: dataset preparation helpers.
- `demo_sample/`: sample inputs for image/video demos.
- `assets/`: figures/GIFs used by the README.
- `data/`: downloaded body models + checkpoints (created by `fetch_demo_data.sh`, ignored by Git).

## Build, Test, and Development Commands

This is a research codebase (Python). Follow the README setup (Python 3.12, PyTorch + CUDA as needed).
When building Detectron2/PHALP, ensure `CUDA_HOME` matches the CUDA version used by your PyTorch install.

```bash
# Install deps (Pixi reads pixi.toml)
pixi install                  # single env only (CUDA 12.6)

# Download demo assets (interactive prompt)
pixi run fetch_demo_data

# Smoke check
pixi run smoke_compileall

# Demos (require demo deps)
pixi run demo_image
pixi run demo_video

# Detectron2 builds CUDA extensions; ensure CUDA toolkit matches PyTorch (cu126).

# Training / evaluation (pick the environment you want)
pixi run train_tokenizer
pixi run train_tokenhmr
pixi run eval
```

## Coding Style & Naming Conventions

- Python: 4-space indentation, keep changes consistent with nearby code.
- Naming: `snake_case` for functions/variables, `CamelCase` for classes, `UPPER_SNAKE_CASE` for constants.
- Config changes: prefer editing Hydra YAMLs under `tokenhmr/lib/configs_hydra/` and tokenizer YAMLs under `tokenization/configs/`.

## Testing Guidelines

No dedicated unit test suite is included. For PRs, do at least a smoke check:

```bash
pixi run smoke_compileall
```

If you have the required assets/models, also run an image demo on `demo_sample/images/`.

## Commit & Pull Request Guidelines

- Commits: keep them small and descriptive; existing history uses short messages, sometimes with a scope like `[Tokenization] ...`.
- PRs: include a clear summary, how to reproduce/verify (commands + expected output), and screenshots/GIFs for demo or visualization changes.
- Do not commit large artifacts: datasets, `data/`, and checkpoints are intentionally ignored by `.gitignore`.

## Licensing & Data Notes

The repository is released for non-commercial scientific research (see `LICENSE`). Downloaded models/datasets may have additional license terms; do not redistribute them in PRs.
