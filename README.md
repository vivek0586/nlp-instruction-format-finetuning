# Supervised fine-tuning experiment

`experiment.py` fine-tunes `Qwen/Qwen2.5-0.5B` with LoRA on the six-class Emotion dataset. It supports three modes:

- `baseline` evaluates the unchanged model.
- `train` trains an adapter, then evaluates the adapter and unchanged model on Emotion test and SST-2 validation using the selected prompt format.
- `cross-eval` loads an existing adapter and evaluates it with the other prompt format. It does not train again.

Use `--format natural` or `--format compact` and set `--seed` for each run. For crossed evaluation, pass the saved adapter directory with `--adapter-path`. The runner writes metrics, predictions, confusion matrices, and paired error analyses under `results/<format>_seed<seed>/`.

`analyze_results.py` checks the original prediction bundle and writes `results_audit.json`. `analyze_seed_comparisons.py` computes format contrasts, class gains and losses, and matched valid-output comparisons. `audit_cross_format.py` checks the crossed prediction bundle against the original results and writes `cross_format_audit.json`.

The analysis scripts expect `colab_results.zip` and, for crossed evaluation, `cross_eval_results.zip` beside the scripts. These bundles are excluded because their per-example predictions contain dataset text. Install the libraries listed in `requirements.txt` in a CUDA-enabled environment for model runs.
