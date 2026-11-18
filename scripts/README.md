# Pipeline scripts

Run in order from the repo root:

1. `build_dataset.py` — labels, candidates, features -> `data/processed/`
2. `train_models.py` — train and compare >= 5 models -> `experiments/logs/`
3. `tune_optuna.py` — Bayesian optimisation of the chosen model -> `experiments/optuna/`
4. `evaluate_hmm.py` — classifier-only vs classifier + HMM/Viterbi
5. `export_artifacts.py` — save final model and configs -> `models/artifacts/`
