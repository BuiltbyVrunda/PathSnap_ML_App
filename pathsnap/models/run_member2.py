"""Full Member 2 run: five-model comparison, Bayesian tuning, tuned-vs-untuned, final artifact.

Expected input layout (written by Member 1's pipeline; csv, trip_id kept as string):
    <data>/train/{candidates,observations,references}.csv
    <data>/student_test/{candidates,observations,references}.csv

    python -m pathsnap.models.run_member2 --data runs/dev_tables --out reports/model-experiments \
        --trials 25 --folds 5
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .schema import RANDOM_SEED
from .training.experiments import evaluate_scorer, fit_scorer, load_data, run_comparison, summary_row
from .tuning.bayes import SEARCH_SPACES, bayes_search


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="reports/model-experiments")
    ap.add_argument("--artifact", default="artifacts/models/pathsnap_scorer.joblib")
    ap.add_argument("--trials", type=int, default=25)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=RANDOM_SEED)
    a = ap.parse_args(argv)
    out = Path(a.out)
    train, test = load_data(Path(a.data) / "train"), load_data(Path(a.data) / "student_test")

    comp = run_comparison(train, test, out / "comparison", n_splits=a.folds, seed=a.seed)
    tunable = comp[comp["model"].isin(SEARCH_SPACES)].sort_values("cv_macro_f1_mean", ascending=False)
    chosen = tunable.iloc[0]["model"]  # selected on grouped-CV macro F1 only (never on student_test)
    search = bayes_search(chosen, train, a.trials, out / "tuning", a.folds, a.seed)

    untuned = fit_scorer(chosen, train, None, a.seed)
    tuned = fit_scorer(chosen, train, search["best_params"], a.seed)
    rows = []
    for tag, sc in (("untuned", untuned), ("tuned", tuned)):
        r = summary_row(f"{chosen}:{tag}", a.seed, None, sc, evaluate_scorer(sc, test)["metrics"])
        rows.append(r)
    pd.DataFrame(rows).to_csv(out / "tuned_vs_untuned.csv", index=False)
    # keep the tuned model only if it also wins on grouped CV (not on student_test, to avoid test-set selection)
    final = tuned if search["best_value"] >= float(tunable.iloc[0]["cv_macro_f1_mean"]) else untuned
    final.save(a.artifact)
    (out / "run_summary.json").write_text(json.dumps({"selected_model": chosen, "best_params": search["best_params"],
                                                      "artifact": a.artifact}, indent=2))
    print(f"selected {chosen}; artifact -> {a.artifact}")


if __name__ == "__main__":
    main()
