# PathSnap

Project structure for the four-member GPS map-matching and route-reconstruction team.

| Owner | Workspace | Handoff |
| --- | --- | --- |
| Member 1 | `pathsnap/data/` | [Data, candidate features, and reference alignment](handoffs/01-data-and-features.md) |
| Member 2 | `pathsnap/models/` | [Model experiments and Bayesian tuning](handoffs/02-models-and-tuning.md) |
| Member 3 | `pathsnap/decoder/` | [HMM/Viterbi and route evaluation](handoffs/03-decoder-and-route-evaluation.md) |
| Member 4 | `pathsnap/service/` | [FastAPI, evaluator UI, Folium, and release](handoffs/04-api-ui-and-release.md) |

```text
pathsnap/
  data/
    loading/
    candidates/
    reference_alignment/
  models/
    training/
    tuning/
    evaluation/
  decoder/
    viterbi/
    evaluation/
  service/
    api/
    ui/
    maps/
artifacts/
  models/
configs/
handoffs/
reports/
  eda/
  model-experiments/
  decoder-comparison/
  demo/
tests/
  integration/
```

The [team plan](PATHSNAP_TEAM_PLAN.md) defines the integration flow from GPS and road files through candidate generation, model scoring, route decoding, and the evaluator UI. Shared configuration goes in `configs/`, end-to-end checks in `tests/integration/`, and report evidence in `reports/`.

## Member 1 data pipeline

Use Python 3.12 and install `python -m pip install -r requirements-data.txt` for inference and reference scoring, or install `requirements-eda.txt` for the full audit and plots. Run the data integration tests with `python -m unittest discover -s tests/integration -v`. The [data handoff](handoffs/01-data-and-features.md#implementation-and-reproduction) explains the API and how to generate development tables, and [Section 2](reports/eda/section-2.md) records measured results and the reasons for the data-processing choices.

## Local data and model files

Keep the provided dataset at `Student folder/Student folder/` or under `data/`. The dataset directories, raw trip files, local experiment output under `runs/`, and organiser materials are ignored by Git.

Run `git lfs install` after cloning. Store fitted models in `artifacts/models/`. The repository tracks common model and checkpoint formats and ZIP files under `artifacts/` through Git LFS.
