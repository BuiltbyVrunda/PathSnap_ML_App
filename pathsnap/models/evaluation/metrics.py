"""Development scoring policy (shared contract): Top-1, candidate miss rate, macro F1, log loss.

Definitions
-----------
* Eligible observations = reference rows with alignment_status == "aligned". Ambiguous and
  unaligned rows are counted and reported but never scored.
* Top-1 accuracy = (# aligned obs whose predicted arc == reference arc) / (# aligned obs).
  Candidate misses, empty candidate sets and null predictions count as incorrect.
* Candidate miss rate = (# aligned obs whose reference arc is not in the candidate set) / (# aligned).
* Macro F1 / precision / recall: classes are trip-qualified arcs (trip_id, arc_id). Class set =
  union of reference arcs and predicted arcs over aligned observations; the unmatched sentinel
  (null prediction) is not a class, but a null prediction is a false negative for the true class.
  A predicted-only arc has F1 = 0, so false positives are penalised.
* Log loss = mean(-log(max(reference emission, EPSILON))) over aligned observations whose
  reference arc is in the candidate set ("covered" subset); size and coverage are reported.
* A metric with zero eligible observations is None (unavailable).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..schema import ALIGNED, AMBIGUOUS, EPSILON, OBS_KEY, UNALIGNED


def _aligned_frame(references: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    ref = references[references["alignment_status"] == ALIGNED][OBS_KEY + ["ref_arc_id"]]
    pred = predictions[OBS_KEY + ["arc_id"]].rename(columns={"arc_id": "pred_arc_id"})
    ref = ref.assign(trip_id=ref["trip_id"].astype(str))
    pred = pred.assign(trip_id=pred["trip_id"].astype(str), obs_idx=pred["obs_idx"].astype("int64"))
    df = ref.merge(pred, on=OBS_KEY, how="left")  # absent prediction row == null prediction
    df["ref_arc_id"] = df["ref_arc_id"].astype("int64")
    df["pred_arc_id"] = df["pred_arc_id"].astype("Int64")
    return df


def macro_prf(df: pd.DataFrame) -> dict:
    """Macro precision/recall/F1 over trip-qualified arcs from an aligned frame."""
    if df.empty:
        return {"macro_f1": None, "macro_precision": None, "macro_recall": None, "n_classes": 0}
    true = df["trip_id"].astype(str) + ":" + df["ref_arc_id"].astype(str)
    has = df["pred_arc_id"].notna()
    pred = pd.Series(np.where(has, df["trip_id"].astype(str) + ":" + df["pred_arc_id"].astype(str), None),
                     index=df.index, dtype="object")
    correct = has & (true == pred)
    tp = correct.groupby(true).sum()
    n_true = true.value_counts()
    n_pred = pred.dropna().value_counts()
    classes = sorted(set(n_true.index) | set(n_pred.index))
    tp = tp.reindex(classes, fill_value=0).astype(float)
    nt = n_true.reindex(classes, fill_value=0).astype(float)
    npd = n_pred.reindex(classes, fill_value=0).astype(float)
    prec = np.divide(tp, npd, out=np.zeros_like(tp), where=npd > 0)
    rec = np.divide(tp, nt, out=np.zeros_like(tp), where=nt > 0)
    denom = prec + rec
    f1 = np.divide(2 * prec * rec, denom, out=np.zeros_like(denom), where=denom > 0)
    return {"macro_f1": float(np.mean(f1)), "macro_precision": float(np.mean(prec)),
            "macro_recall": float(np.mean(rec)), "n_classes": len(classes)}


def evaluate(references: pd.DataFrame, predictions: pd.DataFrame,
             candidates: pd.DataFrame | None = None,
             scored: pd.DataFrame | None = None) -> dict:
    """Score predictions against the development reference table.

    references: one row per observation (Reference record).
    predictions: ClassifierMatch rows (missing rows are treated as null predictions).
    candidates: Candidate/Scored rows, needed for miss rate and log loss.
    scored: ScoredCandidate rows (with emission_prob), needed for log loss.
    """
    status = references["alignment_status"].value_counts()
    out: dict = {
        "n_observations": int(len(references)),
        "n_aligned": int(status.get(ALIGNED, 0)),
        "n_ambiguous": int(status.get(AMBIGUOUS, 0)),
        "n_unaligned": int(status.get(UNALIGNED, 0)),
    }
    df = _aligned_frame(references, predictions)
    n = len(df)
    if n == 0:
        out.update(top1_accuracy=None, n_correct=0, candidate_miss_rate=None, n_candidate_miss=None,
                   macro_f1=None, macro_precision=None, macro_recall=None, n_classes=0,
                   log_loss=None, log_loss_n=0, log_loss_coverage=None, n_null_predictions=0)
        return out
    correct = df["pred_arc_id"].notna() & (df["pred_arc_id"].astype("float64") == df["ref_arc_id"])
    out["n_correct"] = int(correct.sum())
    out["top1_accuracy"] = float(correct.mean())
    out["n_null_predictions"] = int(df["pred_arc_id"].isna().sum())
    out.update(macro_prf(df))

    if candidates is not None:
        ckeys = candidates[OBS_KEY + ["arc_id"]].rename(columns={"arc_id": "ref_arc_id"})
        ckeys = ckeys.assign(trip_id=ckeys["trip_id"].astype(str))
        j = df.merge(ckeys, on=OBS_KEY + ["ref_arc_id"], how="left", indicator=True)
        covered = j["_merge"] == "both"
        out["n_candidate_miss"] = int((~covered).sum())
        out["candidate_miss_rate"] = float((~covered).mean())
        if scored is not None:
            s = scored[OBS_KEY + ["arc_id", "emission_prob"]].rename(columns={"arc_id": "ref_arc_id"})
            s = s.assign(trip_id=s["trip_id"].astype(str))
            ll = df.merge(s, on=OBS_KEY + ["ref_arc_id"], how="inner")
            out["log_loss_n"] = int(len(ll))
            out["log_loss_coverage"] = float(len(ll) / n)
            out["log_loss"] = (float(np.mean(-np.log(np.maximum(ll["emission_prob"], EPSILON))))
                               if len(ll) else None)
    return out
