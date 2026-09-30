"""Train an XGBoost fraud model, pick a threshold for a target precision, and save
the model, feature list and metrics. Logs to MLflow when MLFLOW_TRACKING_URI is set.

python -m fraud.train --out artifacts
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from xgboost import XGBClassifier

from .data import generate
from .features import FEATURES, derive, point_in_time_context

PARAMS = dict(n_estimators=200, max_depth=4, learning_rate=0.08, subsample=0.9,
              colsample_bytree=0.9, eval_metric="aucpr", random_state=7)


def time_split(df: pd.DataFrame, frac: float = 0.8):
    cut = df["ts"].quantile(frac)
    return df[df["ts"] <= cut], df[df["ts"] > cut]


def threshold_for_precision(y, p, target: float) -> float:
    prec, _, thr = precision_recall_curve(y, p)
    ok = np.where(prec[:-1] >= target)[0]
    return float(thr[ok[0]]) if len(ok) else 0.5


def train(df: pd.DataFrame | None = None, out: str | Path = "artifacts",
          target_precision: float = 0.9) -> dict:
    df = generate() if df is None else df
    ctx = point_in_time_context(df)
    train_df, test_df = time_split(ctx)
    X_tr, y_tr = derive(train_df), train_df["is_fraud"]
    X_te, y_te = derive(test_df), test_df["is_fraud"]
    pos_weight = (len(y_tr) - y_tr.sum()) / max(y_tr.sum(), 1)
    model = XGBClassifier(**PARAMS, scale_pos_weight=pos_weight).fit(X_tr, y_tr)
    p = model.predict_proba(X_te)[:, 1]
    thr = threshold_for_precision(y_te, p, target_precision)
    pred = p >= thr
    metrics = {
        "roc_auc": round(float(roc_auc_score(y_te, p)), 4),
        "pr_auc": round(float(average_precision_score(y_te, p)), 4),
        "threshold": round(thr, 4),
        "precision_at_thr": round(float((y_te[pred] == 1).mean()) if pred.any() else 0.0, 4),
        "recall_at_thr": round(float(pred[y_te == 1].mean()), 4),
        "fraud_rate_test": round(float(y_te.mean()), 4),
    }
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    model.save_model(out / "model.json")
    (out / "meta.json").write_text(json.dumps({"features": FEATURES, **metrics}, indent=2))
    _log_mlflow(metrics, out)
    return metrics


def _log_mlflow(metrics: dict, out: Path) -> None:
    if not os.getenv("MLFLOW_TRACKING_URI"):
        return
    import mlflow  # tracking URI is read from the environment

    mlflow.set_experiment("fraud-scoring")
    with mlflow.start_run():
        mlflow.log_params(PARAMS)
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(out / "model.json"))
        mlflow.log_artifact(str(out / "meta.json"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts")
    ap.add_argument("--target-precision", type=float, default=0.9)
    a = ap.parse_args()
    print(json.dumps(train(out=a.out, target_precision=a.target_precision), indent=2))
