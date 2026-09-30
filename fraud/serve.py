"""Real-time scoring API. Loads the trained model and an online feature store.

MODEL_DIR=artifacts uvicorn fraud.serve:app
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field
from xgboost import XGBClassifier

from .features import OnlineFeatureStore, derive


class Txn(BaseModel):
    txn_id: str
    customer_id: str
    ts: str
    amount: float = Field(gt=0)
    country: str = Field(min_length=2, max_length=2)
    device_id: str
    home_country: str = Field(min_length=2, max_length=2)


class Score(BaseModel):
    txn_id: str
    score: float
    decision: str
    reasons: list[str]
    latency_ms: float


def reasons(f) -> list[str]:
    rules = [
        (f["amount_to_avg"] > 3, "amount far above customer average"),
        (f["new_device"] > 0, "new device"),
        (f["foreign"] > 0, "foreign country"),
        (f["txn_last_1h"] >= 2, "high velocity"),
        (f["is_night"] > 0, "night-time"),
    ]
    return [msg for hit, msg in rules if hit]


def create_app(model_dir: str | Path, store: OnlineFeatureStore | None = None) -> FastAPI:
    model_dir = Path(model_dir)
    meta = json.loads((model_dir / "meta.json").read_text())
    model = XGBClassifier()
    model.load_model(model_dir / "model.json")
    store = store or OnlineFeatureStore()
    app = FastAPI(title="fraud-scoring")

    @app.get("/health")
    def health():
        return {"status": "ok", "threshold": meta["threshold"], "customers": len(store.profiles)}

    @app.post("/score", response_model=Score)
    def score(txn: Txn):
        t0 = time.perf_counter()
        feats = derive(store.context(txn.model_dump()))
        p = float(model.predict_proba(feats[meta["features"]])[0, 1])
        decision = "review" if p >= meta["threshold"] else "approve"
        store.update(txn.customer_id, pd.Timestamp(txn.ts), txn.amount, txn.device_id)
        return Score(txn_id=txn.txn_id, score=round(p, 4), decision=decision,
                     reasons=reasons(feats.iloc[0]) if decision == "review" else [],
                     latency_ms=round((time.perf_counter() - t0) * 1000, 2))

    return app


if os.getenv("MODEL_DIR"):  # pragma: no cover
    app = create_app(os.environ["MODEL_DIR"])
