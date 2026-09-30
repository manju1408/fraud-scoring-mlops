import json

from fastapi.testclient import TestClient

from fraud.features import OnlineFeatureStore
from fraud.serve import create_app


def test_model_quality(model_dir):
    _, m = model_dir
    assert m["roc_auc"] > 0.9
    assert m["pr_auc"] > 0.6


def test_artifacts_written(model_dir):
    out, _ = model_dir
    meta = json.loads((out / "meta.json").read_text())
    assert (out / "model.json").exists() and "features" in meta


def test_score_endpoint(model_dir, data):
    out, _ = model_dir
    app = create_app(out, OnlineFeatureStore().load_history(data))
    c = TestClient(app)
    normal = data[data.is_fraud == 0].iloc[-1]
    ok = c.post("/score", json={"txn_id": "n1", "customer_id": normal.customer_id,
                                "ts": "2026-02-01T14:00:00", "amount": float(normal.amount),
                                "country": normal.home_country, "device_id": normal.device_id,
                                "home_country": normal.home_country})
    bad = c.post("/score", json={"txn_id": "f1", "customer_id": normal.customer_id,
                                 "ts": "2026-02-01T03:00:00", "amount": float(normal.amount) * 15,
                                 "country": "RO", "device_id": "never-seen",
                                 "home_country": normal.home_country})
    assert ok.status_code == bad.status_code == 200
    assert bad.json()["score"] > ok.json()["score"]
    assert bad.json()["decision"] == "review" and "new device" in bad.json()["reasons"]


def test_rejects_invalid(model_dir):
    c = TestClient(create_app(model_dir[0]))
    r = c.post("/score", json={"txn_id": "x", "customer_id": "c", "ts": "2026-01-01",
                               "amount": -5, "country": "US", "device_id": "d", "home_country": "US"})
    assert r.status_code == 422
