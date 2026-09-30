"""One feature definition shared by training and serving.

Training builds customer context point-in-time (only transactions strictly before
each event), and serving builds the same context from an online profile, then both
call ``derive`` so training and inference always see the same features.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

FEATURES = ["log_amount", "amount_to_avg", "prior_txn_count", "txn_last_1h",
            "is_night", "new_device", "foreign"]


def derive(ctx: pd.DataFrame) -> pd.DataFrame:
    """ctx columns: amount, cust_avg_amount, prior_txn_count, txn_last_1h, hour,
    device_seen, country, home_country."""
    avg = ctx["cust_avg_amount"].fillna(ctx["amount"])
    return pd.DataFrame({
        "log_amount": np.log1p(ctx["amount"]),
        "amount_to_avg": ctx["amount"] / avg.clip(lower=1.0),
        "prior_txn_count": ctx["prior_txn_count"].astype(float),
        "txn_last_1h": ctx["txn_last_1h"].astype(float),
        "is_night": ctx["hour"].between(0, 4).astype(float),
        "new_device": (~ctx["device_seen"].astype(bool)).astype(float),
        "foreign": (ctx["country"] != ctx["home_country"]).astype(float),
    }, index=ctx.index)[FEATURES]


def point_in_time_context(df: pd.DataFrame) -> pd.DataFrame:
    """Customer context computed from strictly earlier transactions (no leakage)."""
    d = df.sort_values(["customer_id", "ts"]).copy()
    g = d.groupby("customer_id", sort=False)
    d["prior_txn_count"] = g.cumcount()
    d["cust_avg_amount"] = g["amount"].transform(lambda s: s.shift().expanding().mean())
    d["device_seen"] = d.duplicated(["customer_id", "device_id"])
    ts_ns = d["ts"].to_numpy().astype("datetime64[ns]").astype("int64")
    window = 3_600 * 10**9
    last_1h = np.zeros(len(d), dtype=int)
    for idx in g.indices.values():  # positional indices per customer, already time-ordered
        t = ts_ns[idx]
        last_1h[idx] = np.arange(len(t)) - np.searchsorted(t, t - window, "left")
    d["txn_last_1h"] = last_1h
    d["hour"] = d["ts"].dt.hour
    return d.sort_index()


@dataclass
class CustomerProfile:
    total_amount: float = 0.0
    count: int = 0
    devices: set = field(default_factory=set)
    recent_ts: list = field(default_factory=list)


class OnlineFeatureStore:
    """In-memory stand-in for an online feature store (e.g. Redis / Databricks)."""

    def __init__(self):
        self.profiles: dict[str, CustomerProfile] = {}

    def update(self, customer_id: str, ts: pd.Timestamp, amount: float, device_id: str) -> None:
        p = self.profiles.setdefault(customer_id, CustomerProfile())
        p.total_amount += amount
        p.count += 1
        p.devices.add(device_id)
        p.recent_ts = [t for t in p.recent_ts if ts - t < pd.Timedelta(hours=1)] + [ts]

    def load_history(self, df: pd.DataFrame) -> "OnlineFeatureStore":
        for r in df.sort_values("ts").itertuples():
            self.update(r.customer_id, r.ts, r.amount, r.device_id)
        return self

    def context(self, txn: dict) -> pd.DataFrame:
        p = self.profiles.get(txn["customer_id"], CustomerProfile())
        ts = pd.Timestamp(txn["ts"])
        return pd.DataFrame([{
            "amount": txn["amount"],
            "cust_avg_amount": p.total_amount / p.count if p.count else np.nan,
            "prior_txn_count": p.count,
            "txn_last_1h": sum(1 for t in p.recent_ts if ts - t < pd.Timedelta(hours=1)),
            "hour": ts.hour,
            "device_seen": txn["device_id"] in p.devices,
            "country": txn["country"],
            "home_country": txn["home_country"],
        }])
