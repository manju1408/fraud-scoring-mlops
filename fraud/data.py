"""Synthetic card/account transactions with injected fraud patterns. No real data."""
from __future__ import annotations

import numpy as np
import pandas as pd

COUNTRIES = ["US", "US", "US", "US", "CA", "MX", "GB", "NG", "RO"]


def generate(n_customers: int = 800, days: int = 30, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    start = pd.Timestamp("2026-01-01")
    for cid in range(n_customers):
        home = "US" if rng.random() < 0.85 else rng.choice(["CA", "GB"])
        base = rng.lognormal(3.5, 0.6)
        devices = [f"d{cid}-{i}" for i in range(rng.integers(1, 3))]
        n = rng.poisson(days * 1.5) + 1
        ts = np.sort(start + pd.to_timedelta(rng.uniform(0, days * 86400, n), unit="s"))
        for t in map(pd.Timestamp, ts):
            amount, country, device = float(rng.lognormal(np.log(base), 0.4)), home, str(rng.choice(devices))
            r = rng.random()
            if r < 0.03:  # legit travel
                country = str(rng.choice(COUNTRIES[4:]))
            elif r < 0.05:  # legit big purchase
                amount *= rng.uniform(3, 8)
            elif r < 0.065:  # legit new phone / browser
                device = f"alt-{cid}-{rng.integers(99)}"
            if rng.random() < 0.04:  # late-night legit activity
                t = t.replace(hour=int(rng.integers(0, 5)))
            rows.append((f"c{cid:05d}", t, round(amount, 2), country, device, home, 0))
        if rng.random() < 0.08:  # account takeover: burst on a new device, often foreign
            t0 = start + pd.Timedelta(seconds=float(rng.uniform(days * 43200, days * 86400)))
            if rng.random() < 0.6:
                t0 = t0.replace(hour=int(rng.integers(0, 5)))
            foreign = rng.random() < 0.6
            for k in range(int(rng.integers(1, 5))):
                rows.append((f"c{cid:05d}", t0 + pd.Timedelta(minutes=float(rng.uniform(3, 40)) * k),
                             round(float(base * rng.uniform(0.8, 10)), 2),
                             str(rng.choice(COUNTRIES[5:])) if foreign else home,
                             f"new-{cid}-{k % 2}", home, 1))
    df = pd.DataFrame(rows, columns=["customer_id", "ts", "amount", "country", "device_id",
                                     "home_country", "is_fraud"])
    df = df.sort_values(["ts", "customer_id"]).reset_index(drop=True)
    df.insert(0, "txn_id", [f"t{i:07d}" for i in range(len(df))])
    return df
