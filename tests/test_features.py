import pandas as pd

from fraud.features import FEATURES, OnlineFeatureStore, derive, point_in_time_context


def small():
    ts = pd.to_datetime(["2026-01-01 10:00", "2026-01-01 10:20", "2026-01-01 12:00",
                         "2026-01-02 02:00"])
    return pd.DataFrame({"txn_id": list("abcd"), "customer_id": ["c1"] * 4, "ts": ts,
                         "amount": [10.0, 20.0, 30.0, 400.0], "country": ["US", "US", "US", "RO"],
                         "device_id": ["d1", "d1", "d1", "x9"], "home_country": ["US"] * 4,
                         "is_fraud": [0, 0, 0, 1]})


def test_point_in_time_has_no_leakage():
    ctx = point_in_time_context(small())
    assert ctx["prior_txn_count"].tolist() == [0, 1, 2, 3]
    assert ctx["cust_avg_amount"].iloc[1] == 10.0  # only the first txn is visible
    assert ctx["cust_avg_amount"].iloc[3] == 20.0
    assert ctx["txn_last_1h"].tolist() == [0, 1, 0, 0]
    assert ctx["device_seen"].tolist() == [False, True, True, False]


def test_online_matches_offline_features():
    df = small()
    offline = derive(point_in_time_context(df)).iloc[-1]
    store = OnlineFeatureStore().load_history(df.iloc[:-1])
    online = derive(store.context(df.iloc[-1].to_dict())).iloc[0]
    pd.testing.assert_series_equal(offline[FEATURES], online[FEATURES], check_names=False)
