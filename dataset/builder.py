import os, logging
import numpy as np
import pandas as pd
from telemetry.metrics import compute_derived_features

log = logging.getLogger(__name__)
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_telemetry(data_dir=None):
    if data_dir is None:
        data_dir = os.path.join(BASE, "data")
    parquet_path = os.path.join(data_dir, "telemetry.parquet")
    csv_path = os.path.join(data_dir, "telemetry.csv")
    if os.path.exists(parquet_path):
        return pd.read_parquet(parquet_path)
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    raise FileNotFoundError(f"Telemetry not found in {data_dir}")


def label_conflicts(df, kpi_threshold=0.05, lookback=10):
    out = df.copy()


    action_cols = {
        "ts": ["ts_handover", "ts_carrier_agg"],
        "qos": ["qos_prb_reservation", "qos_power_boost", "qos_mcs_override", "qos_sched_weight"],
        "mlb": ["mlb_cio_adjust", "mlb_prb_cap", "mlb_ho_threshold"],
        "es": ["es_mimo_reduce", "es_dtx_drx", "es_carrier_shutdown"],
    }


    all_action_cols = []
    for xapp, cols in action_cols.items():
        for c in cols:
            if c in out.columns:

                out[f"{c}_w"] = out[c]
                all_action_cols.append(c)


    ho_conflict = pd.Series(False, index=out.index)
    if "ts_handover_w" in out.columns:
        for mc in ["mlb_cio_adjust_w", "mlb_ho_threshold_w"]:
            if mc in out.columns:
                ho_conflict |= (out["ts_handover_w"] > 0) & (out[mc] > 0)


    prb_conflict = pd.Series(False, index=out.index)
    if "qos_prb_reservation_w" in out.columns and "mlb_prb_cap_w" in out.columns:

        q_med = out["qos_prb_reservation_w"].median()
        m_med = out["mlb_prb_cap_w"].median()
        prb_conflict = (out["qos_prb_reservation_w"] > q_med) & (out["mlb_prb_cap_w"] > m_med)


    power_conflict = pd.Series(False, index=out.index)
    if "qos_power_boost_w" in out.columns and "es_mimo_reduce_w" in out.columns:
        power_conflict = (out["qos_power_boost_w"] > 0) & (out["es_mimo_reduce_w"] > 0)


    carrier_conflict = pd.Series(False, index=out.index)
    if "ts_carrier_agg_w" in out.columns and "es_carrier_shutdown_w" in out.columns:
        carrier_conflict = (out["ts_carrier_agg_w"] > 0) & (out["es_carrier_shutdown_w"] > 0)


    sched_conflict = pd.Series(False, index=out.index)
    if "qos_sched_weight_w" in out.columns and "mlb_prb_cap_w" in out.columns:
        q_med = out["qos_sched_weight_w"].quantile(0.75)
        m_med = out["mlb_prb_cap_w"].quantile(0.75)
        sched_conflict = (out["qos_sched_weight_w"] > q_med) & (out["mlb_prb_cap_w"] > m_med)

    action_conflict = ho_conflict | prb_conflict | power_conflict | carrier_conflict | sched_conflict


    kpi_deterioration = pd.Series(False, index=out.index)

    thp_ma = [c for c in out.columns if c.startswith("DRB.UEThpDl_ma")]
    if thp_ma and "DRB.UEThpDl" in out.columns:
        baseline = out[thp_ma[0]].clip(lower=1)
        drop = (baseline - out["DRB.UEThpDl"]) / baseline
        kpi_deterioration |= drop > 0.20

    delay_ma = [c for c in out.columns if c.startswith("DRB.RlcSduDelayDl_ma")]
    if delay_ma and "DRB.RlcSduDelayDl" in out.columns:
        baseline = out[delay_ma[0]].clip(lower=0.1)
        spike = (out["DRB.RlcSduDelayDl"] - baseline) / baseline
        kpi_deterioration |= spike > 0.50

    bler_ma = [c for c in out.columns if c.startswith("avg_bler_ma")]
    if bler_ma and "avg_bler" in out.columns:
        kpi_deterioration |= out["avg_bler"] > out[bler_ma[0]] * 3.0


    direct_conflict_w = pd.Series(False, index=out.index)

    out["conflict"] = (action_conflict | kpi_deterioration | direct_conflict_w).astype(int)


    severity = (ho_conflict.astype(float) + prb_conflict.astype(float) +
                power_conflict.astype(float) + carrier_conflict.astype(float) +
                sched_conflict.astype(float))
    if thp_ma and "DRB.UEThpDl" in out.columns:
        baseline = out[thp_ma[0]].clip(lower=1)
        severity += ((baseline - out["DRB.UEThpDl"]) / baseline).clip(lower=0)
    out["conflict_severity"] = severity.clip(lower=0)


    out["conflict_ho"] = ho_conflict.astype(int)
    out["conflict_prb"] = prb_conflict.astype(int)
    out["conflict_power"] = power_conflict.astype(int)
    out["conflict_carrier"] = carrier_conflict.astype(int)
    out["conflict_sched"] = sched_conflict.astype(int)


    w_cols = [c for c in out.columns if c.endswith("_w")]
    out.drop(columns=w_cols, inplace=True)

    return out


def get_feature_columns(df):
    exclude = {"timestamp", "step", "cell_id", "pci", "band",
               "conflict", "conflict_severity", "conflict_ho",
               "conflict_prb", "conflict_power", "conflict_carrier",
               "conflict_sched"}
    cols = []
    for c in df.columns:
        if c in exclude:
            continue
        if df[c].dtype in [np.float64, np.float32, np.int64, np.int32, float, int]:
            if not df[c].isna().all():
                cols.append(c)
    return cols


def train_test_split_dataset(df, test_size=0.2, seed=42):
    n = len(df)
    split_idx = int(n * (1 - test_size))
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()


def build_dataset(cfg=None):
    import yaml
    if cfg is None:
        with open(os.path.join(BASE, "config", "network.yaml")) as f:
            cfg = yaml.safe_load(f)
    data_dir = os.path.join(BASE, cfg["telemetry"]["data_dir"])
    df = load_telemetry(data_dir)
    log.info(f"Loaded {len(df)} telemetry records")

    window = cfg["telemetry"]["window_size"]
    df = compute_derived_features(df, window)

    kpi_thresh = cfg["ml"]["conflict_kpi_threshold"]
    df = label_conflicts(df, kpi_thresh)


    warmup_steps = int(cfg["experiment"].get("warmup_s", 5) / (cfg["telemetry"]["poll_interval_ms"] / 1000))
    if "step" in df.columns:
        df = df[df["step"] >= warmup_steps].reset_index(drop=True)

    parquet_path = os.path.join(data_dir, "dataset.parquet")
    csv_out = os.path.join(data_dir, "dataset.csv")
    df.to_parquet(parquet_path, index=False)
    df.to_csv(csv_out, index=False)
    log.info(f"Dataset: {len(df)} rows, {len(get_feature_columns(df))} features, "
             f"conflict rate={df['conflict'].mean():.2%}")
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    df = build_dataset()
    print(f"\nDataset shape: {df.shape}")
    print(f"Conflict rate: {df['conflict'].mean():.2%}")
    print(f"Features: {len(get_feature_columns(df))}")
    if "conflict_ho" in df.columns:
        print(f"  HO conflicts: {df['conflict_ho'].mean():.2%}")
        print(f"  PRB conflicts: {df['conflict_prb'].mean():.2%}")
        print(f"  Power conflicts: {df['conflict_power'].mean():.2%}")
        print(f"  Carrier conflicts: {df['conflict_carrier'].mean():.2%}")
