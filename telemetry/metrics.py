import numpy as np
import pandas as pd


def compute_derived_features(df: pd.DataFrame, window: int = 100) -> pd.DataFrame:
    out = df.copy()


    kpi_cols = [
        "DRB.UEThpDl", "DRB.UEThpUl", "prb_util_dl", "prb_util_ul",
        "L1M.RS-SINR", "L1M.RSRP", "L1M.RSRQ", "avg_cqi", "avg_mcs",
        "DRB.RlcSduDelayDl", "DRB.PacketLossRateDl", "avg_bler", "cell_load",
    ]
    for col in kpi_cols:
        if col not in out.columns:
            continue
        s = out[col]
        out[f"{col}_ma{window}"] = s.rolling(window, min_periods=1).mean()
        out[f"{col}_std{window}"] = s.rolling(window, min_periods=1).std().fillna(0)
        out[f"{col}_roc"] = s.diff().fillna(0)
        out[f"{col}_roc2"] = s.diff().diff().fillna(0)


    if "DRB.UEThpDl" in out.columns and "RRU.PrbUsedDl" in out.columns:
        prb_used = out["RRU.PrbUsedDl"].clip(lower=1)
        out["spectral_eff_dl"] = out["DRB.UEThpDl"] / (prb_used * 12 * 14 * 30)
        out["spectral_eff_dl"] = out["spectral_eff_dl"].clip(0, 10)


    if "RRU.PrbUsedDl" in out.columns and "RRU.PrbAvailDl" in out.columns:
        avail = out["RRU.PrbAvailDl"].clip(lower=1)
        out["prb_util_ratio"] = out["RRU.PrbUsedDl"] / avail
        out["prb_headroom"] = 1.0 - out["prb_util_ratio"]


    if "RRC.ConnEstabSucc" in out.columns and "RRC.ConnEstabAtt" in out.columns:
        att = out["RRC.ConnEstabAtt"].clip(lower=1)
        out["rrc_success_rate"] = out["RRC.ConnEstabSucc"] / att


    if "HO.SuccOutInterEnb" in out.columns and "HO.AttOutInterEnb" in out.columns:
        ho_att = out["HO.AttOutInterEnb"].clip(lower=1)
        out["ho_success_rate"] = out["HO.SuccOutInterEnb"] / ho_att

        out["ho_success_rate"] = out["ho_success_rate"].fillna(1.0).clip(0, 1)


    mcs_cols = ["TB.TotNbrDl.Qpsk", "TB.TotNbrDl.16Qam", "TB.TotNbrDl.64Qam", "TB.TotNbrDl.256Qam"]
    if all(c in out.columns for c in mcs_cols):
        mcs_arr = out[mcs_cols].values + 1e-10
        mcs_norm = mcs_arr / mcs_arr.sum(axis=1, keepdims=True)
        out["mcs_entropy"] = -(mcs_norm * np.log2(mcs_norm)).sum(axis=1)


    if "DRB.UEThpDl" in out.columns:
        roll_std = out["DRB.UEThpDl"].rolling(window, min_periods=1).std().fillna(0)
        roll_mean = out["DRB.UEThpDl"].rolling(window, min_periods=1).mean().clip(lower=1)
        out["throughput_burstiness"] = roll_std / roll_mean


    if "L1M.RS-SINR" in out.columns and "avg_cqi" in out.columns:
        expected_cqi = ((out["L1M.RS-SINR"] + 5) / 2.7).clip(1, 15)
        out["cqi_deviation"] = out["avg_cqi"] - expected_cqi


    if "DRB.RlcSduDelayDl" in out.columns:
        out["delay_jitter"] = out["DRB.RlcSduDelayDl"].diff().abs().fillna(0)


    action_cols = [c for c in out.columns if c.startswith(("ts_", "qos_", "mlb_", "es_"))]
    if action_cols:
        out["total_xapp_actions"] = out[action_cols].sum(axis=1)
        for c in action_cols:
            out[f"{c}_rolling"] = out[c].rolling(window, min_periods=1).sum()


    if "cell_id" in out.columns and "prb_util_dl" in out.columns:
        cell_mean = out.groupby("step")["prb_util_dl"].transform("mean")
        out["prb_util_dl_vs_avg"] = out["prb_util_dl"] - cell_mean

    if "cell_id" in out.columns and "DRB.UEThpDl" in out.columns:
        cell_mean = out.groupby("step")["DRB.UEThpDl"].transform("mean")
        out["thp_dl_vs_avg"] = out["DRB.UEThpDl"] - cell_mean


    if "cell_id" in out.columns and "cell_load" in out.columns:
        cell_std = out.groupby("step")["cell_load"].transform("std").fillna(0)
        out["load_imbalance_idx"] = cell_std

    return out
