import os, sys, json, time
import numpy as np
import pandas as pd
import streamlit as st

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

st.set_page_config(page_title="5G RAN AI Lab", page_icon="📡", layout="wide")
st.title("📡 5G RAN AI Lab — xApp Conflict Detection")


st.sidebar.header("Navigation")
page = st.sidebar.radio("Page", [
    "Cell KPI Overview",
    "xApp Conflict Analysis",
    "Model Comparison",
    "Live Inference",
    "Benchmark Results",
])


def load_dataset():
    p = os.path.join(BASE, "data", "dataset.parquet")
    if not os.path.exists(p):
        st.warning("No dataset found. Run: python -m telemetry.collector && python -m dataset.builder")
        return None
    return pd.read_parquet(p)


def load_telemetry():
    p = os.path.join(BASE, "data", "telemetry.parquet")
    if not os.path.exists(p):
        return None
    return pd.read_parquet(p)


def load_benchmark():
    p = os.path.join(BASE, "models", "benchmark_results.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


if page == "Cell KPI Overview":
    st.header("Cell-Level 3GPP KPI Dashboard")
    tdf = load_telemetry()
    if tdf is not None:
        cells = sorted(tdf["cell_id"].unique()) if "cell_id" in tdf.columns else []
        if cells:
            sel_cell = st.selectbox("Select Cell", cells)
            cdf = tdf[tdf["cell_id"] == sel_cell].copy()
            cdf = cdf.reset_index(drop=True)

            col1, col2, col3 = st.columns(3)
            kpis = {
                "prb_util_dl": "PRB Utilization DL (%)",
                "thp_dl_mbps": "DL Throughput (Mbps)",
                "avg_sinr_db": "Avg SINR (dB)",
            }
            for col, (k, label) in zip([col1, col2, col3], kpis.items()):
                if k in cdf.columns:
                    col.metric(label, f"{cdf[k].iloc[-1]:.1f}", f"Δ {cdf[k].diff().iloc[-1]:.2f}")

            st.subheader("KPI Time Series")
            ts_cols = [c for c in ["prb_util_dl", "thp_dl_mbps", "avg_sinr_db",
                                   "avg_rsrp_dbm", "connected_ues", "ho_success_count",
                                   "avg_cqi", "avg_mcs", "avg_bler"] if c in cdf.columns]
            sel_kpi = st.multiselect("Select KPIs", ts_cols, default=ts_cols[:3])
            if sel_kpi:
                st.line_chart(cdf[sel_kpi])

            st.subheader("Raw Data (last 50 rows)")
            st.dataframe(cdf.tail(50), use_container_width=True)
        else:
            st.info("No cell_id column in telemetry data")
    else:
        st.info("No telemetry data. Run: python -m telemetry.collector")


elif page == "xApp Conflict Analysis":
    st.header("xApp Conflict Analysis")
    df = load_dataset()
    if df is not None:
        conflict_cols = [c for c in df.columns if c.startswith("conflict")]
        if conflict_cols:
            st.subheader("Conflict Label Distribution")
            dist = {}
            for c in conflict_cols:
                if c in df.columns:
                    dist[c] = {"positive": int((df[c] > 0).sum()), "negative": int((df[c] == 0).sum())}
            st.dataframe(pd.DataFrame(dist).T, use_container_width=True)

            if "conflict_severity" in df.columns:
                st.subheader("Conflict Severity Distribution")
                st.bar_chart(df["conflict_severity"].value_counts().sort_index())

            st.subheader("Conflict Rate Over Time")
            if "conflict" in df.columns:
                window = st.slider("Rolling window", 10, 200, 50)
                rolling_rate = df["conflict"].rolling(window).mean()
                st.line_chart(rolling_rate.dropna())

        xapp_cols = [c for c in df.columns if c.startswith("n_action_")]
        if xapp_cols:
            st.subheader("xApp Action Counts")
            st.line_chart(df[xapp_cols])
    else:
        st.info("No dataset. Run pipeline first.")


elif page == "Model Comparison":
    st.header("Model Performance Comparison")

    results_path = os.path.join(BASE, "models", "training_results.json")
    if os.path.exists(results_path):
        with open(results_path) as f:
            results = json.load(f)
        rows = []
        for name, m in results.items():
            rows.append({
                "Model": name,
                "AUC": m.get("auc", 0),
                "F1": m.get("f1", 0),
                "Train-Test Gap": m.get("gap", 0),
            })
        rdf = pd.DataFrame(rows).sort_values("AUC", ascending=False)
        st.dataframe(rdf, use_container_width=True)
        st.bar_chart(rdf.set_index("Model")["AUC"])
    else:
        st.info("No training results. Run: python -m ml.train")

    bench = load_benchmark()
    if bench:
        st.subheader("Inference Latency Benchmark")
        brows = []
        for name, m in bench.items():
            brows.append({"Format": name, **m})
        bdf = pd.DataFrame(brows).sort_values("mean_ms")
        st.dataframe(bdf, use_container_width=True)
        st.bar_chart(bdf.set_index("Format")["mean_ms"])


elif page == "Live Inference":
    st.header("Live Conflict Prediction")
    df = load_dataset()
    if df is not None:
        fc_path = os.path.join(BASE, "models", "feature_cols.json")
        if os.path.exists(fc_path):
            with open(fc_path) as f:
                feature_cols = json.load(f)
            cols_avail = [c for c in feature_cols if c in df.columns]

            st.sidebar.subheader("Sample Selection")
            idx = st.sidebar.slider("Row index", 0, len(df) - 1, len(df) // 2)
            sample = df.iloc[idx]

            st.subheader("Input Features")
            feat_dict = {c: float(sample[c]) for c in cols_avail}
            st.json(feat_dict)

            if st.button("Run Prediction"):
                try:
                    from ml.inference import ConflictPredictor
                    pred = ConflictPredictor(prefer="lgbm")
                    result = pred.predict(feat_dict)
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Probability", f"{result['probability']:.4f}")
                    c2.metric("Conflict", "YES" if result["conflict"] else "NO")
                    c3.metric("Latency", f"{result['latency_ms']:.3f} ms")
                    st.json(result)
                except Exception as e:
                    st.error(f"Prediction failed: {e}")
        else:
            st.info("No feature_cols.json. Run training first.")
    else:
        st.info("No dataset available.")


elif page == "Benchmark Results":
    st.header("Inference Benchmark Results")
    bench = load_benchmark()
    if bench:
        for name, m in sorted(bench.items(), key=lambda x: x[1]["mean_ms"]):
            with st.expander(f"{name} — mean={m['mean_ms']:.4f}ms", expanded=True):
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Mean", f"{m['mean_ms']:.4f} ms")
                c2.metric("P50", f"{m['p50_ms']:.4f} ms")
                c3.metric("P95", f"{m['p95_ms']:.4f} ms")
                c4.metric("P99", f"{m['p99_ms']:.4f} ms")
        sub_ms = [n for n, m in bench.items() if m["p99_ms"] < 1.0]
        if sub_ms:
            st.success(f"Sub-millisecond (p99 < 1ms): {', '.join(sub_ms)}")
        else:
            st.warning("No format achieved sub-ms p99 latency")
    else:
        st.info("No benchmark results. Run: python -m ml.optimize")

st.sidebar.markdown("---")
st.sidebar.caption("5G RAN AI Lab v2.0 | 3GPP/O-RAN Compliant")
