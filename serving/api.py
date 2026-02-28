import os, sys, json, time, logging
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

app = FastAPI(title="5G RAN AI Lab - xApp Conflict Detection", version="2.0")


predictor = None
lime_exp = None
shap_exp = None
cf_gen = None
feature_cols = None
train_data = None
request_count = 0
total_latency = 0.0
latency_buckets = []


class PredictRequest(BaseModel):
    features: Dict[str, float]

class BatchPredictRequest(BaseModel):
    instances: List[Dict[str, float]]

class PredictResponse(BaseModel):
    probability: float
    conflict: int
    latency_ms: float
    model_type: str
    top_explanations: Optional[Dict] = None
    counterfactual: Optional[Dict] = None


@app.on_event("startup")
def startup():
    global predictor, lime_exp, shap_exp, cf_gen, feature_cols, train_data
    from ml.inference import ConflictPredictor
    predictor = ConflictPredictor(prefer="lgbm")
    feature_cols = predictor.feature_cols

    import pandas as pd
    ds_path = os.path.join(BASE, "data", "dataset.parquet")
    if os.path.exists(ds_path) and feature_cols:
        df = pd.read_parquet(ds_path)
        cols_avail = [c for c in feature_cols if c in df.columns]
        train_data = df[cols_avail].values.astype(np.float32)[:500]

        try:
            from explainability.lime_analysis import LimeExplainer, make_sklearn_predict_fn
            import joblib
            lgbm = joblib.load(os.path.join(BASE, "models", "lgbm_model.joblib"))
            lime_exp = LimeExplainer(train_data, cols_avail, make_sklearn_predict_fn(lgbm))
            log.info("LIME explainer loaded")
        except Exception as e:
            log.warning(f"LIME init failed: {e}")

        try:
            from explainability.shap_analysis import ShapExplainer
            import joblib
            lgbm = joblib.load(os.path.join(BASE, "models", "lgbm_model.joblib"))
            shap_exp = ShapExplainer(lgbm, "lgbm", train_data, cols_avail)
            log.info("SHAP explainer loaded")
        except Exception as e:
            log.warning(f"SHAP init failed: {e}")

        try:
            from explainability.counterfactual import CounterfactualGenerator, build_feature_ranges
            ranges = build_feature_ranges(df, cols_avail)
            import joblib
            lgbm_m = joblib.load(os.path.join(BASE, "models", "lgbm_model.joblib"))
            def pred_fn(X):
                return lgbm_m.predict_proba(X)[:, 1]
            cf_gen = CounterfactualGenerator(pred_fn, cols_avail, ranges)
            log.info("Counterfactual generator loaded")
        except Exception as e:
            log.warning(f"Counterfactual init failed: {e}")

    log.info(f"Server ready. Model: {predictor.model_type if predictor else 'none'}")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": predictor is not None,
        "model_type": predictor.model_type if predictor else None,
        "feature_count": len(feature_cols) if feature_cols else 0,
        "explainers": {
            "lime": lime_exp is not None,
            "shap": shap_exp is not None,
            "counterfactual": cf_gen is not None,
        },
    }


@app.get("/metrics")
def metrics():
    global request_count, total_latency, latency_buckets
    p50 = float(np.median(latency_buckets)) if latency_buckets else 0
    p95 = float(np.percentile(latency_buckets, 95)) if latency_buckets else 0
    p99 = float(np.percentile(latency_buckets, 99)) if latency_buckets else 0
    return {
        "request_count": request_count,
        "avg_latency_ms": round(total_latency / max(request_count, 1), 4),
        "p50_latency_ms": round(p50, 4),
        "p95_latency_ms": round(p95, 4),
        "p99_latency_ms": round(p99, 4),
    }


def _record_latency(ms):
    global request_count, total_latency, latency_buckets
    request_count += 1
    total_latency += ms
    latency_buckets.append(ms)
    if len(latency_buckets) > 1000:
        latency_buckets = latency_buckets[-1000:]


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if predictor is None:
        raise HTTPException(503, "Model not loaded")
    result = predictor.predict(req.features)
    _record_latency(result["latency_ms"])

    explanations = {}
    if lime_exp and feature_cols:
        try:
            x = np.array([req.features.get(c, 0.0) for c in feature_cols], dtype=np.float32)
            explanations["lime"] = lime_exp.explain_instance(x, num_features=5)
        except Exception:
            pass
    if shap_exp and feature_cols:
        try:
            x = np.array([req.features.get(c, 0.0) for c in feature_cols], dtype=np.float32)
            explanations["shap"] = shap_exp.explain_instance(x, top_k=5)
        except Exception:
            pass

    cf_result = None
    if cf_gen and feature_cols:
        try:
            x = np.array([req.features.get(c, 0.0) for c in feature_cols], dtype=np.float32)
            imp = explanations.get("lime", {}).get("features", {})
            cf_result = cf_gen.generate(x, feature_importances=imp)
        except Exception:
            pass

    return PredictResponse(
        probability=result["probability"],
        conflict=result["conflict"],
        latency_ms=result["latency_ms"],
        model_type=result["model_type"],
        top_explanations=explanations or None,
        counterfactual=cf_result,
    )


@app.post("/predict/batch")
def predict_batch(req: BatchPredictRequest):
    if predictor is None:
        raise HTTPException(503, "Model not loaded")
    results = []
    for inst in req.instances:
        r = predictor.predict(inst)
        _record_latency(r["latency_ms"])
        results.append(r)
    return {"predictions": results, "count": len(results)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("serving.api:app", host="0.0.0.0", port=8000, workers=1, log_level="info")
