import os, logging, time, json
import numpy as np

log = logging.getLogger(__name__)
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def export_onnx(models_dir=None, input_dim=None):
    if models_dir is None:
        models_dir = os.path.join(BASE, "models")
    try:
        import torch
        if input_dim is None:
            fc_path = os.path.join(models_dir, "feature_cols.json")
            if os.path.exists(fc_path):
                with open(fc_path) as f:
                    input_dim = len(json.load(f))
            else:
                input_dim = 50
        pt_path = os.path.join(models_dir, "mlp_traced.pt")
        if not os.path.exists(pt_path):
            log.warning("No TorchScript model, skipping ONNX export")
            return None
        model = torch.jit.load(pt_path, map_location="cpu")
        model.eval()
        dummy = torch.randn(1, input_dim)
        onnx_path = os.path.join(models_dir, "mlp_model.onnx")
        torch.onnx.export(model, dummy, onnx_path,
                          input_names=["features"],
                          output_names=["conflict_prob"],
                          dynamic_axes={"features": {0: "batch"}, "conflict_prob": {0: "batch"}},
                          opset_version=13, dynamo=False)
        log.info(f"ONNX exported: {onnx_path}")
        return onnx_path
    except Exception as e:
        log.warning(f"ONNX export failed: {e}")
        return None


def quantize_onnx(models_dir=None):
    if models_dir is None:
        models_dir = os.path.join(BASE, "models")
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        onnx_path = os.path.join(models_dir, "mlp_model.onnx")
        quant_path = os.path.join(models_dir, "mlp_model_quant.onnx")
        if not os.path.exists(onnx_path):
            return None
        quantize_dynamic(onnx_path, quant_path, weight_type=QuantType.QUInt8)
        log.info(f"Quantized ONNX: {quant_path}")
        return quant_path
    except Exception as e:
        log.warning(f"Quantization failed: {e}")
        return None


def benchmark_all(models_dir=None, n=1000):
    if models_dir is None:
        models_dir = os.path.join(BASE, "models")
    fc_path = os.path.join(models_dir, "feature_cols.json")
    if os.path.exists(fc_path):
        with open(fc_path) as f:
            ndim = len(json.load(f))
    else:
        ndim = 50
    dummy = np.random.randn(1, ndim).astype(np.float32)
    results = {}


    scaler_mean, scaler_std = None, None
    sp = os.path.join(models_dir, "mlp_scaler.npz")
    if os.path.exists(sp):
        d = np.load(sp)
        scaler_mean, scaler_std = d["mean"], d["std"]


    try:
        import joblib
        m = joblib.load(os.path.join(models_dir, "lgbm_model.joblib"))

        for _ in range(50):
            m.predict_proba(dummy)
        times = []
        for _ in range(n):
            t0 = time.perf_counter()
            m.predict_proba(dummy)
            times.append((time.perf_counter() - t0) * 1000)
        times = np.array(times)
        results["lgbm"] = {"mean_ms": round(float(times.mean()), 4),
                           "p50_ms": round(float(np.median(times)), 4),
                           "p95_ms": round(float(np.percentile(times, 95)), 4),
                           "p99_ms": round(float(np.percentile(times, 99)), 4)}
    except Exception:
        pass


    try:
        import joblib
        m = joblib.load(os.path.join(models_dir, "xgb_model.joblib"))
        for _ in range(50):
            m.predict_proba(dummy)
        times = []
        for _ in range(n):
            t0 = time.perf_counter()
            m.predict_proba(dummy)
            times.append((time.perf_counter() - t0) * 1000)
        times = np.array(times)
        results["xgboost"] = {"mean_ms": round(float(times.mean()), 4),
                              "p50_ms": round(float(np.median(times)), 4),
                              "p95_ms": round(float(np.percentile(times, 95)), 4),
                              "p99_ms": round(float(np.percentile(times, 99)), 4)}
    except Exception:
        pass


    try:
        import torch
        m = torch.jit.load(os.path.join(models_dir, "mlp_traced.pt"), map_location="cpu")
        m.eval()
        torch.set_num_threads(1)
        x = torch.from_numpy(dummy)
        if scaler_mean is not None:
            x = torch.from_numpy(((dummy - scaler_mean) / scaler_std).astype(np.float32))
        for _ in range(50):
            with torch.no_grad(): m(x)
        times = []
        for _ in range(n):
            t0 = time.perf_counter()
            with torch.no_grad(): m(x)
            times.append((time.perf_counter() - t0) * 1000)
        times = np.array(times)
        results["torchscript"] = {"mean_ms": round(float(times.mean()), 4),
                                  "p50_ms": round(float(np.median(times)), 4),
                                  "p95_ms": round(float(np.percentile(times, 95)), 4),
                                  "p99_ms": round(float(np.percentile(times, 99)), 4)}
    except Exception:
        pass


    for label, fname in [("onnx", "mlp_model.onnx"), ("onnx_quant", "mlp_model_quant.onnx")]:
        try:
            import onnxruntime as ort
            p = os.path.join(models_dir, fname)
            if not os.path.exists(p):
                continue
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 1
            opts.inter_op_num_threads = 1
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess = ort.InferenceSession(p, opts, providers=["CPUExecutionProvider"])
            inp_name = sess.get_inputs()[0].name
            x = dummy
            if scaler_mean is not None:
                x = ((dummy - scaler_mean) / scaler_std).astype(np.float32)
            for _ in range(50):
                sess.run(None, {inp_name: x})
            times = []
            for _ in range(n):
                t0 = time.perf_counter()
                sess.run(None, {inp_name: x})
                times.append((time.perf_counter() - t0) * 1000)
            times = np.array(times)
            results[label] = {"mean_ms": round(float(times.mean()), 4),
                              "p50_ms": round(float(np.median(times)), 4),
                              "p95_ms": round(float(np.percentile(times, 95)), 4),
                              "p99_ms": round(float(np.percentile(times, 99)), 4)}
        except Exception:
            pass

    log.info("=" * 60)
    log.info("INFERENCE BENCHMARK (n=%d)", n)
    log.info("=" * 60)
    for name, m in sorted(results.items(), key=lambda x: x[1]["mean_ms"]):
        log.info(f"  {name:15s}: mean={m['mean_ms']:.4f}ms  p50={m['p50_ms']:.4f}ms  p95={m['p95_ms']:.4f}ms  p99={m['p99_ms']:.4f}ms")
    log.info("=" * 60)

    with open(os.path.join(models_dir, "benchmark_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    export_onnx()
    quantize_onnx()
    benchmark_all()
