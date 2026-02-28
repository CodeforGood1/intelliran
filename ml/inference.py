import os, json, time, logging
import numpy as np

log = logging.getLogger(__name__)
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class ConflictPredictor:

    def __init__(self, models_dir=None, prefer="onnx"):
        if models_dir is None:
            models_dir = os.path.join(BASE, "models")
        self.models_dir = models_dir
        self.model = None
        self.model_type = None
        self.feature_cols = None
        self._input_buffer = None
        self._scaler_mean = None
        self._scaler_std = None
        self._load_features()
        self._load_scaler()

        loaders = {
            "onnx": [self._try_onnx, self._try_torchscript, self._try_lgbm],
            "torchscript": [self._try_torchscript, self._try_onnx, self._try_lgbm],
            "lgbm": [self._try_lgbm, self._try_onnx, self._try_torchscript],
        }
        for loader in loaders.get(prefer, loaders["onnx"]):
            if loader():
                break
        if self.model is None:
            raise RuntimeError("No trained model found")


        if self.feature_cols:
            self._input_buffer = np.zeros((1, len(self.feature_cols)), dtype=np.float32)


        self._warmup()
        log.info(f"Predictor ready: {self.model_type}, {len(self.feature_cols or [])} features")

    def _load_features(self):
        fp = os.path.join(self.models_dir, "feature_cols.json")
        if os.path.exists(fp):
            with open(fp) as f:
                self.feature_cols = json.load(f)

    def _load_scaler(self):
        sp = os.path.join(self.models_dir, "mlp_scaler.npz")
        if os.path.exists(sp):
            d = np.load(sp)
            self._scaler_mean = d["mean"]
            self._scaler_std = d["std"]

    def _try_onnx(self):
        try:
            import onnxruntime as ort

            for fname in ["mlp_model_quant.onnx", "mlp_model.onnx"]:
                p = os.path.join(self.models_dir, fname)
                if os.path.exists(p):
                    opts = ort.SessionOptions()
                    opts.intra_op_num_threads = 1
                    opts.inter_op_num_threads = 1
                    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                    opts.enable_cpu_mem_arena = True
                    opts.enable_mem_pattern = True
                    self.model = ort.InferenceSession(p, opts, providers=["CPUExecutionProvider"])
                    self.model_type = "onnx"
                    self._onnx_input_name = self.model.get_inputs()[0].name
                    log.info(f"Loaded ONNX: {fname}")
                    return True
        except Exception as e:
            log.debug(f"ONNX load failed: {e}")
        return False

    def _try_torchscript(self):
        try:
            import torch
            p = os.path.join(self.models_dir, "mlp_traced.pt")
            if os.path.exists(p):
                self.model = torch.jit.load(p, map_location="cpu")
                self.model.eval()
                torch.set_num_threads(1)
                self.model_type = "torchscript"
                return True
        except Exception as e:
            log.debug(f"TorchScript load failed: {e}")
        return False

    def _try_lgbm(self):
        try:
            import joblib
            p = os.path.join(self.models_dir, "lgbm_model.joblib")
            if os.path.exists(p):
                self.model = joblib.load(p)
                self.model_type = "lgbm"
                return True
        except Exception as e:
            log.debug(f"LightGBM load failed: {e}")
        return False

    def _warmup(self, n=50):
        if self.feature_cols:
            dummy = {c: 0.0 for c in self.feature_cols}
            for _ in range(n):
                self.predict(dummy)

    def predict(self, features: dict) -> dict:
        t0 = time.perf_counter()


        if self._input_buffer is not None and self.feature_cols:
            for i, c in enumerate(self.feature_cols):
                self._input_buffer[0, i] = features.get(c, 0.0)
            x = self._input_buffer
        else:
            x = np.array([list(features.values())], dtype=np.float32)

        if self.model_type == "onnx":

            if self._scaler_mean is not None:
                x_scaled = (x - self._scaler_mean) / self._scaler_std
            else:
                x_scaled = x
            prob = float(self.model.run(None, {self._onnx_input_name: x_scaled.astype(np.float32)})[0][0][0])
        elif self.model_type == "torchscript":
            import torch
            if self._scaler_mean is not None:
                x_scaled = (x - self._scaler_mean) / self._scaler_std
            else:
                x_scaled = x
            with torch.no_grad():
                prob = float(self.model(torch.from_numpy(x_scaled.astype(np.float32))).item())
        elif self.model_type == "lgbm":
            prob = float(self.model.predict_proba(x)[0, 1])
        else:
            prob = 0.5

        latency_ms = (time.perf_counter() - t0) * 1000
        return {
            "probability": round(prob, 4),
            "conflict": int(prob >= 0.5),
            "latency_ms": round(latency_ms, 4),
            "model_type": self.model_type,
        }

    def benchmark(self, features: dict, n=1000) -> dict:
        times = []
        for _ in range(n):
            r = self.predict(features)
            times.append(r["latency_ms"])
        times = np.array(times)
        return {
            "model_type": self.model_type,
            "n": n,
            "mean_ms": round(float(times.mean()), 4),
            "p50_ms": round(float(np.percentile(times, 50)), 4),
            "p95_ms": round(float(np.percentile(times, 95)), 4),
            "p99_ms": round(float(np.percentile(times, 99)), 4),
            "min_ms": round(float(times.min()), 4),
            "max_ms": round(float(times.max()), 4),
        }
