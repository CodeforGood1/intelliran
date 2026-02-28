import logging, os
import numpy as np

log = logging.getLogger(__name__)


class ShapExplainer:
    def __init__(self, model, model_type, background_data, feature_names):
        import shap
        self.feature_names = feature_names
        self.model_type = model_type
        n = min(len(background_data), 100)
        bg = background_data[np.random.choice(len(background_data), n, replace=False)]

        if model_type in ("lgbm", "xgboost", "rf"):
            self.explainer = shap.TreeExplainer(model)
        else:
            self.explainer = shap.KernelExplainer(model, bg)

    def explain_instance(self, instance, top_k=10):
        x = instance.reshape(1, -1) if instance.ndim == 1 else instance
        shap_values = self.explainer.shap_values(x)
        if isinstance(shap_values, list):
            vals = shap_values[1][0]
        elif shap_values.ndim == 3:
            vals = shap_values[0, :, 1]
        else:
            vals = shap_values[0]
        indices = np.argsort(np.abs(vals))[-top_k:][::-1]
        result = {}
        for i in indices:
            name = self.feature_names[i] if i < len(self.feature_names) else f"f{i}"
            result[name] = round(float(vals[i]), 4)
        ev = self.explainer.expected_value
        base = float(ev[1]) if isinstance(ev, (list, np.ndarray)) else float(ev)
        return {"shap_values": result, "base_value": round(base, 4)}

    def explain_batch(self, X, top_k=5):
        return [self.explain_instance(X[i], top_k) for i in range(len(X))]
