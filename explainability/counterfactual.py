import logging
import numpy as np

log = logging.getLogger(__name__)


class CounterfactualGenerator:
    def __init__(self, predict_fn, feature_names, feature_ranges):
        self.predict_fn = predict_fn
        self.feature_names = feature_names
        self.feature_ranges = feature_ranges

    def generate(self, instance, feature_importances=None, max_changes=5, steps=10):
        x = instance.copy().astype(np.float32)
        orig_prob = float(self.predict_fn(x.reshape(1, -1))[0])
        orig_label = int(orig_prob >= 0.5)
        target_label = 1 - orig_label

        if feature_importances:
            ranked = sorted(feature_importances.items(), key=lambda kv: abs(kv[1]), reverse=True)
            ranked_indices = []
            for name, _ in ranked[:max_changes]:
                if name in self.feature_names:
                    ranked_indices.append(self.feature_names.index(name))
        else:
            ranked_indices = list(range(min(max_changes, len(self.feature_names))))

        best_cf = None
        best_dist = float("inf")

        for idx in ranked_indices:
            fname = self.feature_names[idx]
            lo, hi = self.feature_ranges.get(fname, (x[idx] * 0.5, x[idx] * 1.5))
            for val in np.linspace(lo, hi, steps):
                cf = x.copy()
                cf[idx] = val
                cf_prob = float(self.predict_fn(cf.reshape(1, -1))[0])
                if int(cf_prob >= 0.5) == target_label:
                    dist = abs(val - x[idx])
                    if dist < best_dist:
                        best_dist = dist
                        best_cf = cf.copy()

        if best_cf is not None:
            changes = {}
            for i in range(len(x)):
                if abs(best_cf[i] - x[i]) > 1e-6:
                    changes[self.feature_names[i]] = {
                        "from": round(float(x[i]), 4),
                        "to": round(float(best_cf[i]), 4),
                        "delta": round(float(best_cf[i] - x[i]), 4),
                    }
            new_prob = float(self.predict_fn(best_cf.reshape(1, -1))[0])
            return {
                "found": True,
                "original_prediction": orig_prob,
                "counterfactual_prediction": new_prob,
                "changes": changes,
            }
        return {"found": False, "original_prediction": orig_prob, "message": "No counterfactual found"}


def build_feature_ranges(df, feature_names):
    ranges = {}
    for col in feature_names:
        if col in df.columns:
            ranges[col] = (float(df[col].min()), float(df[col].max()))
    return ranges
