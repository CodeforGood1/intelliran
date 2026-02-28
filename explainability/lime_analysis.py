import logging, os, json
import numpy as np

log = logging.getLogger(__name__)
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class LimeExplainer:
    def __init__(self, training_data, feature_names, model_predict_fn):
        from lime.lime_tabular import LimeTabularExplainer
        n = min(len(training_data), 500)
        idx = np.random.choice(len(training_data), n, replace=False)
        self.explainer = LimeTabularExplainer(
            training_data[idx],
            feature_names=feature_names,
            class_names=["no_conflict", "conflict"],
            mode="classification",
            discretize_continuous=True,
        )
        self.predict_fn = model_predict_fn
        self.feature_names = feature_names

    def explain_instance(self, instance, num_features=10):
        exp = self.explainer.explain_instance(
            instance,
            self.predict_fn,
            num_features=num_features,
            num_samples=200,
        )
        feat_weights = exp.as_list()
        return {
            "features": {name: round(weight, 4) for name, weight in feat_weights},
            "prediction": int(exp.predict_proba[1] >= 0.5) if hasattr(exp, "predict_proba") else None,
            "intercept": round(float(exp.intercept[1]), 4) if hasattr(exp, "intercept") else 0,
        }


def make_sklearn_predict_fn(model):
    def fn(X):
        return model.predict_proba(X)
    return fn


def make_mlp_predict_fn(mlp_model):
    def fn(X):
        probs = mlp_model.predict_proba(X)
        return np.column_stack([1 - probs, probs])
    return fn


make_lgbm_predict_fn = make_sklearn_predict_fn
