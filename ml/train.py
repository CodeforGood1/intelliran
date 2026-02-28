import os, sys, logging, yaml, time, json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, confusion_matrix

log = logging.getLogger(__name__)
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)


def set_seeds(seed=42):
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def evaluate_model(name, model, X_train, y_train, X_test, y_test, train_time, is_mlp=False):
    if is_mlp:
        test_probs = model.predict_proba(X_test)
        train_probs = model.predict_proba(X_train)
    else:
        test_probs = model.predict_proba(X_test)[:, 1]
        train_probs = model.predict_proba(X_train)[:, 1]

    test_preds = (test_probs >= 0.5).astype(int)
    test_auc = roc_auc_score(y_test, test_probs) if len(np.unique(y_test)) > 1 else 0.5
    train_auc = roc_auc_score(y_train, train_probs) if len(np.unique(y_train)) > 1 else 0.5
    p, r, f, _ = precision_recall_fscore_support(y_test, test_preds, average="binary", zero_division=0)
    cm = confusion_matrix(y_test, test_preds).tolist()
    gap = round(train_auc - test_auc, 4)

    log.info(f"{name}: AUC={test_auc:.4f} (train={train_auc:.4f}, gap={gap:+.4f}), "
             f"P={p:.3f}, R={r:.3f}, F1={f:.3f}, time={train_time:.1f}s")

    return {
        "auc": test_auc, "train_auc": train_auc, "overfit_gap": gap,
        "precision": p, "recall": r, "f1": f,
        "confusion_matrix": cm, "train_time": train_time,
    }


def train_all(cfg=None):
    if cfg is None:
        with open(os.path.join(BASE, "config", "network.yaml")) as f:
            cfg = yaml.safe_load(f)

    seed = cfg["ml"]["seed"]
    set_seeds(seed)

    from dataset.builder import build_dataset, train_test_split_dataset, get_feature_columns
    data_path = os.path.join(BASE, cfg["telemetry"]["data_dir"], "dataset.parquet")
    if os.path.exists(data_path):
        df = pd.read_parquet(data_path)
        log.info(f"Loaded existing dataset: {len(df)} rows")
    else:
        df = build_dataset(cfg)

    feature_cols = get_feature_columns(df)
    df = df.dropna(subset=feature_cols + ["conflict"])

    train_df, test_df = train_test_split_dataset(df, cfg["ml"]["test_size"], seed)
    X_train, y_train = train_df[feature_cols].values.astype(np.float32), train_df["conflict"].values
    X_test, y_test = test_df[feature_cols].values.astype(np.float32), test_df["conflict"].values
    log.info(f"Train: {len(X_train)}, Test: {len(X_test)}, Features: {len(feature_cols)}")
    log.info(f"Train conflict rate: {y_train.mean():.2%}, Test: {y_test.mean():.2%}")

    models_dir = os.path.join(BASE, "models")
    os.makedirs(models_dir, exist_ok=True)
    results = {}

    with open(os.path.join(models_dir, "feature_cols.json"), "w") as f:
        json.dump(feature_cols, f)

    import joblib
    cfg_ml = cfg["ml"]


    log.info("Training LightGBM...")
    t0 = time.time()
    from ml.models import train_lgbm
    lgbm = train_lgbm(X_train, y_train, X_test, y_test, cfg_ml, seed)
    results["lgbm"] = evaluate_model("LightGBM", lgbm, X_train, y_train, X_test, y_test, time.time()-t0)
    joblib.dump(lgbm, os.path.join(models_dir, "lgbm_model.joblib"))


    log.info("Training XGBoost...")
    t0 = time.time()
    from ml.models import train_xgboost
    xgb_model = train_xgboost(X_train, y_train, X_test, y_test, cfg_ml, seed)
    results["xgboost"] = evaluate_model("XGBoost", xgb_model, X_train, y_train, X_test, y_test, time.time()-t0)
    joblib.dump(xgb_model, os.path.join(models_dir, "xgb_model.joblib"))


    log.info("Training Random Forest...")
    t0 = time.time()
    from ml.models import train_random_forest
    rf = train_random_forest(X_train, y_train, seed)
    results["random_forest"] = evaluate_model("RF", rf, X_train, y_train, X_test, y_test, time.time()-t0)
    joblib.dump(rf, os.path.join(models_dir, "rf_model.joblib"))


    log.info("Training Logistic Regression...")
    t0 = time.time()
    from ml.models import train_logreg
    lr = train_logreg(X_train, y_train, seed)
    results["logistic_regression"] = evaluate_model("LogReg", lr, X_train, y_train, X_test, y_test, time.time()-t0)
    joblib.dump(lr, os.path.join(models_dir, "logreg_model.joblib"))


    try:
        import torch
        log.info("Training PyTorch MLP...")
        t0 = time.time()
        from ml.models import ConflictMLP
        mlp_cfg = cfg_ml.get("mlp", {})
        hidden = tuple(mlp_cfg.get("hidden_layers", [128, 64, 32]))
        mlp = ConflictMLP(input_dim=len(feature_cols), hidden_layers=hidden,
                          dropout=mlp_cfg.get("dropout", 0.15))
        mlp.fit(X_train, y_train,
                epochs=mlp_cfg.get("epochs", 50),
                lr=mlp_cfg.get("lr", 0.0005),
                batch_size=mlp_cfg.get("batch_size", 256),
                weight_decay=mlp_cfg.get("weight_decay", 0.0001))
        results["mlp"] = evaluate_model("MLP", mlp, X_train, y_train, X_test, y_test, time.time()-t0, is_mlp=True)


        mlp.model.eval()
        example = torch.randn(1, len(feature_cols)).to(mlp.device)
        traced = torch.jit.trace(mlp.model, example)
        traced.save(os.path.join(models_dir, "mlp_traced.pt"))
        torch.save(mlp.model.state_dict(), os.path.join(models_dir, "mlp_state.pth"))

        np.savez(os.path.join(models_dir, "mlp_scaler.npz"),
                 mean=mlp.scaler_mean, std=mlp.scaler_std)
        log.info("MLP saved (TorchScript + state + scaler)")
    except Exception as e:
        log.warning(f"MLP training failed: {e}")
        results["mlp"] = {"error": str(e)}


    log.info("=" * 60)
    log.info("MODEL COMPARISON SUMMARY")
    log.info("=" * 60)
    for name, m in sorted(results.items(), key=lambda x: x[1].get("auc", 0), reverse=True):
        if "error" in m:
            log.info(f"  {name:20s}: ERROR - {m['error']}")
        else:
            log.info(f"  {name:20s}: AUC={m['auc']:.4f}  F1={m['f1']:.3f}  gap={m['overfit_gap']:+.4f}  time={m['train_time']:.1f}s")
    log.info("=" * 60)

    with open(os.path.join(models_dir, "training_results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    train_all()
