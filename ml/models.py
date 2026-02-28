import logging, os
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def train_lgbm(X_train, y_train, X_test, y_test, cfg_ml=None, seed=42):
    import lightgbm as lgb
    params = {
        "objective": "binary",
        "metric": "auc",
        "verbosity": -1,
        "seed": seed,
        "n_jobs": 1,
        "n_estimators": 300,
        "max_depth": 6,
        "num_leaves": 31,
        "learning_rate": 0.03,
        "min_child_samples": 30,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.5,
        "reg_lambda": 1.0,
    }
    if cfg_ml and "lgbm" in cfg_ml:
        params.update(cfg_ml["lgbm"])
        params["seed"] = seed
        params["verbosity"] = -1
        params["n_jobs"] = 1
    model = lgb.LGBMClassifier(**params)
    model.fit(X_train, y_train,
              eval_set=[(X_test, y_test)],
              callbacks=[lgb.early_stopping(30, verbose=False),
                         lgb.log_evaluation(period=100)])
    return model


def train_xgboost(X_train, y_train, X_test, y_test, cfg_ml=None, seed=42):
    import xgboost as xgb
    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "seed": seed,
        "n_jobs": 1,
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.03,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.5,
        "reg_lambda": 1.0,
        "tree_method": "hist",
    }
    if cfg_ml and "xgb" in cfg_ml:
        params.update(cfg_ml["xgb"])
        params["seed"] = seed
        params["n_jobs"] = 1
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train,
              eval_set=[(X_test, y_test)],
              verbose=False)
    return model


def train_random_forest(X_train, y_train, seed=42):
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(
        n_estimators=200, max_depth=8, min_samples_split=20,
        min_samples_leaf=10, max_features="sqrt",
        random_state=seed, n_jobs=1)
    model.fit(X_train, y_train)
    return model


def train_logreg(X_train, y_train, seed=42):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(C=1.0, max_iter=1000, random_state=seed, solver="lbfgs"))
    ])
    model.fit(X_train, y_train)
    return model


def _get_device():
    try:
        import torch
        if torch.cuda.is_available():
            return torch.device("cuda")
    except Exception:
        pass
    try:
        import torch
        return torch.device("cpu")
    except ImportError:
        return None


class ConflictMLP:

    def __init__(self, input_dim, hidden_layers=(128, 64, 32), dropout=0.15, device=None):
        import torch
        import torch.nn as nn
        self.device = device or _get_device()
        layers = []
        prev = input_dim
        for h in hidden_layers:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        layers.append(nn.Sigmoid())
        self.model = nn.Sequential(*layers).to(self.device)
        self.scaler_mean = None
        self.scaler_std = None

    def fit(self, X_train, y_train, epochs=50, lr=0.0005, batch_size=256,
            weight_decay=0.0001, X_val=None, y_val=None):
        import torch
        from torch.utils.data import TensorDataset, DataLoader

        X_np = X_train.values if hasattr(X_train, "values") else X_train
        y_np = y_train.values if hasattr(y_train, "values") else y_train


        self.scaler_mean = X_np.mean(axis=0).astype(np.float32)
        self.scaler_std = X_np.std(axis=0).astype(np.float32)
        self.scaler_std[self.scaler_std < 1e-8] = 1.0
        X_scaled = ((X_np - self.scaler_mean) / self.scaler_std).astype(np.float32)

        X_t = torch.from_numpy(X_scaled).to(self.device)
        y_t = torch.from_numpy(y_np.astype(np.float32)).unsqueeze(1).to(self.device)
        ds = TensorDataset(X_t, y_t)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

        opt = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        loss_fn = torch.nn.BCELoss()

        self.model.train()
        best_loss = float("inf")
        patience, patience_count = 10, 0
        for epoch in range(epochs):
            total_loss = 0
            for xb, yb in dl:
                opt.zero_grad()
                pred = self.model(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()
                total_loss += loss.item() * xb.size(0)
            scheduler.step()
            avg_loss = total_loss / len(ds)
            if avg_loss < best_loss - 0.001:
                best_loss = avg_loss
                patience_count = 0
            else:
                patience_count += 1
            if patience_count >= patience:
                log.info(f"Early stopping at epoch {epoch+1}")
                break
            if (epoch + 1) % 10 == 0:
                log.info(f"  Epoch {epoch+1}/{epochs}, loss={avg_loss:.4f}, lr={scheduler.get_last_lr()[0]:.6f}")

    def predict_proba(self, X):
        import torch
        self.model.eval()
        X_np = X.values if hasattr(X, "values") else X
        if self.scaler_mean is not None:
            X_np = ((X_np - self.scaler_mean) / self.scaler_std).astype(np.float32)
        x_t = torch.from_numpy(X_np.astype(np.float32)).to(self.device)
        with torch.no_grad():
            return self.model(x_t).cpu().numpy().flatten()
