import os, sys, json, time
import numpy as np
import pandas as pd
import pytest

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)


@pytest.fixture
def cfg():
    from simulator.topology import load_config
    return load_config()


class TestRANSimulator:
    def test_init(self, cfg):
        from simulator.topology import RANSimulator
        sim = RANSimulator(cfg)
        assert len(sim.cells) > 0
        assert len(sim.ues) > 0

    def test_step(self, cfg):
        from simulator.topology import RANSimulator
        sim = RANSimulator(cfg)
        sim.step(0.01)

    def test_cell_snapshot(self, cfg):
        from simulator.topology import RANSimulator
        sim = RANSimulator(cfg)
        sim.step(0.01)
        for cid in sim.cells:
            snap = sim.get_cell_snapshot(cid)
            assert "cell_id" in snap
            assert "prb_util_dl" in snap
            assert "connected_ues" in snap

    def test_handover(self, cfg):
        from simulator.topology import RANSimulator
        sim = RANSimulator(cfg)
        sim.step(0.01)
        ue_ids = list(sim.ues.keys())
        cell_ids = list(sim.cells.keys())
        if len(ue_ids) > 0 and len(cell_ids) > 1:
            ue = sim.ues[ue_ids[0]]
            target = [c for c in cell_ids if c != ue.serving_cell][0]
            sim.trigger_handover(ue_ids[0], target)


class TestTrafficModel:
    def test_init(self, cfg):
        from simulator.traffic import TrafficModel
        from simulator.topology import RANSimulator
        sim = RANSimulator(cfg)
        tm = TrafficModel(sim, cfg)
        assert tm is not None

    def test_step(self, cfg):
        from simulator.traffic import TrafficModel
        from simulator.topology import RANSimulator
        sim = RANSimulator(cfg)
        tm = TrafficModel(sim, cfg)
        sim.step(0.01)
        tm.step(0.01)
        for cid in sim.cells:
            load = tm.get_cell_load(cid)
            assert 0 <= load <= 1.5


class TestControllers:
    def test_xapps(self, cfg):
        from simulator.topology import RANSimulator
        from simulator.traffic import TrafficModel
        from simulator.controllers import ControllerRunner
        sim = RANSimulator(cfg)
        tm = TrafficModel(sim, cfg)
        runner = ControllerRunner(sim, tm, cfg)
        sim.step(0.01)
        tm.step(0.01)
        runner._control_step()
        assert runner._step_count >= 1


class TestTelemetry:
    def test_collector(self, cfg):
        from simulator.topology import RANSimulator
        from simulator.traffic import TrafficModel
        from simulator.controllers import ControllerRunner
        from telemetry.collector import TelemetryCollector
        sim = RANSimulator(cfg)
        tm = TrafficModel(sim, cfg)
        runner = ControllerRunner(sim, tm, cfg)
        coll = TelemetryCollector(sim, tm, runner, cfg)
        sim.step(0.01)
        tm.step(0.01)
        rows = coll._collect_one()
        assert len(rows) > 0
        assert "cell_id" in rows[0]
        assert "prb_util_dl" in rows[0]

    def test_feature_engineering(self):
        from telemetry.metrics import compute_derived_features
        n = 100
        df = pd.DataFrame({
            "cell_id": ["cell_1"] * n,
            "prb_util_dl": np.random.rand(n) * 80,
            "DRB.UEThpDl": np.random.rand(n) * 500,
            "L1M.RS-SINR": np.random.randn(n) * 5 + 15,
            "L1M.RSRP": np.random.randn(n) * 5 - 90,
            "L1M.RSRQ": np.random.randn(n) * 3 - 10,
            "connected_ues": np.random.randint(10, 50, n),
            "avg_cqi": np.random.randint(5, 15, n),
            "avg_mcs": np.random.randint(5, 28, n),
            "avg_bler": np.random.rand(n) * 0.1,
            "RRC.ConnEstabSucc": np.random.randint(80, 100, n),
            "RRC.ConnEstabAtt": np.random.randint(90, 110, n),
            "HO.SuccOutInterEnb": np.random.randint(0, 5, n),
            "HO.AttOutInterEnb": np.random.randint(0, 8, n),
            "DRB.RlcSduDelayDl": np.random.rand(n) * 20,
            "DRB.UEThpUl": np.random.rand(n) * 100,
            "prb_util_ul": np.random.rand(n) * 60,
            "cell_load": np.random.rand(n) * 0.8 + 0.1,
            "step": list(range(n)),
            "RRU.PrbUsedDl": np.random.randint(50, 200, n),
            "RRU.PrbAvailDl": np.full(n, 273),
            "TB.TotNbrDl.Qpsk": np.random.randint(0, 10, n),
            "TB.TotNbrDl.16Qam": np.random.randint(0, 20, n),
            "TB.TotNbrDl.64Qam": np.random.randint(0, 30, n),
            "TB.TotNbrDl.256Qam": np.random.randint(0, 40, n),
            "DRB.PacketLossRateDl": np.random.rand(n) * 0.01,
            "ts_handover": np.random.randint(0, 3, n),
            "qos_prb_reservation": np.random.randint(0, 3, n),
            "mlb_cio_adjust": np.random.randint(0, 3, n),
            "es_mimo_reduce": np.random.randint(0, 3, n),
        })
        result = compute_derived_features(df)
        assert len(result) > 0
        assert result.shape[1] > df.shape[1]


class TestDatasetBuilder:
    def test_label_conflicts(self):
        from dataset.builder import label_conflicts
        n = 200
        df = pd.DataFrame({
            "cell_id": ["cell_1"] * n,
            "ts_handover": np.random.randint(0, 3, n),
            "ts_carrier_agg": np.random.randint(0, 2, n),
            "qos_prb_reservation": np.random.randint(0, 3, n),
            "qos_power_boost": np.random.randint(0, 2, n),
            "qos_mcs_override": np.random.randint(0, 2, n),
            "qos_sched_weight": np.random.randint(0, 3, n),
            "mlb_cio_adjust": np.random.randint(0, 3, n),
            "mlb_prb_cap": np.random.randint(0, 3, n),
            "mlb_ho_threshold": np.random.randint(0, 2, n),
            "es_mimo_reduce": np.random.randint(0, 2, n),
            "es_dtx_drx": np.random.randint(0, 2, n),
            "es_carrier_shutdown": np.random.randint(0, 1, n),
            "n_conflicts": np.random.randint(0, 5, n),
            "DRB.UEThpDl": np.random.rand(n) * 500 + 100,
            "DRB.RlcSduDelayDl": np.random.rand(n) * 20,
            "avg_bler": np.random.rand(n) * 0.1,
        })
        result = label_conflicts(df)
        assert "conflict" in result.columns
        assert "conflict_severity" in result.columns


class TestMLPipeline:
    @pytest.fixture(autouse=True)
    def setup_data(self, tmp_path):
        self.tmp = tmp_path
        n = 500
        ncols = 30
        X = np.random.randn(n, ncols).astype(np.float32)
        y = (X[:, 0] + X[:, 1] * 0.5 + np.random.randn(n) * 0.3 > 0).astype(int)
        cols = [f"feat_{i}" for i in range(ncols)]
        df = pd.DataFrame(X, columns=cols)
        df["conflict"] = y
        self.df = df
        self.feature_cols = cols
        self.models_dir = str(tmp_path / "models")
        os.makedirs(self.models_dir, exist_ok=True)

    def test_lgbm(self):
        from ml.models import train_lgbm
        from sklearn.model_selection import train_test_split
        X = self.df[self.feature_cols].values
        y = self.df["conflict"].values
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
        model = train_lgbm(Xtr, ytr, Xte, yte)
        preds = model.predict_proba(Xte)
        assert preds.shape == (len(Xte), 2)

    def test_xgboost(self):
        from ml.models import train_xgboost
        from sklearn.model_selection import train_test_split
        X = self.df[self.feature_cols].values
        y = self.df["conflict"].values
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
        model = train_xgboost(Xtr, ytr, Xte, yte)
        preds = model.predict_proba(Xte)
        assert preds.shape == (len(Xte), 2)

    def test_rf(self):
        from ml.models import train_random_forest
        from sklearn.model_selection import train_test_split
        X = self.df[self.feature_cols].values
        y = self.df["conflict"].values
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
        model = train_random_forest(Xtr, ytr)
        preds = model.predict_proba(Xte)
        assert preds.shape == (len(Xte), 2)

    def test_logreg(self):
        from ml.models import train_logreg
        from sklearn.model_selection import train_test_split
        X = self.df[self.feature_cols].values
        y = self.df["conflict"].values
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
        model = train_logreg(Xtr, ytr)
        preds = model.predict_proba(Xte)
        assert preds.shape == (len(Xte), 2)

    def test_mlp(self):
        from ml.models import ConflictMLP
        from sklearn.model_selection import train_test_split
        X = self.df[self.feature_cols].values.astype(np.float32)
        y = self.df["conflict"].values
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
        mlp = ConflictMLP(input_dim=X.shape[1])
        mlp.fit(Xtr, ytr, epochs=5, X_val=Xte, y_val=yte)
        preds = mlp.predict_proba(Xte)
        assert len(preds) == len(Xte)
        assert all(0 <= p <= 1 for p in preds)


class TestInference:
    def test_latency_target(self):
        try:
            import onnxruntime as ort
            assert ort.get_available_providers() is not None
        except ImportError:
            pytest.skip("onnxruntime not installed")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
