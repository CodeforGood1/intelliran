import logging, time, os, csv, json, threading
from multiprocessing import Queue
from typing import Optional

log = logging.getLogger(__name__)
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TelemetryCollector:

    def __init__(self, ran_sim, traffic_model, controller_runner, cfg,
                 live_queue: Optional[Queue] = None):
        self.sim = ran_sim
        self.traffic = traffic_model
        self.controllers = controller_runner
        self.cfg = cfg
        self.poll_interval = cfg["telemetry"]["poll_interval_ms"] / 1000.0
        self.log_dir = os.path.join(BASE, cfg["telemetry"]["log_dir"])
        self.data_dir = os.path.join(BASE, cfg["telemetry"]["data_dir"])
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)
        self.csv_path = os.path.join(self.data_dir, "telemetry.csv")
        self.jsonl_path = os.path.join(self.log_dir, "telemetry.jsonl")
        self.running = False
        self.records = []
        self.live_queue = live_queue
        self._last_action_ts = 0.0
        self._csv_writer = None
        self._csv_file = None
        self._jsonl_file = None
        self._step = 0

    def _collect_one(self):
        ts = time.time()
        action_summary = self.controllers.get_action_summary(self._last_action_ts)
        self._last_action_ts = ts

        recent_conflicts = self.controllers.get_recent_conflicts(self._last_action_ts - 1.0)
        records = []
        for cid in self.sim.cells:
            snap = self.sim.get_cell_snapshot(cid)
            load = self.traffic.get_cell_load(cid)
            record = {
                "timestamp": ts,
                "step": self._step,
                "cell_id": snap["cell_id"],
                "pci": snap["pci"],
                "band": snap["band"],
                "connected_ues": snap["connected_ues"],

                "DRB.UEThpDl": snap["DRB.UEThpDl"],
                "DRB.UEThpUl": snap["DRB.UEThpUl"],
                "RRU.PrbUsedDl": snap["RRU.PrbUsedDl"],
                "RRU.PrbUsedUl": snap["RRU.PrbUsedUl"],
                "RRU.PrbAvailDl": snap["RRU.PrbAvailDl"],
                "prb_util_dl": snap["prb_util_dl"],
                "prb_util_ul": snap["prb_util_ul"],
                "L1M.RS-SINR": snap["L1M.RS-SINR"],
                "L1M.RSRP": snap["L1M.RSRP"],
                "L1M.RSRQ": snap["L1M.RSRQ"],
                "avg_cqi": snap["avg_cqi"],
                "avg_mcs": snap["avg_mcs"],
                "avg_rank": snap["avg_rank"],
                "DRB.RlcSduDelayDl": snap["DRB.RlcSduDelayDl"],
                "DRB.PacketLossRateDl": snap["DRB.PacketLossRateDl"],
                "DRB.PacketLossRateUl": snap["DRB.PacketLossRateUl"],
                "avg_bler": snap["avg_bler"],
                "TB.TotNbrDl.Qpsk": snap["TB.TotNbrDl.Qpsk"],
                "TB.TotNbrDl.16Qam": snap["TB.TotNbrDl.16Qam"],
                "TB.TotNbrDl.64Qam": snap["TB.TotNbrDl.64Qam"],
                "TB.TotNbrDl.256Qam": snap["TB.TotNbrDl.256Qam"],
                "TB.ErrTotalNbrDl": snap["TB.ErrTotalNbrDl"],
                "RRC.ConnEstabSucc": snap["RRC.ConnEstabSucc"],
                "RRC.ConnEstabAtt": snap["RRC.ConnEstabAtt"],
                "HO.SuccOutInterEnb": snap["HO.SuccOutInterEnb"],
                "HO.AttOutInterEnb": snap["HO.AttOutInterEnb"],

                "mimo_layers_active": snap["mimo_layers_active"],
                "active_beams": snap["active_beams"],
                "sleep_mode": snap["sleep_mode"],
                "carrier_on": snap["carrier_on"],
                "scheduling_weight": snap["scheduling_weight"],
                "cio_db": snap["cio_db"],
                "tx_power_dbm": snap["tx_power_dbm"],
                "cell_load": load,

                "ts_handover": action_summary.get(f"ts-xapp_handover", 0),
                "ts_carrier_agg": action_summary.get(f"ts-xapp_carrier_aggregation", 0),
                "qos_prb_reservation": action_summary.get(f"qos-xapp_prb_reservation", 0),
                "qos_power_boost": action_summary.get(f"qos-xapp_power_boost", 0),
                "qos_mcs_override": action_summary.get(f"qos-xapp_mcs_override", 0),
                "qos_sched_weight": action_summary.get(f"qos-xapp_scheduling_weight", 0),
                "mlb_cio_adjust": action_summary.get(f"mlb-xapp_cio_adjust", 0),
                "mlb_prb_cap": action_summary.get(f"mlb-xapp_prb_cap", 0),
                "mlb_ho_threshold": action_summary.get(f"mlb-xapp_ho_threshold", 0),
                "es_mimo_reduce": action_summary.get(f"es-xapp_mimo_layer_reduce", 0),
                "es_dtx_drx": action_summary.get(f"es-xapp_dtx_drx", 0),
                "es_carrier_shutdown": action_summary.get(f"es-xapp_carrier_shutdown", 0),

                "n_conflicts": len([c for c in recent_conflicts if c.get("cell_id") == cid]),
            }
            records.append(record)
        self._step += 1
        return records

    def _write_records(self, records):
        for record in records:
            if self._csv_writer is None:
                self._csv_file = open(self.csv_path, "w", newline="")
                self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=record.keys())
                self._csv_writer.writeheader()
            self._csv_writer.writerow(record)
            self._csv_file.flush()
            if self._jsonl_file is None:
                self._jsonl_file = open(self.jsonl_path, "w")
            self._jsonl_file.write(json.dumps(record, default=str) + "\n")
            self._jsonl_file.flush()

    def _run_loop(self):
        while self.running:
            try:
                records = self._collect_one()
                self.records.extend(records)
                self._write_records(records)
                if self.live_queue is not None:
                    for r in records:
                        try:
                            self.live_queue.put_nowait(r)
                        except Exception:
                            pass
            except Exception as e:
                log.error(f"Telemetry error: {e}")
            time.sleep(self.poll_interval)

    def start(self):
        self.running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        log.info(f"Telemetry collector started ({1/self.poll_interval:.0f} Hz)")

    def stop(self):
        self.running = False
        if hasattr(self, "_thread"):
            self._thread.join(timeout=2)
        if self._csv_file:
            self._csv_file.close()
        if self._jsonl_file:
            self._jsonl_file.close()
        log.info(f"Telemetry stopped. {len(self.records)} records collected")

    def get_records(self):
        return list(self.records)


if __name__ == "__main__":
    import yaml, sys, pandas as pd
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    sys.path.insert(0, BASE)
    from simulator.topology import load_config, RANSimulator
    from simulator.traffic import TrafficModel
    from simulator.controllers import ControllerRunner

    cfg = load_config()
    sim = RANSimulator(cfg)
    sim.start()
    traffic = TrafficModel(sim, cfg)
    traffic.start()
    controllers = ControllerRunner(sim, traffic, cfg)
    controllers.start()
    collector = TelemetryCollector(sim, traffic, controllers, cfg)
    collector.start()

    duration = min(cfg["experiment"]["duration_s"], 30)
    dt = cfg["telemetry"]["poll_interval_ms"] / 1000.0
    log.info(f"Collecting for {duration}s at {1/dt:.0f} Hz...")
    steps = int(duration / dt)
    for i in range(steps):
        sim.step(dt)
        traffic.step(dt)
        if (i+1) % (steps // 10) == 0:
            log.info(f"Progress: {(i+1)*100//steps}% ({len(collector.records)} records)")

    collector.stop()
    controllers.stop()
    traffic.stop()
    sim.stop()


    if collector.records:
        df = pd.DataFrame(collector.records)
        parquet_path = os.path.join(collector.data_dir, "telemetry.parquet")
        df.to_parquet(parquet_path, index=False)
        log.info(f"Done. {len(df)} records -> {parquet_path} ({df.shape[1]} cols)")
