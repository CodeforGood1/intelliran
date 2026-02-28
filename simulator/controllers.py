import logging, time, random, threading
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional

log = logging.getLogger(__name__)


@dataclass
class XAppAction:
    timestamp: float
    xapp: str
    action_type: str
    target_cell: str
    target_ue: Optional[str] = None
    dimension: str = ""
    value: float = 0.0
    priority: int = 0

    def to_dict(self):
        return asdict(self)


class TrafficSteeringXApp:

    def __init__(self, cfg, ran_sim):
        self.sim = ran_sim
        self.name = "ts-xapp"
        self.priority = 1
        self.rsrp_threshold = -100
        self.hysteresis = 3

    def decide(self, cell_snapshots: Dict, traffic_loads: Dict) -> List[XAppAction]:
        actions = []
        for ue in self.sim.ues.values():
            if not ue.connected:
                continue
            if ue.rsrp_dbm < self.rsrp_threshold:

                best_cell, best_rsrp = None, -999
                for cid, cell in self.sim.cells.items():
                    if cid == ue.serving_cell:
                        continue
                    import math
                    dist = math.sqrt((ue.x - cell.pos_x)**2 + (ue.y - cell.pos_y)**2)
                    est_rsrp = 43 - (28 + 22*math.log10(max(dist,10)) + 20*math.log10(3.5))
                    if est_rsrp > best_rsrp:
                        best_rsrp = est_rsrp
                        best_cell = cid
                if best_cell and best_rsrp > ue.rsrp_dbm + self.hysteresis:
                    actions.append(XAppAction(
                        timestamp=time.time(), xapp=self.name,
                        action_type="handover", target_cell=best_cell,
                        target_ue=ue.ue_id, dimension="ho",
                        value=best_rsrp - ue.rsrp_dbm, priority=self.priority))

            if ue.profile == "eMBB_premium" and ue.sinr_db > 20:
                if random.random() < 0.05:
                    actions.append(XAppAction(
                        timestamp=time.time(), xapp=self.name,
                        action_type="carrier_aggregation", target_cell=ue.serving_cell,
                        target_ue=ue.ue_id, dimension="carrier",
                        value=1.0, priority=self.priority))
        return actions


class QoSOptimizationXApp:

    def __init__(self, cfg, ran_sim):
        self.sim = ran_sim
        self.name = "qos-xapp"
        self.priority = 2

    def decide(self, cell_snapshots: Dict, traffic_loads: Dict) -> List[XAppAction]:
        actions = []
        for cid, snap in cell_snapshots.items():
            cell = self.sim.cells[cid]

            urllc_ues = [self.sim.ues[uid] for uid in cell.connected_ues
                        if uid in self.sim.ues and self.sim.ues[uid].profile == "URLLC"]
            if urllc_ues and snap["prb_util_dl"] > 0.7:
                actions.append(XAppAction(
                    timestamp=time.time(), xapp=self.name,
                    action_type="prb_reservation", target_cell=cid,
                    dimension="prb", value=20, priority=self.priority))

            poor_ues = [self.sim.ues[uid] for uid in cell.connected_ues
                       if uid in self.sim.ues and self.sim.ues[uid].sinr_db < 5]
            if poor_ues:
                actions.append(XAppAction(
                    timestamp=time.time(), xapp=self.name,
                    action_type="power_boost", target_cell=cid,
                    dimension="power", value=3.0, priority=self.priority))

            if snap["avg_bler"] > 0.1:
                actions.append(XAppAction(
                    timestamp=time.time(), xapp=self.name,
                    action_type="mcs_override", target_cell=cid,
                    dimension="mcs", value=-2, priority=self.priority))

            if snap["prb_util_dl"] > 0.85:
                actions.append(XAppAction(
                    timestamp=time.time(), xapp=self.name,
                    action_type="scheduling_weight", target_cell=cid,
                    dimension="scheduling", value=1.5, priority=self.priority))
        return actions


class MLBXApp:

    def __init__(self, cfg, ran_sim):
        self.sim = ran_sim
        self.name = "mlb-xapp"
        self.priority = 3
        self.load_diff_threshold = 0.2

    def decide(self, cell_snapshots: Dict, traffic_loads: Dict) -> List[XAppAction]:
        actions = []
        loads = {cid: snap["prb_util_dl"] for cid, snap in cell_snapshots.items()}
        avg_load = sum(loads.values()) / max(len(loads), 1)
        for cid, load in loads.items():
            if load > avg_load + self.load_diff_threshold:

                actions.append(XAppAction(
                    timestamp=time.time(), xapp=self.name,
                    action_type="cio_adjust", target_cell=cid,
                    dimension="ho", value=2.0, priority=self.priority))

                actions.append(XAppAction(
                    timestamp=time.time(), xapp=self.name,
                    action_type="prb_cap", target_cell=cid,
                    dimension="prb", value=0.8, priority=self.priority))
            elif load < avg_load - self.load_diff_threshold:

                actions.append(XAppAction(
                    timestamp=time.time(), xapp=self.name,
                    action_type="cio_adjust", target_cell=cid,
                    dimension="ho", value=-2.0, priority=self.priority))

                actions.append(XAppAction(
                    timestamp=time.time(), xapp=self.name,
                    action_type="ho_threshold", target_cell=cid,
                    dimension="ho", value=-1.0, priority=self.priority))
        return actions


class EnergySavingXApp:

    def __init__(self, cfg, ran_sim):
        self.sim = ran_sim
        self.name = "es-xapp"
        self.priority = 4
        self.low_load_threshold = 0.25

    def decide(self, cell_snapshots: Dict, traffic_loads: Dict) -> List[XAppAction]:
        actions = []
        for cid, snap in cell_snapshots.items():
            load = snap["prb_util_dl"]
            if load < self.low_load_threshold:

                actions.append(XAppAction(
                    timestamp=time.time(), xapp=self.name,
                    action_type="mimo_layer_reduce", target_cell=cid,
                    dimension="power", value=2, priority=self.priority))

                actions.append(XAppAction(
                    timestamp=time.time(), xapp=self.name,
                    action_type="dtx_drx", target_cell=cid,
                    dimension="sleep", value=1.0, priority=self.priority))

                if load < 0.1 and len(self.sim.cells) > 1:
                    actions.append(XAppAction(
                        timestamp=time.time(), xapp=self.name,
                        action_type="carrier_shutdown", target_cell=cid,
                        dimension="carrier", value=1.0, priority=self.priority))
            elif load > 0.6:

                actions.append(XAppAction(
                    timestamp=time.time(), xapp=self.name,
                    action_type="mimo_layer_reduce", target_cell=cid,
                    dimension="power", value=-2, priority=self.priority))
        return actions


CONFLICT_DIMENSIONS = {
    ("handover", "cio_adjust"): "ho",
    ("handover", "ho_threshold"): "ho",
    ("prb_reservation", "prb_cap"): "prb",
    ("power_boost", "mimo_layer_reduce"): "power",
    ("power_boost", "carrier_shutdown"): "power",
    ("scheduling_weight", "prb_cap"): "prb",
    ("carrier_aggregation", "carrier_shutdown"): "carrier",
    ("mcs_override", "scheduling_weight"): "scheduling",
}


def detect_action_conflicts(actions: List[XAppAction]) -> List[dict]:
    conflicts = []
    cell_actions = {}
    for a in actions:
        key = a.target_cell
        cell_actions.setdefault(key, []).append(a)
    for cell_id, cell_acts in cell_actions.items():
        for i in range(len(cell_acts)):
            for j in range(i+1, len(cell_acts)):
                a1, a2 = cell_acts[i], cell_acts[j]
                if a1.xapp == a2.xapp:
                    continue
                pair = tuple(sorted([a1.action_type, a2.action_type]))
                dim_conflict = pair in CONFLICT_DIMENSIONS or (
                    a1.dimension == a2.dimension and a1.dimension != ""
                )
                value_conflict = (
                    (a1.value > 0 and a2.value < 0) or
                    (a1.value < 0 and a2.value > 0)
                )
                if dim_conflict or value_conflict:
                    conflicts.append({
                        "cell_id": cell_id,
                        "xapp_a": a1.xapp, "action_a": a1.action_type,
                        "xapp_b": a2.xapp, "action_b": a2.action_type,
                        "dimension": a1.dimension or a2.dimension,
                        "severity": abs(a1.value) + abs(a2.value),
                    })
    return conflicts


class ControllerRunner:

    def __init__(self, ran_sim, traffic_model, cfg):
        self.sim = ran_sim
        self.traffic = traffic_model
        self.cfg = cfg
        self.ts = TrafficSteeringXApp(cfg, ran_sim)
        self.qos = QoSOptimizationXApp(cfg, ran_sim)
        self.mlb = MLBXApp(cfg, ran_sim)
        self.es = EnergySavingXApp(cfg, ran_sim)
        self.action_log: List[XAppAction] = []
        self.conflict_log: List[dict] = []
        self.running = False
        self._lock = threading.Lock()
        self._step_count = 0
        self.control_interval_s = cfg["ric"]["control_loop_ms"] / 1000.0

    def _run_loop(self):
        while self.running:
            time.sleep(self.control_interval_s)
            if not self.running:
                break
            try:
                self._control_step()
            except Exception as e:
                log.error(f"Control loop error: {e}")

    def _control_step(self):

        snapshots = {}
        for cid in self.sim.cells:
            snapshots[cid] = self.sim.get_cell_snapshot(cid)
        loads = self.traffic.get_all_loads()

        ts_actions = self.ts.decide(snapshots, loads)
        qos_actions = self.qos.decide(snapshots, loads)
        mlb_actions = self.mlb.decide(snapshots, loads)
        es_actions = self.es.decide(snapshots, loads)
        all_actions = ts_actions + qos_actions + mlb_actions + es_actions

        conflicts = detect_action_conflicts(all_actions)

        for a in all_actions:
            self._apply_action(a)
        with self._lock:
            self.action_log.extend(all_actions)
            self.conflict_log.extend(conflicts)
        self._step_count += 1

    def _apply_action(self, action: XAppAction):
        cell = self.sim.cells.get(action.target_cell)
        if not cell:
            return
        if action.action_type == "handover" and action.target_ue:
            self.sim.trigger_handover(action.target_ue, action.target_cell)
        elif action.action_type == "cio_adjust":
            cell.cio_db += action.value
            cell.cio_db = max(-10, min(10, cell.cio_db))
        elif action.action_type == "prb_cap":
            pass
        elif action.action_type == "prb_reservation":
            pass
        elif action.action_type == "power_boost":
            cell.tx_power_dbm = min(49, cell.tx_power_dbm + action.value)
        elif action.action_type == "mimo_layer_reduce":
            cell.mimo_layers_active = max(1, min(cell.max_layers,
                                                  cell.mimo_layers_active + int(action.value)))
        elif action.action_type == "scheduling_weight":
            cell.scheduling_weight = max(0.1, min(5.0, action.value))
        elif action.action_type == "carrier_shutdown":
            cell.carrier_on = not bool(action.value > 0)
        elif action.action_type == "dtx_drx":
            cell.sleep_mode = bool(action.value > 0)

    def get_recent_actions(self, since_ts: float) -> List[XAppAction]:
        with self._lock:
            return [a for a in self.action_log if a.timestamp > since_ts]

    def get_all_actions(self) -> List[XAppAction]:
        with self._lock:
            return list(self.action_log)

    def get_recent_conflicts(self, since_ts: float) -> List[dict]:
        with self._lock:
            return [c for c in self.conflict_log if True]

    def get_action_summary(self, since_ts: float) -> dict:
        recent = self.get_recent_actions(since_ts)
        summary = {}
        for a in recent:
            key = f"{a.xapp}_{a.action_type}"
            summary[key] = summary.get(key, 0) + 1
        return summary

    def start(self):
        self.running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        log.info("xApp Controller Runner started (4 xApps)")

    def stop(self):
        self.running = False
        if hasattr(self, "_thread"):
            self._thread.join(timeout=2)
        log.info(f"Controller Runner stopped. {len(self.action_log)} actions, {len(self.conflict_log)} conflicts logged")
