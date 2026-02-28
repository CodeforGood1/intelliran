import logging, yaml, os, math, random
from dataclasses import dataclass, field
from typing import Dict, List, Optional

log = logging.getLogger(__name__)
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_config():
    cfg_path = os.path.join(BASE, "config", "network.yaml")
    with open(cfg_path) as f:
        return yaml.safe_load(f)


@dataclass
class UEState:
    ue_id: str
    profile: str
    serving_cell: str
    rsrp_dbm: float = -85.0
    rsrq_db: float = -10.0
    sinr_db: float = 15.0
    cqi: int = 12
    mcs: int = 20
    rank: int = 2
    dl_throughput_kbps: float = 50000.0
    ul_throughput_kbps: float = 15000.0
    bler: float = 0.01
    rlc_retx_rate: float = 0.005
    pdcp_delay_ms: float = 5.0
    buffer_occupancy_bytes: int = 10000
    mobility_kmh: float = 3.0
    x: float = 0.0
    y: float = 0.0
    connected: bool = True
    qfi: int = 9
    drx_cycle_ms: int = 40

    def update_channel(self, cell_pos, fading_std=4.0, doppler_hz=5.0):
        dist = math.sqrt((self.x - cell_pos[0])**2 + (self.y - cell_pos[1])**2)
        dist = max(dist, 10.0)

        fc_ghz = 3.5
        pl = 28.0 + 22.0 * math.log10(dist) + 20.0 * math.log10(fc_ghz)
        shadow = random.gauss(0, fading_std)
        fast = random.gauss(0, 2.0)
        self.rsrp_dbm = 43.0 - pl + shadow + fast
        self.rsrp_dbm = max(-140.0, min(-44.0, self.rsrp_dbm))
        self.rsrq_db = self.rsrp_dbm / 10.0 + random.gauss(0, 1.0)
        self.rsrq_db = max(-20.0, min(-3.0, self.rsrq_db))
        self.sinr_db = self.rsrp_dbm + 120 + random.gauss(0, 3.0)
        self.sinr_db = max(-5.0, min(35.0, self.sinr_db))

        self.cqi = max(1, min(15, int((self.sinr_db + 5) / 2.7)))

        self.mcs = max(0, min(28, int(self.cqi * 1.9)))

        self.rank = min(4, max(1, int(self.sinr_db / 10) + 1))

        sinr_target = 2.0 + self.mcs * 1.1
        self.bler = min(0.5, max(0.0001, 0.1 * math.exp(-(self.sinr_db - sinr_target) * 0.3)))

        bw_hz = 100e6
        eff = 0.6
        capacity_bps = bw_hz * eff * math.log2(1 + 10**(self.sinr_db/10)) * self.rank
        self.dl_throughput_kbps = capacity_bps / 1000 * (1 - self.bler)
        self.ul_throughput_kbps = self.dl_throughput_kbps * 0.3
        self.rlc_retx_rate = self.bler * 0.8
        self.pdcp_delay_ms = max(0.5, 5.0 + 50.0 * self.bler + random.gauss(0, 1))

    def move(self, dt_s, boundary=1000.0):
        speed_ms = self.mobility_kmh / 3.6
        angle = random.uniform(0, 2 * math.pi)
        self.x += speed_ms * dt_s * math.cos(angle)
        self.y += speed_ms * dt_s * math.sin(angle)
        self.x = max(-boundary, min(boundary, self.x))
        self.y = max(-boundary, min(boundary, self.y))


@dataclass
class CellState:
    cell_id: str
    pci: int
    band: str
    total_prbs: int
    scs_khz: int
    tx_power_dbm: float
    antenna_ports: int
    max_layers: int
    max_ue_capacity: int
    cu_id: str
    du_id: str
    ru_id: str

    connected_ues: List[str] = field(default_factory=list)
    prb_used_dl: int = 0
    prb_used_ul: int = 0
    avg_sinr_db: float = 15.0
    avg_cqi: float = 12.0
    dl_throughput_mbps: float = 0.0
    ul_throughput_mbps: float = 0.0
    rrc_conn_established: int = 0
    rrc_conn_attempted: int = 0
    ho_success: int = 0
    ho_attempted: int = 0
    bler_dl: float = 0.01
    active_beams: int = 8
    csi_rs_period_ms: int = 20
    pos_x: float = 0.0
    pos_y: float = 0.0
    sleep_mode: bool = False
    carrier_on: bool = True
    mimo_layers_active: int = 4
    scheduling_weight: float = 1.0
    cio_db: float = 0.0

    def prb_util_dl(self):
        return self.prb_used_dl / max(self.total_prbs, 1)

    def prb_util_ul(self):
        return self.prb_used_ul / max(self.total_prbs, 1)


class RANSimulator:

    def __init__(self, cfg):
        self.cfg = cfg
        self.cells: Dict[str, CellState] = {}
        self.ues: Dict[str, UEState] = {}
        self.running = False
        self._init_cells()
        self._init_ues()

    def _init_cells(self):
        cell_positions = [(0, 0), (500, 433), (-500, 433)]
        for i, cell_cfg in enumerate(self.cfg["ran"]["cells"]):
            px, py = cell_positions[i % len(cell_positions)]
            cell = CellState(
                cell_id=cell_cfg["cell_id"],
                pci=cell_cfg["pci"],
                band=cell_cfg["band"],
                total_prbs=cell_cfg["total_prbs"],
                scs_khz=cell_cfg["scs_khz"],
                tx_power_dbm=cell_cfg["tx_power_dbm"],
                antenna_ports=cell_cfg["antenna_ports"],
                max_layers=cell_cfg["max_layers"],
                max_ue_capacity=cell_cfg["max_ue_capacity"],
                cu_id=cell_cfg["cu_id"],
                du_id=cell_cfg["du_id"],
                ru_id=cell_cfg["ru_id"],
                pos_x=px, pos_y=py,
                mimo_layers_active=min(4, cell_cfg["max_layers"]),
            )
            self.cells[cell.cell_id] = cell

    def _init_ues(self):
        ue_idx = 0
        cell_ids = list(self.cells.keys())
        for profile in self.cfg["ue_profiles"]:
            mob_speed = {"stationary": 0, "pedestrian": 3, "vehicular": 60}.get(profile["mobility"], 3)
            for _ in range(profile["count"]):
                ue_id = f"ue-{ue_idx:04d}"
                cell_id = cell_ids[ue_idx % len(cell_ids)]
                ue = UEState(
                    ue_id=ue_id, profile=profile["name"],
                    serving_cell=cell_id, qfi=profile["qfi"],
                    mobility_kmh=mob_speed + random.gauss(0, 1),
                    x=self.cells[cell_id].pos_x + random.gauss(0, 200),
                    y=self.cells[cell_id].pos_y + random.gauss(0, 200),
                )
                self.ues[ue_id] = ue
                self.cells[cell_id].connected_ues.append(ue_id)
                ue_idx += 1

    def start(self):
        self.running = True
        log.info(f"RAN Simulator started: {len(self.cells)} cells, {len(self.ues)} UEs")

    def stop(self):
        self.running = False
        log.info("RAN Simulator stopped")

    def step(self, dt_s=0.01):
        for ue in self.ues.values():
            if not ue.connected:
                continue
            ue.move(dt_s)
            cell = self.cells.get(ue.serving_cell)
            if cell:
                ue.update_channel((cell.pos_x, cell.pos_y),
                                  fading_std=4.0,
                                  doppler_hz=self.cfg["experiment"].get("doppler_hz", 5))

        for cell in self.cells.values():
            ues_in_cell = [self.ues[uid] for uid in cell.connected_ues
                           if uid in self.ues and self.ues[uid].connected]
            n = max(len(ues_in_cell), 1)
            cell.avg_sinr_db = sum(u.sinr_db for u in ues_in_cell) / n if ues_in_cell else 15.0
            cell.avg_cqi = sum(u.cqi for u in ues_in_cell) / n if ues_in_cell else 12.0
            cell.dl_throughput_mbps = sum(u.dl_throughput_kbps for u in ues_in_cell) / 1000
            cell.ul_throughput_mbps = sum(u.ul_throughput_kbps for u in ues_in_cell) / 1000
            cell.bler_dl = sum(u.bler for u in ues_in_cell) / n if ues_in_cell else 0.01

            load = len(ues_in_cell) / max(cell.max_ue_capacity, 1)
            cell.prb_used_dl = int(cell.total_prbs * min(load * 1.5 + random.gauss(0, 0.05), 1.0))
            cell.prb_used_ul = int(cell.total_prbs * min(load * 0.8 + random.gauss(0, 0.03), 1.0))
            cell.prb_used_dl = max(0, min(cell.total_prbs, cell.prb_used_dl))
            cell.prb_used_ul = max(0, min(cell.total_prbs, cell.prb_used_ul))
            cell.rrc_conn_established = len(ues_in_cell)
            cell.rrc_conn_attempted = len(ues_in_cell) + random.randint(0, 3)

    def get_cell_snapshot(self, cell_id: str) -> dict:
        cell = self.cells[cell_id]
        ues = [self.ues[uid] for uid in cell.connected_ues
               if uid in self.ues and self.ues[uid].connected]
        n = max(len(ues), 1)

        mcs_list = [u.mcs for u in ues] if ues else [15]
        qpsk = sum(1 for m in mcs_list if m <= 9) / n
        qam16 = sum(1 for m in mcs_list if 10 <= m <= 16) / n
        qam64 = sum(1 for m in mcs_list if 17 <= m <= 24) / n
        qam256 = sum(1 for m in mcs_list if m >= 25) / n
        return {
            "cell_id": cell.cell_id,
            "pci": cell.pci,
            "band": cell.band,
            "connected_ues": len(ues),
            "DRB.UEThpDl": cell.dl_throughput_mbps * 1000,
            "DRB.UEThpUl": cell.ul_throughput_mbps * 1000,
            "RRU.PrbUsedDl": cell.prb_used_dl,
            "RRU.PrbUsedUl": cell.prb_used_ul,
            "RRU.PrbAvailDl": cell.total_prbs,
            "RRU.PrbTotDl": cell.total_prbs,
            "prb_util_dl": cell.prb_util_dl(),
            "prb_util_ul": cell.prb_util_ul(),
            "L1M.RS-SINR": cell.avg_sinr_db,
            "L1M.RSRP": sum(u.rsrp_dbm for u in ues) / n if ues else -90.0,
            "L1M.RSRQ": sum(u.rsrq_db for u in ues) / n if ues else -12.0,
            "avg_cqi": cell.avg_cqi,
            "avg_mcs": sum(u.mcs for u in ues) / n if ues else 15,
            "avg_rank": sum(u.rank for u in ues) / n if ues else 2,
            "DRB.RlcSduDelayDl": sum(u.pdcp_delay_ms for u in ues) / n if ues else 5.0,
            "DRB.PacketLossRateDl": sum(u.bler for u in ues) / n * 100 if ues else 0.5,
            "DRB.PacketLossRateUl": sum(u.bler for u in ues) / n * 80 if ues else 0.4,
            "avg_bler": cell.bler_dl,
            "TB.TotNbrDl.Qpsk": qpsk,
            "TB.TotNbrDl.16Qam": qam16,
            "TB.TotNbrDl.64Qam": qam64,
            "TB.TotNbrDl.256Qam": qam256,
            "TB.ErrTotalNbrDl": cell.bler_dl,
            "RRC.ConnEstabSucc": cell.rrc_conn_established,
            "RRC.ConnEstabAtt": cell.rrc_conn_attempted,
            "HO.SuccOutInterEnb": cell.ho_success,
            "HO.AttOutInterEnb": cell.ho_attempted,
            "mimo_layers_active": cell.mimo_layers_active,
            "active_beams": cell.active_beams,
            "sleep_mode": int(cell.sleep_mode),
            "carrier_on": int(cell.carrier_on),
            "scheduling_weight": cell.scheduling_weight,
            "cio_db": cell.cio_db,
            "tx_power_dbm": cell.tx_power_dbm,
        }

    def trigger_handover(self, ue_id: str, target_cell: str):
        ue = self.ues.get(ue_id)
        if not ue or target_cell not in self.cells:
            return False
        src = ue.serving_cell
        if src == target_cell:
            return False
        self.cells[src].connected_ues = [u for u in self.cells[src].connected_ues if u != ue_id]
        self.cells[src].ho_attempted += 1
        if random.random() < 0.95:
            ue.serving_cell = target_cell
            self.cells[target_cell].connected_ues.append(ue_id)
            self.cells[src].ho_success += 1
            self.cells[target_cell].ho_success += 1
            return True
        else:
            self.cells[src].connected_ues.append(ue_id)
            return False
