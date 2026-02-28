import logging, time, random, math
from typing import Dict

log = logging.getLogger(__name__)


MOBILITY_SPEEDS = {"stationary": 0, "pedestrian": 3, "vehicular": 60}


class TrafficModel:

    def __init__(self, ran_sim, cfg):
        self.sim = ran_sim
        self.cfg = cfg
        self.running = False
        self._time = 0.0

        self._cell_load = {cid: random.uniform(0.3, 0.6) for cid in ran_sim.cells}

        self._event_active = False
        self._event_cell = None
        self._event_end = 0.0

    def start(self):
        self.running = True
        log.info("5G Traffic model started")

    def stop(self):
        self.running = False
        log.info("5G Traffic model stopped")

    def step(self, dt_s=0.01):
        self._time += dt_s

        for cid in self._cell_load:
            mu = 0.5
            theta = 0.1
            sigma = 0.08
            dW = random.gauss(0, math.sqrt(dt_s))
            self._cell_load[cid] += theta * (mu - self._cell_load[cid]) * dt_s + sigma * dW
            self._cell_load[cid] = max(0.05, min(0.98, self._cell_load[cid]))


        if not self._event_active and random.random() < 0.001:
            self._event_active = True
            self._event_cell = random.choice(list(self._cell_load.keys()))
            self._event_end = self._time + random.uniform(3, 10)
            log.debug(f"Flash crowd on {self._event_cell}")
        if self._event_active:
            self._cell_load[self._event_cell] = min(0.98, self._cell_load[self._event_cell] + 0.3)
            if self._time > self._event_end:
                self._event_active = False


        for ue in self.sim.ues.values():
            if not ue.connected:
                continue
            cell = self.sim.cells.get(ue.serving_cell)
            if not cell:
                continue
            load = self._cell_load.get(cell.cell_id, 0.5)

            sharing_factor = max(0.05, 1.0 - load * 0.8)
            ue.dl_throughput_kbps *= sharing_factor
            ue.ul_throughput_kbps *= sharing_factor

            ue.pdcp_delay_ms += load * 10 * random.uniform(0.5, 1.5)
            ue.pdcp_delay_ms = max(0.5, min(200, ue.pdcp_delay_ms))

            if ue.profile == "eMBB_premium":
                ue.buffer_occupancy_bytes = int(50000 * load + random.gauss(0, 5000))
            elif ue.profile == "URLLC":
                ue.buffer_occupancy_bytes = int(500 * load + random.gauss(0, 50))
            elif ue.profile == "mMTC_IoT":
                ue.buffer_occupancy_bytes = random.randint(0, 100) if random.random() < 0.01 else 0
            else:
                ue.buffer_occupancy_bytes = int(20000 * load + random.gauss(0, 3000))

    def get_cell_load(self, cell_id: str) -> float:
        return self._cell_load.get(cell_id, 0.5)

    def get_all_loads(self) -> Dict[str, float]:
        return dict(self._cell_load)

    def inject_load_spike(self, cell_id: str, duration_s: float = 5.0):
        self._event_active = True
        self._event_cell = cell_id
        self._event_end = self._time + duration_s
