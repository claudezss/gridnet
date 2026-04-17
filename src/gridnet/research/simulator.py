"""
Toy transmission grid simulator for ML dataset generation.
Physics: simplified iterative voltage propagation + impedance-weighted branch flows.
Not real power flow — designed to produce graph-structured (topology, injections) -> (V, loading) data.
"""

from __future__ import annotations
import random

import numpy as np
import networkx as nx
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class Bus:
    bus_id: int
    bus_type: str  # "slack" | "PQ" | "PV"
    P: float = 0.0  # net injection (gen - load), pu
    Q: float = 0.0  # reactive injection, pu
    source_v: float = 1.0  # only used if slack or PV
    # outputs
    V: float = 1.0
    angle: float = 0.0  # pseudo-angle, not real theta


@dataclass
class Branch:
    branch_id: int
    from_bus: int
    to_bus: int
    r_pu: float = 0.01
    x_pu: float = 0.05
    b_pu: float = 0.0
    rating: float = 1.0  # MVA pu
    in_service: bool = True
    # outputs
    S_pu: float = 0.0
    loading_pct: float = 0.0


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------


class GridSimulator:
    """
    Iterative voltage propagation simulator.

    Voltage update rule (per iteration, non-slack buses):
        V_i = sum(w_ij * V_j) / sum(w_ij)  +  delta_i
        where w_ij = 1 / x_ij  (in-service branches only)
              delta_i = -(P_i * rp + Q_i * rq)   injection correction

    Branch flow:
        S_ij = (|V_i - V_j| / x_ij) * flow_scale
        loading_pct = S_ij / rating * 100

    Angle (pseudo, for feature use only):
        angle_i = mean(angle_j + (P_i - P_j) * x_ij)  over neighbors
    """

    def __init__(
        self,
        buses: list[Bus],
        branches: list[Branch],
        n_iter: int = 30,
        rp: float = 0.08,  # P -> V droop sensitivity
        rq: float = 0.12,  # Q -> V droop sensitivity
        flow_scale: float = 0.25,
    ):
        self.buses: dict[int, Bus] = {b.bus_id: b for b in buses}
        self.branches: dict[int, Branch] = {b.branch_id: b for b in branches}
        self.n_iter = n_iter
        self.rp = rp
        self.rq = rq
        self.flow_scale = flow_scale
        self._build_graph()

    def _build_graph(self):
        self.G = nx.Graph()
        for bid, bus in self.buses.items():
            self.G.add_node(bid, bus=bus)
        for br in self.branches.values():
            if br.in_service:
                self.G.add_edge(
                    br.from_bus,
                    br.to_bus,
                    branch_id=br.branch_id,
                    x=br.x_pu,
                    r=br.r_pu,
                    w=1.0 / max(br.x_pu, 1e-4),
                )

    def _reset_voltages(self):
        for bus in self.buses.values():
            if bus.bus_type in ("slack", "PV"):
                bus.V = bus.source_v
            else:
                bus.V = 1.0
            bus.angle = 0.0

    def solve(self) -> dict:
        """Run iterative propagation, return state dict."""
        self._reset_voltages()

        slack_buses = [b for b in self.buses.values() if b.bus_type == "slack"]
        if not slack_buses:
            raise ValueError("No slack bus defined")

        for _ in range(self.n_iter):
            new_V = {}
            new_angle = {}

            for bid, bus in self.buses.items():
                if bus.bus_type == "slack":
                    new_V[bid] = bus.source_v
                    new_angle[bid] = 0.0
                    continue

                nbrs = list(self.G.neighbors(bid))
                if not nbrs:
                    # islanded bus
                    new_V[bid] = 0.0
                    new_angle[bid] = 0.0
                    continue

                w_sum = 0.0
                wV_sum = 0.0
                angle_sum = 0.0
                for nb in nbrs:
                    edata = self.G[bid][nb]
                    w = edata["w"]
                    w_sum += w
                    wV_sum += w * self.buses[nb].V
                    # pseudo-angle: propagate angle + P*x drop
                    x = edata["x"]
                    angle_sum += self.buses[nb].angle + (bus.P - self.buses[nb].P) * x

                v_nb_mean = wV_sum / w_sum
                # injection correction: load (negative P) drags V down
                delta = -(bus.P * self.rp + bus.Q * self.rq)
                v_new = v_nb_mean + delta

                if bus.bus_type == "PV":
                    v_new = bus.source_v  # PV pins magnitude

                new_V[bid] = float(np.clip(v_new, 0.5, 1.2))
                new_angle[bid] = angle_sum / len(nbrs)

            for bid in self.buses:
                self.buses[bid].V = new_V[bid]
                self.buses[bid].angle = new_angle[bid]

        self._compute_branch_flows()
        return self._state_dict()

    def _compute_branch_flows(self):
        for br in self.branches.values():
            if not br.in_service:
                br.S_pu = 0.0
                br.loading_pct = 0.0
                continue
            Vi = self.buses[br.from_bus].V
            Vj = self.buses[br.to_bus].V
            ai = self.buses[br.from_bus].angle
            aj = self.buses[br.to_bus].angle
            dV = abs(Vi - Vj)
            dA = abs(ai - aj)
            # S ~ dV/x (voltage-driven) + dA/x (angle-driven)
            S = (
                dV / max(br.x_pu, 1e-4) + dA / max(br.x_pu, 1e-4) * 0.3
            ) * self.flow_scale
            br.S_pu = float(S)
            br.loading_pct = float(S / max(br.rating, 1e-6) * 100.0)

    def _state_dict(self) -> dict:
        return {
            "buses": {
                bid: {
                    "V": b.V,
                    "angle": b.angle,
                    "P": b.P,
                    "Q": b.Q,
                    "type": b.bus_type,
                }
                for bid, b in self.buses.items()
            },
            "branches": {
                brid: {
                    "S_pu": br.S_pu,
                    "loading_pct": br.loading_pct,
                    "in_service": br.in_service,
                    "from": br.from_bus,
                    "to": br.to_bus,
                    "x": br.x_pu,
                    "r": br.r_pu,
                    "rating": br.rating,
                }
                for brid, br in self.branches.items()
            },
        }

    def apply_contingency(self, branch_ids: list[int]):
        """Drop branches, rebuild graph, re-solve."""
        for bid in branch_ids:
            self.branches[bid].in_service = False
        self._build_graph()
        return self.solve()

    def restore_all(self):
        for br in self.branches.values():
            br.in_service = True
        self._build_graph()


# ---------------------------------------------------------------------------
# Topology generator — meshed transmission-like
# ---------------------------------------------------------------------------


def generate_transmission_topology(
    n_buses: int = 50,
    k_neighbors: int = None,  # WS base connectivity (auto-scaled if None)
    rewire_prob: float = 0.25,  # WS rewiring
    extra_ties: int = None,  # additional random tie lines (auto-scaled if None)
    target_avg_degree: float = 2.8,  # transmission ~2.5-3.5; caps edge density
    seed: int = None,
) -> tuple[list[Bus], list[Branch]]:
    """
    Build a meshed transmission-like network.
    Scales correctly for small (3-10 buses) through large (1000+) networks.
    Density is capped by target_avg_degree to avoid over-meshing small graphs.
    """
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    if n_buses < 3:
        raise ValueError("n_buses must be >= 3")

    # k=2 base (simple ring); extra ties build mesh up to target_avg_degree.
    # This decouples base connectivity from density control.
    if k_neighbors is None:
        k_neighbors = 2
    k_neighbors = min(k_neighbors, n_buses - 1)
    if k_neighbors % 2 != 0:
        k_neighbors = max(2, k_neighbors - 1)

    # --- base topology ---
    ws = nx.watts_strogatz_graph(n_buses, k_neighbors, rewire_prob, seed=seed)
    if not nx.is_connected(ws):
        components = list(nx.connected_components(ws))
        for i in range(len(components) - 1):
            a = rng.choice(list(components[i]))
            b = rng.choice(list(components[i + 1]))
            ws.add_edge(a, b)

    # density cap: max edges = n * avg_degree / 2
    max_edges = int(n_buses * target_avg_degree / 2)

    if extra_ties is None:
        # target_avg_degree - k_neighbors gives the degree to add via tie lines
        extra_ties = max(1, int((target_avg_degree - k_neighbors) * n_buses / 2))

    nodes = list(ws.nodes())
    added = 0
    attempts = 0
    while (
        added < extra_ties
        and ws.number_of_edges() < max_edges
        and attempts < extra_ties * 30
    ):
        a, b = rng.sample(nodes, 2)
        if not ws.has_edge(a, b):
            ws.add_edge(a, b)
            added += 1
        attempts += 1

    # --- buses ---
    buses = []
    slack_id = 0  # bus 0 is slack

    # bus type mix (of non-slack buses):
    #   ~15% PV  (generator, voltage-controlled)
    #   ~35% PQ  (load bus)
    #   ~50% passive (transit/junction, P=Q=0)
    non_slack = [i for i in range(1, n_buses)]
    n_gen = max(1, int(n_buses * 0.15))
    n_load = max(1, int(n_buses * 0.35))
    shuffled = non_slack.copy()
    rng.shuffle(shuffled)
    gen_ids = set(shuffled[:n_gen])
    load_ids = set(shuffled[n_gen : n_gen + n_load])
    # remaining are passive

    for i in range(n_buses):
        if i == slack_id:
            bus = Bus(bus_id=i, bus_type="slack", P=0.0, Q=0.0, source_v=1.0)
        elif i in gen_ids:
            P_gen = float(np_rng.uniform(0.3, 1.0))
            bus = Bus(
                bus_id=i,
                bus_type="PV",
                P=P_gen,
                Q=0.0,
                source_v=float(np_rng.uniform(0.98, 1.02)),
            )
        elif i in load_ids:
            P_load = float(-np_rng.uniform(0.05, 0.4))
            Q_load = float(-np_rng.uniform(0.01, 0.15))
            bus = Bus(bus_id=i, bus_type="PQ", P=P_load, Q=Q_load)
        else:
            bus = Bus(bus_id=i, bus_type="passive", P=0.0, Q=0.0)
        buses.append(bus)

    # --- branches ---
    branches = []
    for brid, (u, v) in enumerate(ws.edges()):
        x = float(np_rng.uniform(0.02, 0.15))
        r = float(np_rng.uniform(0.005, x * 0.4))
        b = float(np_rng.uniform(0.0, 0.05))
        rating = float(np_rng.uniform(1.2, 3.0))
        branches.append(
            Branch(
                branch_id=brid,
                from_bus=u,
                to_bus=v,
                r_pu=r,
                x_pu=x,
                b_pu=b,
                rating=rating,
            )
        )

    return buses, branches


# ---------------------------------------------------------------------------
# Dataset generator
# ---------------------------------------------------------------------------


def generate_dataset(
    n_samples: int = 1000,
    n_buses: int = 50,
    n_contingency_lines: int = 1,  # 1 or 2
    seed: int = 42,
) -> list[dict]:
    """
    For each sample:
      1. Generate topology (or reuse fixed)
      2. Perturb injections
      3. Solve base case
      4. Apply N-k contingency, solve again
      5. Store (features, base_state, post_contingency_state, contingency_mask)
    """
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    dataset = []

    # fixed topology, vary injections + contingency
    buses, branches = generate_transmission_topology(n_buses=n_buses, seed=seed)
    non_slack_branch_ids = [br.branch_id for br in branches]

    for sample_idx in range(n_samples):
        # --- perturb injections ---
        for bus in buses:
            if bus.bus_type == "PQ":
                bus.P = float(-np_rng.uniform(0.05, 0.5))
                bus.Q = float(-np_rng.uniform(0.01, 0.2))
            elif bus.bus_type == "PV":
                bus.P = float(np_rng.uniform(0.2, 1.0))

        sim = GridSimulator(buses, branches)

        # base case
        base_state = sim.solve()

        # contingency: drop k lines
        k = (
            n_contingency_lines
            if isinstance(n_contingency_lines, int)
            else rng.choice([1, 2])
        )
        dropped = rng.sample(non_slack_branch_ids, k)

        post_state = sim.apply_contingency(dropped)
        sim.restore_all()

        dataset.append(
            {
                "sample_id": sample_idx,
                "contingency_branches": dropped,
                "base": base_state,
                "post": post_state,
            }
        )

    return dataset


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Topology generation ===")
    buses, branches = generate_transmission_topology(n_buses=20, seed=0)
    print(
        f"  Buses: {len(buses)}  slack=1  PV={sum(1 for b in buses if b.bus_type == 'PV')}  PQ={sum(1 for b in buses if b.bus_type == 'PQ')}  passive={sum(1 for b in buses if b.bus_type == 'passive')}"
    )
    print(f"  Branches: {len(branches)}")

    print("\n=== Base case solve ===")
    sim = GridSimulator(buses, branches)
    state = sim.solve()
    vs = [v["V"] for v in state["buses"].values()]
    ls = [v["loading_pct"] for v in state["branches"].values()]
    print(f"  V range: [{min(vs):.3f}, {max(vs):.3f}]")
    print(f"  Loading range: [{min(ls):.1f}%, {max(ls):.1f}%]")

    print("\n=== N-1 contingency (drop branch 0) ===")
    post = sim.apply_contingency([0])
    vs_post = [v["V"] for v in post["buses"].values()]
    ls_post = [v["loading_pct"] for v in post["branches"].values()]
    print(f"  V range: [{min(vs_post):.3f}, {max(vs_post):.3f}]")
    print(f"  Loading range: [{min(ls_post):.1f}%, {max(ls_post):.1f}%]")
    print(
        f"  Max loading increase: {max(ls_post[i] - ls[i] for i in range(len(ls))):.1f}%"
    )

    print("\n=== Dataset generation (100 samples) ===")
    ds = generate_dataset(n_samples=100, n_buses=50, n_contingency_lines=1, seed=42)
    print(f"  Samples: {len(ds)}")
    sample = ds[0]
    print(f"  Sample keys: {list(sample.keys())}")
    print(f"  Contingency branches: {sample['contingency_branches']}")

    base_v = [v["V"] for v in sample["base"]["buses"].values()]
    post_v = [v["V"] for v in sample["post"]["buses"].values()]
    dv = [abs(a - b) for a, b in zip(base_v, post_v)]
    print(f"  Max |ΔV| from contingency: {max(dv):.4f} pu")
    print("\nDone.")
