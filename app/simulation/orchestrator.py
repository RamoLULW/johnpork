from typing import List, Dict
import base64
from io import BytesIO
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt


from app.core.types import (
    SimulationRequestDTO,
    SimulationResultDTO,
    SimulationStatsDTO,
    PlantDTO,
    FrameDTO,
    TractorStateDTO,
    TractorInternalState,
    TerrainDTO,
    CellType,
)
from app.env.environment import Environment
from app.env.blackboard import Blackboard
from app.learning.qlearning import QLearningPolicy
from app.agents.tractor import TractorAgent
from app.agents.interfaces import IHighLevelPolicy


def _figure_to_base64(fig) -> str:
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    data = buf.read()
    plt.close(fig)
    return base64.b64encode(data).decode("ascii")


def _build_dashboard_png(
    req: SimulationRequestDTO,
    ticks: List[int],
    remaining: List[int],
    fuel_series: Dict[int, List[float]],
    tractor_good: Dict[int, int],
    tractor_bad: Dict[int, int],
    env: Environment,
    total_plants: int,
) -> str:
    if not ticks:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.axis("off")
        return _figure_to_base64(fig)

    harvested_total = [total_plants - r for r in remaining]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle(f"Tractor Field Simulation (N={req.N}, tractors={req.num_tractors})", fontsize=16)

    ax = axes[0, 0]
    ax.plot(ticks, harvested_total, color="cyan", label="Total harvested")
    ax.set_title("Harvest Progress")
    ax.set_xlabel("Tick")
    ax.set_ylabel("Harvested plants")
    ax.grid(True)
    ax.legend()

    ax = axes[0, 1]
    ax.plot(ticks, remaining, color="yellow")
    ax.set_title("Remaining Plants")
    ax.set_xlabel("Tick")
    ax.set_ylabel("Plants left")
    ax.grid(True)

    ax = axes[0, 2]
    for tid, series in fuel_series.items():
        ax.plot(range(len(series)), series, label=f"T{tid}")
    ax.set_title("Fuel per Tractor")
    ax.set_xlabel("Step")
    ax.set_ylabel("Fuel")
    ax.grid(True)
    if fuel_series:
        ax.legend()

    ax = axes[1, 0]
    tractor_ids = sorted(fuel_series.keys())
    efficiencies = []
    labels = []
    for tid in tractor_ids:
        good = tractor_good.get(tid, 0)
        bad = tractor_bad.get(tid, 0)
        total_h = good + bad
        fs = fuel_series.get(tid, [])
        used = sum(max(0.0, fs[i] - fs[i + 1]) for i in range(len(fs) - 1)) if len(fs) > 1 else 0.0
        if used <= 0.0:
            used = 1.0
        eff = total_h / used
        efficiencies.append(eff)
        labels.append(f"T{tid}")
    if labels:
        ax.bar(labels, efficiencies, color="cyan")
    ax.set_title("Efficiency (harvest / fuel used)")
    ax.set_ylabel("Efficiency")
    ax.grid(True, axis="y")

    ax = axes[1, 1]
    type_counts = {
        CellType.EMPTY: 0,
        CellType.PLANT: 0,
        CellType.ROCK: 0,
        CellType.WATER: 0,
        CellType.MUD: 0,
        CellType.ROAD: 0,
        CellType.STATION: 0,
    }
    for pos, cell_type_name, cost, trans in env.iter_terrain():
        t = CellType[cell_type_name]
        type_counts[t] += 1
    labels_pie = []
    counts_pie = []
    colors_pie = []
    label_color = {
        CellType.EMPTY: ("Empty", "gray"),
        CellType.PLANT: ("Plant", "green"),
        CellType.ROCK: ("Rock", "red"),
        CellType.WATER: ("Water", "blue"),
        CellType.MUD: ("Mud", "saddlebrown"),
        CellType.ROAD: ("Road", "gold"),
        CellType.STATION: ("Station", "white"),
    }
    for t, c in type_counts.items():
        if c > 0:
            label, color = label_color[t]
            labels_pie.append(label)
            counts_pie.append(c)
            colors_pie.append(color)
    if counts_pie:
        ax.pie(counts_pie, labels=labels_pie, colors=colors_pie, autopct="%1.1f%%")
    ax.set_title("Terrain Distribution")

    axes[1, 2].axis("off")

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    return _figure_to_base64(fig)


def run_simulation(req: SimulationRequestDTO, policy: IHighLevelPolicy = None) -> SimulationResultDTO:
    env = Environment(
        N=req.N,
        p_good=req.p_good,
        seed=req.seed,
        obstacle_intensity=req.obstacle_intensity,
        dynamic_weather=req.dynamic_weather,
    )
    blackboard = Blackboard()
    for pos, is_good in env.iter_plants():
        blackboard.add_plant(pos)
    if policy is None:
        policy = QLearningPolicy(seed=req.seed) if req.use_qlearning else None

    tractors: List[TractorAgent] = []
    station_pos = env.nearest_station((0, 0))
    for i in range(req.num_tractors):
        x, y = station_pos
        internal = TractorInternalState(
            id=i,
            x=x,
            y=y,
            fuel=req.fuel_capacity,
            max_fuel=req.fuel_capacity,
            load=0,
            capacity=req.capacity,
        )
        blackboard.register_tractor(i)
        tractor = TractorAgent(
            internal_state=internal,
            env=env,
            blackboard=blackboard,
            policy=policy,
            use_qlearning=req.use_qlearning,
        )
        tractors.append(tractor)

    initial_plants: List[PlantDTO] = []
    for pos, is_good in env.iter_plants():
        x, y = pos
        cost_mov = env.movement_cost(pos)
        initial_plants.append(
            PlantDTO(
                x=x,
                y=y,
                buena=is_good,
                mala=not is_good,
                costo_mov=cost_mov,
                transitable=env.is_transitable(pos),
            )
        )

    terrain: List[TerrainDTO] = []
    for pos, cell_type_name, cost, transitable in env.iter_terrain():
        x, y = pos
        terrain.append(
            TerrainDTO(
                x=x,
                y=y,
                cell_type=cell_type_name,
                costo_mov=cost,
                transitable=transitable,
            )
        )

    frames: List[FrameDTO] = []

    ticks_series: List[int] = []
    remaining_series: List[int] = []
    fuel_series: Dict[int, List[float]] = {}

    for tractor in tractors:
        st = tractor.get_internal_state()
        fuel_series[st.id] = []

    for tick in range(req.max_ticks):
        env.update_climate()
        if blackboard.remaining_plants() == 0:
            break
        for tractor in tractors:
            tractor.step()
        tractor_states: List[TractorStateDTO] = []
        for tractor in tractors:
            st = tractor.get_internal_state()
            tractor_states.append(
                TractorStateDTO(
                    id=st.id,
                    x=st.x,
                    y=st.y,
                    fuel=st.fuel,
                    load=st.load,
                )
            )
            fuel_series[st.id].append(st.fuel)
        frames.append(FrameDTO(tick=tick, tractors=tractor_states))
        ticks_series.append(tick)
        remaining_series.append(blackboard.remaining_plants())
        for pos, is_good in env.iter_plants():
            blackboard.add_plant(pos)

    total_good = 0
    total_bad = 0
    total_fuel_used = 0.0
    tractor_good: Dict[int, int] = {}
    tractor_bad: Dict[int, int] = {}

    for tractor in tractors:
        st = tractor.get_internal_state()
        total_good += st.harvested_good
        total_bad += st.harvested_bad
        total_fuel_used += st.total_fuel_used
        tractor_good[st.id] = st.harvested_good
        tractor_bad[st.id] = st.harvested_bad

    remaining = blackboard.remaining_plants()
    total_plants = len(initial_plants)

    stats = SimulationStatsDTO(
        total_plants=total_plants,
        good_harvested=total_good,
        bad_harvested=total_bad,
        remaining_plants=remaining,
        total_fuel_used=total_fuel_used,
        ticks_run=len(frames),
        num_tractors=req.num_tractors,
        grid_size=req.N,
    )

    dashboard_png = _build_dashboard_png(
        req=req,
        ticks=ticks_series,
        remaining=remaining_series,
        fuel_series=fuel_series,
        tractor_good=tractor_good,
        tractor_bad=tractor_bad,
        env=env,
        total_plants=total_plants,
    )

    return SimulationResultDTO(
        initial_plants=initial_plants,
        terrain=terrain,
        frames=frames,
        stats=stats,
        dashboard_png=dashboard_png,
    )
