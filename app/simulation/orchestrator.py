from typing import List
from app.core.types import SimulationRequestDTO, SimulationResultDTO, SimulationStatsDTO, PlantDTO, FrameDTO, TractorStateDTO, TractorInternalState, TerrainDTO
from app.env.environment import Environment
from app.env.blackboard import Blackboard
from app.learning.qlearning import QLearningPolicy
from app.agents.tractor import TractorAgent
from app.agents.interfaces import IHighLevelPolicy


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
        policy: IHighLevelPolicy = QLearningPolicy(seed=req.seed) if req.use_qlearning else None
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
        frames.append(FrameDTO(tick=tick, tractors=tractor_states))
        for pos, is_good in env.iter_plants():
            blackboard.add_plant(pos)
    total_good = 0
    total_bad = 0
    total_fuel_used = 0.0
    for tractor in tractors:
        st = tractor.get_internal_state()
        total_good += st.harvested_good
        total_bad += st.harvested_bad
        total_fuel_used += st.total_fuel_used
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
    return SimulationResultDTO(
        initial_plants=initial_plants,
        terrain=terrain,
        frames=frames,
        stats=stats,
    )
