from fastapi import APIRouter
from app.core.types import SimulationRequestDTO, SimulationResultDTO
from app.simulation.orchestrator import run_simulation
from app.learning.trainer import train_qlearning_policy

router = APIRouter()


@router.post("/simulate", response_model=SimulationResultDTO)
def simulate(req: SimulationRequestDTO):
    if req.use_qlearning:
        result = train_qlearning_policy(req)
    else:
        result = run_simulation(req)
    return result
