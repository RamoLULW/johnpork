from fastapi import APIRouter
from app.core.types import SimulationRequestDTO, SimulationResultDTO
from app.simulation.orchestrator import run_simulation

router = APIRouter()


@router.post("/simulate", response_model=SimulationResultDTO)
def simulate(req: SimulationRequestDTO):
    result = run_simulation(req)
    return result
