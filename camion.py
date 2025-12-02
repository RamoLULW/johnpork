from __future__ import annotations
from typing import Any, Dict, Optional

from python.simulacion.simulacion_tractores import (
    run_simulation_return_history as _run_sim,
)
from python.aprendizaje.q_learning import PoliticaQLearning
from python.aprendizaje.blackboard import BlackboardCampo


def run_simulation_return_history(
    N: int,
    T: int,
    capacity: int,
    p_good: float = 0.5,
    seed: int = 42,
    max_ticks: int = 2000,
    use_qlearning: bool = False,
) -> Dict[str, Any]:
    politica: Optional[PoliticaQLearning] = None
    blackboard: Optional[BlackboardCampo] = None

    if use_qlearning:
        politica = PoliticaQLearning()
        blackboard = BlackboardCampo()

    return _run_sim(
        N=N,
        T=T,
        capacity=capacity,
        p_good=p_good,
        seed=seed,
        max_ticks=max_ticks,
        politica_q=politica,
        blackboard=blackboard,
    )
