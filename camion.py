from __future__ import annotations
from typing import Any, Dict

from python.simulacion.simulacion_tractores import (
    run_simulation_return_history as _run_sim,
)


def run_simulation_return_history(
    N: int,
    T: int,
    capacity: int,
    p_good: float = 0.5,
    seed: int = 42,
    max_ticks: int = 2000,
) -> Dict[str, Any]:
    return _run_sim(
        N=N,
        T=T,
        capacity=capacity,
        p_good=p_good,
        seed=seed,
        max_ticks=max_ticks,
    )
