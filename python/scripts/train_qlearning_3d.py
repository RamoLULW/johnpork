from __future__ import annotations
from typing import Optional

from ..aprendizaje.q_learning import PoliticaQLearning
from ..aprendizaje.blackboard import BlackboardCampo
from ..simulacion.simulacion_tractores import run_simulation_return_history


def entrenar_qlearning_3d(
    episodios: int,
    N: int,
    T: int,
    capacity: int,
    p_good: float = 0.5,
    seed_inicial: int = 42,
    max_ticks: int = 2000,
) -> PoliticaQLearning:
    politica = PoliticaQLearning()

    for ep in range(episodios):
        seed = seed_inicial + ep
        blackboard = BlackboardCampo()
        run_simulation_return_history(
            N=N,
            T=T,
            capacity=capacity,
            p_good=p_good,
            seed=seed,
            max_ticks=max_ticks,
            politica_q=politica,
            blackboard=blackboard,
        )
        politica.decaer_epsilon()

    return politica


if __name__ == "__main__":
    politica = entrenar_qlearning_3d(
        episodios=1000,
        N=12,
        T=4,
        capacity=144,
        p_good=0.6,
        seed_inicial=42,
        max_ticks=1000,
    )
    print("Entrenamiento terminado. Epsilon final:", politica.epsilon)
