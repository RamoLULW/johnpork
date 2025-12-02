from __future__ import annotations
from dataclasses import asdict
from typing import Any, Dict, List, Optional
import random

from ..ambiente.ambiente_grid3d import AmbienteGrid3D
from ..agentes.agente_tractor import TractorAgente
from ..core.types import Coord3D, EstadoAgente
from ..aprendizaje.blackboard import BlackboardCampo
from ..aprendizaje.q_learning import PoliticaQLearning
from .simulacion import Simulacion


def run_simulation_return_history(
    N: int,
    T: int,
    capacity: int,
    p_good: float = 0.5,
    seed: int = 42,
    max_ticks: int = 2000,
    politica_q: Optional[PoliticaQLearning] = None,
    blackboard: Optional[BlackboardCampo] = None,
) -> Dict[str, Any]:
    random.seed(seed)

    size_x = N
    size_y = N
    size_z = 1

    ambiente = AmbienteGrid3D(size_x=size_x, size_y=size_y, size_z=size_z)

    estacion: Coord3D = (0, 0, 0)
    ambiente.poner_estacion(estacion)

    for x in range(size_x):
        for y in range(size_y):
            if (x, y) == (0, 0):
                continue
            if random.random() < 0.1:
                buena = random.random() < p_good
                ambiente.poner_planta((x, y, 0), densa=False, buena=buena)

    if blackboard is None:
        blackboard = BlackboardCampo()

    tractores: List[TractorAgente] = []
    for i in range(T):
        meta: Coord3D = (size_x - 1, size_y - 1, 0)
        tractor = TractorAgente(
            _id=i + 1,
            ambiente=ambiente,
            posicion_inicial=estacion,
            meta=meta,
            capacidad_maxima=capacity,
        )
        tractor.blackboard = blackboard
        if politica_q is not None:
            tractor.politica_q = politica_q
        tractores.append(tractor)

    simulacion = Simulacion(_ambiente=ambiente, _tractores=tractores)
    simulacion.inicializar()

    history: Dict[str, Any] = {
        "tick": [],
        "pos": [],
        "load": [],
        "plants": ambiente.obtener_plantas(),
    }

    for tick in range(max_ticks):
        history["tick"].append(tick)

        frame_pos: List[Dict[str, Any]] = []
        frame_load: List[int] = []

        for tractor in simulacion.obtener_tractores():
            dto = tractor.estado_dto()
            dto_dict = asdict(dto)
            frame_pos.append(dto_dict)
            carga = dto.carga_actual if dto.carga_actual is not None else 0
            frame_load.append(int(carga))

        history["pos"].append(frame_pos)
        history["load"].append(frame_load)

        if all(t.estado == EstadoAgente.TERMINADO for t in simulacion.obtener_tractores()):
            break

        simulacion.step()

    return history
