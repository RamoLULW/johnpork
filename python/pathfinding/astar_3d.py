from __future__ import annotations
import heapq
from typing import Dict, List, Optional
from ..core.types import Coord3D
from ..core.interfaces import IAmbiente

def heuristica_manhattan_3d(a: Coord3D, b: Coord3D) -> float:
    return abs(a[0] - b[0]) + abs(a[1] - b[1]) + abs(a[2] - b[2])

def _reconstruir_camino(
        came_from: Dict[Coord3D, Coord3D],
        inicio: Coord3D,
        meta: Coord3D,
) -> List[Coord3D]:
    actual = meta
    camino: List[Coord3D] = [actual]

    while actual != inicio:
        actual = came_from[actual]
        camino.append(actual)

    camino.reverse()
    return camino

def astar_3d(
        ambiente: IAmbiente,
        inicio: Coord3D,
        meta: Coord3D,
) -> List[Coord3D]:
    if not ambiente.en_rango_3d(inicio):
        raise ValueError(f"Coordenada de inicio fuera de rango: {inicio}")
    if not ambiente.en_rango_3d(meta):
        raise ValueError(f"Coordenada de meta fuera de rango: {meta}")
    if not ambiente.es_transitable_3d(inicio):
        raise ValueError(f"La celda de inicio no es transitable: {inicio}")
    if not ambiente.es_transitable_3d(meta):
        raise ValueError(f"La celda de meta no es transitable: {meta}")
    
    abiertos: List[tuple[float, int, Coord3D]] = []
    heapq.heapify(abiertos)

    came_from: Dict[Coord3D, Coord3D] = {}
    g_score: Dict[Coord3D, float] = {inicio: 0.0}

    contador = 0
    f_inicio = heuristica_manhattan_3d(inicio, meta)
    heapq.heappush(abiertos, (f_inicio, contador, inicio))
    visitados: Dict[Coord3D, bool] = {}

    while abiertos:
        _, _, actual = heapq.heappop(abiertos)

        if visitados.get(actual, False):
            continue
        visitados[actual] = True

        if actual == meta:
            return _reconstruir_camino(came_from, inicio, meta)
        
        for vecino in ambiente.vecinos_3d(actual):
            try:
                costo_celda = ambiente.cost_mov_3d(vecino)
            except ValueError:
                continue

            if costo_celda == float("inf"):
                continue

            tentativo_g = g_score[actual] + costo_celda

            if tentativo_g < g_score.get(vecino, float("inf")):
                came_from[vecino] = actual
                g_score[vecino] = tentativo_g

                f_score = tentativo_g + heuristica_manhattan_3d(vecino, meta)
                contador += 1
                heapq.heappush(abiertos, (f_score, contador, vecino))

    return []