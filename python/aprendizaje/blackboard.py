from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional, Set

from ..core.types import Coord3D


@dataclass
class BlackboardCampo:
    plantas_buenas: Set[Coord3D] = field(default_factory=set)
    plantas_malas: Set[Coord3D] = field(default_factory=set)
    asignaciones: Dict[Coord3D, int] = field(default_factory=dict)

    def registrar_planta(self, coord: Coord3D, buena: bool) -> None:
        if buena:
            self.plantas_buenas.add(coord)
            self.plantas_malas.discard(coord)
        else:
            self.plantas_malas.add(coord)
            self.plantas_buenas.discard(coord)

    def hay_planta_disponible(self) -> bool:
        for coord in self.plantas_buenas:
            if coord not in self.asignaciones:
                return True
        return False

    def obtener_planta_para_tractor(self, tractor_id: int) -> Optional[Coord3D]:
        for coord in self.plantas_buenas:
            asignado = self.asignaciones.get(coord)
            if asignado is None or asignado == tractor_id:
                self.asignaciones[coord] = tractor_id
                return coord
        return None
