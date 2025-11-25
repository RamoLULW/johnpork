from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional
from core.types import (
    Coord3D,
    EstadoAgente,
    EstadoTractorDTO,
)
from core.interfaces import ITractor, IAmbiente
from pathfinding.astar_3d import astar_3d

@dataclass
class TractorAgente(ITractor):
    _id: int
    ambiente: IAmbiente
    posicion_inicial: Coord3D
    meta: Coord3D
    
    estado: EstadoAgente = EstadoAgente.AFK
    posicion_actual: Coord3D = field(init=False)
    ruta_actual: List[Coord3D] = field(default_factory=list)
    _indice_ruta: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.posicion_actual = self.posicion_inicial

    def id(self) -> int: 
        return self._id
    
    def inicializar(self) -> None:
        self.planear_ruta()

    def step(self, tick_actual: int) -> None:
        if self.estado == EstadoAgente.TERMINADO:
            return
        
        if self.posicion_actual == self.meta:
            self.estado = EstadoAgente.TERMINADO
            return
        
        if not self.ruta_actual or self._indice_ruta >= len(self.ruta_actual):
            self.planear_ruta()
            return
        
        self.mover_un_paso()

    def planear_ruta(self) -> None:
        self.estado = EstadoAgente.PLANEANDO

        ruta = astar_3d(
            ambiente=self.ambiente,
            inicio=self.posicion_actual,
            meta=self.meta,
        )

        self.ruta_actual = ruta
        self._indice_ruta = 0

        if not self.ruta_actual:
            self.estado = EstadoAgente.BLOQUEADO
        elif self.posicion_actual == self.meta:
            self.estado = EstadoAgente.TERMINADO
        else:
            self.estado = EstadoAgente.MOVIENDO

    def mover_un_paso(self) -> None:
        if not self.ruta_actual or self._indice_ruta >= len(self.ruta_actual):
            self.estado = EstadoAgente.BLOQUEADO
            return
        
        siguiente = self.ruta_actual[self._indice_ruta]

        if not self.ambiente.es_transitable_3d(siguiente):
            self.estado = EstadoAgente.BLOQUEADO
            return
        
        self.posicion_actual = siguiente
        self._indice_ruta += 1

        if self.posicion_actual == self.meta:
            self.estado = EstadoAgente.TERMINADO
        else:
            self.estado = EstadoAgente.MOVIENDO

    def estado_dto(self) -> EstadoTractorDTO:
        x, y, z = self.posicion_actual
        gx, gy, gz = self.meta

        ruta_dto = [(float(px), float(py), float(pz)) for (px, py, pz) in self.ruta_actual]

        return EstadoTractorDTO(
            id=self.id,
            x=float(x),
            y=float(y),
            z=float(z),
            goal_x=float(gx),
            goal_y=float(gy),
            goal_z=float(gz),
            estado=self.estado.value,
            path=ruta_dto,
        )
    
    def reiniciar(
            self,
            nueva_posicion: Optional[Coord3D] = None,
            nueva_meta: Optional[Coord3D] = None
    ) -> None:
        if nueva_posicion is not None:
            self.posicion_inicial = nueva_posicion
        if nueva_meta is not None:
            self.meta = nueva_meta

        self.posicion_actual = self.posicion_inicial
        self.ruta_actual = []
        self._indice_ruta = 0
        self.estado = EstadoAgente.AFK