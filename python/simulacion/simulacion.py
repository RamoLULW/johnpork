from __future__ import annotations
from dataclasses import dataclass
from typing import List
from ..core.interfaces import ISimulacion, IAmbiente, ITractor
from ..core.types import EstadoTractorDTO, AmbienteInfoDTO

@dataclass
class Simulacion(ISimulacion):
    _ambiente: IAmbiente
    _tractores: List[ITractor]
    _tick_actual: int = 0

    def inicializar(self) -> None:
        self._tick_actual = 0
        for tractor in self._tractores:
            tractor.inicializar()

    def step(self) -> None:
        self._tick_actual += 1
        for tractor in self._tractores:
            tractor.step(self._tick_actual)

    def obtener_tick_actual(self) -> int:
        return self._tick_actual
    
    def obtener_tractores(self) -> List[ITractor]:
        return self._tractores
    
    def obtener_ambiente(self) -> IAmbiente:
        return self._ambiente
    
    def estado_tractores_dto(self) -> List[EstadoTractorDTO]:
        return [tractor.estado_dto() for tractor in self._tractores]
    
    def info_ambiente_dto(self) -> AmbienteInfoDTO:
        return self._ambiente.info_ambiente()
    
    def reiniciar(self) -> None:
        self.inicializar()