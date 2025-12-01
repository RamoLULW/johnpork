from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Iterable, List
from .types import (
    Coord2D,
    Coord3D,
    Dimensiones,
    EstadoTractorDTO,
    AmbienteInfoDTO,
)

class IAmbiente(ABC):
    @abstractmethod
    def obtener_dimensiones(self) -> Dimensiones:
        raise NotImplementedError
    
    @abstractmethod
    def en_rango_2d(self, coord: Coord2D) -> bool:
        raise NotImplementedError
    
    @abstractmethod
    def en_rango_3d(self, coord: Coord3D) -> bool:
        raise NotImplementedError
    
    @abstractmethod
    def es_transitable_2d(self, coord: Coord2D) -> bool:
        raise NotImplementedError
    
    @abstractmethod
    def es_transitable_3d(self, coord: Coord3D) -> bool:
        raise NotImplementedError
    
    @abstractmethod
    def cost_mov_2d(self, coord: Coord2D) -> float:
        raise NotImplementedError
    
    @abstractmethod
    def cost_mov_3d(self, coord: Coord3D) -> float:
        raise NotImplementedError
    
    @abstractmethod
    def vecinos_2d(self, coord: Coord2D) -> Iterable[Coord2D]:
        raise NotImplementedError
    
    @abstractmethod
    def vecinos_3d(self, coord: Coord3D) -> Iterable[Coord3D]:
        raise NotImplementedError
    
    def info_ambiente(self) -> AmbienteInfoDTO:
        raise NotImplementedError
    

class IAgente(ABC):
    @property
    @abstractmethod
    def id(self) -> int:
        raise NotImplementedError
    
    @abstractmethod
    def inicializar(self) -> None:
        raise NotImplementedError
    
    @abstractmethod
    def step(self, tick_actual: int) -> None:
        raise NotImplementedError
    
class ITractor(IAgente):
    def estado_dto(self) -> EstadoTractorDTO:
        raise NotImplementedError

class ISimulacion(ABC):
    @abstractmethod
    def inicializar(self) -> None:
        raise NotImplementedError
    
    @abstractmethod
    def step(self) -> None:
        raise NotImplementedError
    
    @abstractmethod
    def obtener_tick_actual(self) -> int:
        raise NotImplementedError
    
    @abstractmethod
    def obtener_tractores(self) -> List[ITractor]:
        raise NotImplementedError
    
    @abstractmethod
    def obtener_ambiente(self) -> IAmbiente:
        raise NotImplementedError
    
    @abstractmethod
    def estado_tractores_dto(self) -> List[EstadoTractorDTO]:
        raise NotImplementedError
    
    @abstractmethod
    def info_ambiente_dto(self) -> AmbienteInfoDTO:
        raise NotImplementedError