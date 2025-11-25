from __future__ import annotations
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Dict, List, Tuple

Coord2D = Tuple[int, int]
Coord3D = Tuple[int, int, int]

class Dimensiones(Enum):
    DOS_D = 2
    TRES_D = 3

class TipoCelda(Enum):
    TIERRA = "tierra"
    PIEDRA = "piedra"
    PLANTA = "planta"
    AGUA = "agua"
    CAMINO = "camino"
    OBSTACULO = "obstaculo"

@dataclass
class TerrenoCelda:
    tipo: TipoCelda
    transitable: bool = True
    costo_mov: float = 1.0

class EstadoAgente(Enum):
    AFK = "afk"
    PLANEANDO = "planeando"
    MOVIENDO = "moviendo"
    BLOQUEADO = "bloqueado"
    TERMINADO = "terminado"

class TipoMensaje(Enum):
    P2P = auto()
    EVENTO = auto()
    CONTRACT_NET_CFP = auto()
    CONTRACT_NET_PROPOSAL = auto()
    CONTRACT_NET_AWARD = auto()
    CONTRACT_NET_REJECT = auto()

@dataclass
class Mensaje:
    id_emisor: int
    id_receptor: int | None
    tipo: TipoMensaje
    contenido: Dict[str, Any]
    tick: int

@dataclass
class EstadoTractorDTO:
    id: int
    x: float
    y: float
    z: float
    goal_x: float
    goal_y: float
    goal_z: float
    estado: str
    path: List[Tuple[float, float, float]]

@dataclass
class AmbienteInfoDTO:
    size_x: int
    size_y: int
    size_z: int

__all__ = [
    "Coord2D",
    "Coord3D",
    "Dimensiones",
    "TipoCelda",
    "TerrenoCelda",
    "EstadoAgente",
    "TipoMensaje",
    "Mensaje",
    "EstadoTractorDTO",
    "AmbienteInfoDTO",
]