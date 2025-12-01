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
    LODO = "lodo"
    ESTACION = "estacion"

class TipoClima(Enum):
    SOLEADO = "soleado"
    LLUVIA = "lluvia"
    NUBLADO = "nublado"

@dataclass
class TerrenoCelda:
    tipo: TipoCelda
    transitable: bool = True
    costo_mov: float = 1.0
    metadata: Dict[str, Any] | None = None

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

    combustible: float | None = None
    combustible_max: float | None = None
    carga_actual: int | None = None
    capacidad_maxima: int | None = None

@dataclass
class AmbienteInfoDTO:
    size_x: int
    size_y: int
    size_z: int

    clima: str | None = None
    estaciones: List[Tuple[int, int, int]] | None = None

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
    "TipoClima",
]