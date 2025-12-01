from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Iterable, List

from ..core.types import (
    Coord2D,
    Coord3D,
    Dimensiones,
    TerrenoCelda,
    TipoCelda,
    AmbienteInfoDTO,
    TipoClima,
)
from ..core.interfaces import IAmbiente


@dataclass
class _DimensionesGrid:
    size_x: int
    size_y: int
    size_z: int


class AmbienteGrid3D(IAmbiente):
    def __init__(self, size_x: int, size_y: int, size_z: int = 1):
        if size_x <= 0 or size_y <= 0 or size_z <= 0:
            raise ValueError("Dimensiones tienen que ser positivas")

        self._dim = _DimensionesGrid(size_x=size_x, size_y=size_y, size_z=size_z)
        self._celdas: Dict[Coord3D, TerrenoCelda] = {}

        self._clima: TipoClima = TipoClima.SOLEADO
        self._estaciones: List[Coord3D] = []
        self._plantas_buenas: set[Coord3D] = set()
        self._plantas_malas: set[Coord3D] = set()

        for x in range(size_x):
            for y in range(size_y):
                for z in range(size_z):
                    self._celdas[(x, y, z)] = TerrenoCelda(
                        tipo=TipoCelda.TIERRA,
                        transitable=True,
                        costo_mov=1.0,
                    )

    def obtener_dimensiones(self) -> Dimensiones:
        return Dimensiones.TRES_D

    def en_rango_2d(self, coord: Coord2D) -> bool:
        x, y = coord
        return (
            0 <= x < self._dim.size_x
            and 0 <= y < self._dim.size_y
            and 0 <= 0 < self._dim.size_z
        )

    def en_rango_3d(self, coord: Coord3D) -> bool:
        x, y, z = coord
        return (
            0 <= x < self._dim.size_x
            and 0 <= y < self._dim.size_y
            and 0 <= z < self._dim.size_z
        )

    def es_transitable_2d(self, coord: Coord2D) -> bool:
        if not self.en_rango_2d(coord):
            return False
        x, y = coord
        celda = self._celdas[(x, y, 0)]
        return celda.transitable

    def es_transitable_3d(self, coord: Coord3D) -> bool:
        if not self.en_rango_3d(coord):
            return False
        celda = self._celdas[coord]
        return celda.transitable

    def cost_mov_2d(self, coord: Coord2D) -> float:
        if not self.en_rango_2d(coord):
            raise ValueError(f"Coordenada fuera de rango (2D): {coord}")
        x, y = coord
        celda = self._celdas[(x, y, 0)]
        return celda.costo_mov

    def cost_mov_3d(self, coord: Coord3D) -> float:
        if not self.en_rango_3d(coord):
            raise ValueError(f"Coordenada fuera de rango (3D): {coord}")
        celda = self._celdas[coord]
        return celda.costo_mov

    def vecinos_2d(self, coord: Coord2D) -> Iterable[Coord2D]:
        x, y = coord
        candidatos = [
            (x + 1, y),
            (x - 1, y),
            (x, y + 1),
            (x, y - 1),
        ]
        for nx, ny in candidatos:
            if self.es_transitable_2d((nx, ny)):
                yield (nx, ny)

    def vecinos_3d(self, coord: Coord3D) -> Iterable[Coord3D]:
        x, y, z = coord
        candidatos = [
            (x + 1, y, z),
            (x - 1, y, z),
            (x, y + 1, z),
            (x, y - 1, z),
            (x, y, z + 1),
            (x, y, z - 1),
        ]
        for ncoord in candidatos:
            if self.es_transitable_3d(ncoord):
                yield ncoord

    def info_ambiente(self) -> AmbienteInfoDTO:
        return AmbienteInfoDTO(
            size_x=self._dim.size_x,
            size_y=self._dim.size_y,
            size_z=self._dim.size_z,
        )

    def set_celda(
        self,
        coord: Coord3D,
        tipo: TipoCelda,
        transitable: bool,
        costo_mov: float = 1.0,
    ) -> None:
        if not self.en_rango_3d(coord):
            raise ValueError(f"Coordenada fuera de rango al setear celda: {coord}")
        self._celdas[coord] = TerrenoCelda(
            tipo=tipo,
            transitable=transitable,
            costo_mov=costo_mov,
        )

    def poner_piedra(self, coord: Coord3D) -> None:
        self.set_celda(
            coord=coord,
            tipo=TipoCelda.PIEDRA,
            transitable=False,
            costo_mov=float("inf"),
        )

    def poner_planta(self, coord: Coord3D, densa: bool = True, buena: bool = True) -> None:
        if not self.en_rango_3d(coord):
            raise ValueError(f"Coordenada fuera de rango al poner planta: {coord}")

        if densa:
            self.set_celda(
                coord=coord,
                tipo=TipoCelda.PLANTA,
                transitable=False,
                costo_mov=float("inf"),
            )
        else:
            self.set_celda(
                coord=coord,
                tipo=TipoCelda.PLANTA,
                transitable=True,
                costo_mov=2.0,
            )

        if buena:
            self._plantas_buenas.add(coord)
            self._plantas_malas.discard(coord)
        else:
            self._plantas_malas.add(coord)
            self._plantas_buenas.discard(coord)

    def poner_agua(self, coord: Coord3D, transitable: bool = False) -> None:
        self.set_celda(
            coord=coord,
            tipo=TipoCelda.AGUA,
            transitable=transitable,
            costo_mov=3.0 if transitable else float("inf"),
        )

    def poner_camino(self, coord: Coord3D) -> None:
        self.set_celda(
            coord=coord,
            tipo=TipoCelda.CAMINO,
            transitable=True,
            costo_mov=0.5,
        )

    def obtener_celda(self, coord: Coord3D) -> TerrenoCelda:
        if not self.en_rango_3d(coord):
            raise ValueError(f"Coordenada fuera de rango: {coord}")
        return self._celdas[coord]

    def poner_estacion(self, coord: Coord3D) -> None:
        if not self.en_rango_3d(coord):
            raise ValueError(f"Coordenada fuera de rango al poner estación: {coord}")

        self.set_celda(
            coord=coord,
            tipo=TipoCelda.CAMINO,
            transitable=True,
            costo_mov=0.5,
        )
        self._estaciones.append(coord)

    def obtener_estaciones(self) -> List[Coord3D]:
        return list(self._estaciones)

    def es_planta(self, coord: Coord3D) -> bool:
        if not self.en_rango_3d(coord):
            return False
        return self._celdas[coord].tipo == TipoCelda.PLANTA

    def es_planta_buena(self, coord: Coord3D) -> bool:
        return coord in self._plantas_buenas

    def es_planta_mala(self, coord: Coord3D) -> bool:
        return coord in self._plantas_malas

    def obtener_plantas(self) -> List[dict]:
        plantas: List[dict] = []
        for (x, y, z), celda in self._celdas.items():
            if celda.tipo == TipoCelda.PLANTA:
                plantas.append(
                    {
                        "x": x,
                        "y": y,
                        "z": z,
                        "buena": (x, y, z) in self._plantas_buenas,
                        "mala": (x, y, z) in self._plantas_malas,
                        "costo_mov": celda.costo_mov,
                        "transitable": celda.transitable,
                    }
                )
        return plantas

    @property
    def clima(self) -> TipoClima:
        return self._clima

    def cambiar_clima(self, nuevo_clima: TipoClima) -> None:
        self._clima = nuevo_clima

        if nuevo_clima == TipoClima.LLUVIA:
            for celda in self._celdas.values():
                if celda.tipo == TipoCelda.TIERRA and celda.transitable:
                    celda.costo_mov = max(celda.costo_mov, 3.0)
        else:
            for celda in self._celdas.values():
                if celda.tipo == TipoCelda.TIERRA and celda.transitable:
                    celda.costo_mov = 1.0
