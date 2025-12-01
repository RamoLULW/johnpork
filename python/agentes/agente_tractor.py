from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional

from ..core.types import (
    Coord3D,
    EstadoAgente,
    EstadoTractorDTO,
)
from ..core.interfaces import ITractor, IAmbiente
from ..pathfinding.astar_3d import astar_3d
from ..aprendizaje.blackboard import BlackboardCampo
from ..aprendizaje.q_learning import PoliticaQLearning, AccionTractor


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

    capacidad_maxima: int = 100
    carga_actual: int = 0

    combustible_max: float = 100.0
    combustible_actual: float = 100.0
    consumo_base: float = 1.0

    estacion: Coord3D = field(init=False)
    meta_trabajo: Coord3D = field(init=False)

    blackboard: Optional[BlackboardCampo] = None
    politica_q: Optional[PoliticaQLearning] = None

    def __post_init__(self) -> None:
        self.posicion_actual = self.posicion_inicial
        self.estacion = self.posicion_inicial
        self.meta_trabajo = self.meta

    @property
    def id(self) -> int:
        return self._id

    def inicializar(self) -> None:
        self.planear_ruta()

    def necesita_recargar(self) -> bool:
        return self.combustible_actual <= 0.2 * self.combustible_max

    def necesita_descargar(self) -> bool:
        return self.carga_actual >= self.capacidad_maxima

    def _estado_q(self) -> tuple[int, int, int]:
        combustible_bajo = 1 if self.necesita_recargar() else 0
        capacidad_llena = 1 if self.necesita_descargar() else 0
        planta_disponible = 0
        if self.blackboard is not None and self.blackboard.hay_planta_disponible():
            planta_disponible = 1
        return (combustible_bajo, capacidad_llena, planta_disponible)

    def step(self, tick_actual: int) -> None:
        if self.politica_q is None or self.blackboard is None:
            self._step_simple()
        else:
            self._step_con_q(tick_actual)

    def _step_simple(self) -> None:
        if self.estado == EstadoAgente.TERMINADO:
            return

        if self.combustible_actual <= 0.0:
            self.estado = EstadoAgente.BLOQUEADO
            return

        if (
            self.posicion_actual == self.meta
            and self.meta == self.meta_trabajo
            and not self.necesita_recargar()
            and not self.necesita_descargar()
        ):
            self.estado = EstadoAgente.TERMINADO
            return

        if self.posicion_actual == self.estacion:
            self.combustible_actual = self.combustible_max
            self.carga_actual = 0
            if self.meta != self.meta_trabajo:
                self.meta = self.meta_trabajo
                self.planear_ruta()
                return

        if (self.necesita_recargar() or self.necesita_descargar()) and self.meta != self.estacion:
            self.meta = self.estacion
            self.planear_ruta()
            return

        if not self.ruta_actual or self._indice_ruta >= len(self.ruta_actual):
            self.planear_ruta()
            return

        self.mover_un_paso()

    def _step_con_q(self, tick_actual: int) -> None:
        if self.estado == EstadoAgente.TERMINADO:
            return

        if self.combustible_actual <= 0.0:
            self.estado = EstadoAgente.BLOQUEADO
            return

        if self.posicion_actual == self.estacion:
            self.combustible_actual = self.combustible_max
            self.carga_actual = 0

        estado_actual = self._estado_q()
        accion = self.politica_q.elegir_accion(estado_actual)

        if accion == AccionTractor.IR_ESTACION:
            if self.meta != self.estacion:
                self.meta = self.estacion
                self.planear_ruta()
        elif accion == AccionTractor.IR_PLANTA_BUENA and self.blackboard is not None:
            coord = self.blackboard.obtener_planta_para_tractor(self.id)
            if coord is not None and coord != self.meta:
                self.meta_trabajo = coord
                self.meta = coord
                self.planear_ruta()
        elif accion == AccionTractor.EXPLORAR:
            if not self.ruta_actual or self.posicion_actual == self.meta:
                x, y, z = self.posicion_actual
                nx = x
                ny = y
                if hasattr(self.ambiente, "_dim"):
                    nx = min(self.ambiente._dim.size_x - 1, x + 1)
                    ny = min(self.ambiente._dim.size_y - 1, y + 1)
                nueva_meta = (nx, ny, z)
                self.meta_trabajo = nueva_meta
                self.meta = nueva_meta
                self.planear_ruta()

        if (
            self.posicion_actual == self.meta
            and self.meta == self.meta_trabajo
            and not self.necesita_recargar()
            and not self.necesita_descargar()
        ):
            self.estado = EstadoAgente.TERMINADO
            return

        if not self.ruta_actual or self._indice_ruta >= len(self.ruta_actual):
            self.planear_ruta()

        before_carga = self.carga_actual
        before_estado = self.estado

        if self.estado != EstadoAgente.BLOQUEADO:
            self.mover_un_paso()

        recompensa = -0.1
        if self.carga_actual > before_carga:
            recompensa += 5.0
        if self.posicion_actual == self.estacion:
            recompensa += 1.0
        if (
            self.posicion_actual == self.meta_trabajo
            and self.meta == self.meta_trabajo
        ):
            recompensa += 10.0
        if self.estado == EstadoAgente.BLOQUEADO and before_estado != EstadoAgente.BLOQUEADO:
            recompensa -= 5.0

        estado_siguiente = self._estado_q()
        self.politica_q.actualizar_q(estado_actual, accion, recompensa, estado_siguiente)

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

        costo_mov = self.ambiente.cost_mov_3d(siguiente)
        consumo = self.consumo_base * float(costo_mov)

        if self.combustible_actual < consumo:
            self.estado = EstadoAgente.BLOQUEADO
            return

        self.combustible_actual -= consumo
        if self.combustible_actual < 0.0:
            self.combustible_actual = 0.0

        self.posicion_actual = siguiente
        self._indice_ruta += 1

        self.interactuar_con_celda()

        if self.posicion_actual == self.meta:
            self.estado = EstadoAgente.MOVIENDO
        else:
            self.estado = EstadoAgente.MOVIENDO

    def interactuar_con_celda(self) -> None:
        coord = self.posicion_actual

        if hasattr(self.ambiente, "es_planta") and self.ambiente.es_planta(coord):
            buena = False
            if hasattr(self.ambiente, "es_planta_buena") and self.ambiente.es_planta_buena(coord):
                buena = True
            if self.blackboard is not None:
                self.blackboard.registrar_planta(coord, buena=buena)
            if buena and self.carga_actual < self.capacidad_maxima:
                self.carga_actual += 1

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
            combustible=self.combustible_actual,
            combustible_max=self.combustible_max,
            carga_actual=self.carga_actual,
            capacidad_maxima=self.capacidad_maxima,
        )

    def reiniciar(
        self,
        nueva_posicion: Optional[Coord3D] = None,
        nueva_meta: Optional[Coord3D] = None,
    ) -> None:
        if nueva_posicion is not None:
            self.posicion_inicial = nueva_posicion
        if nueva_meta is not None:
            self.meta = nueva_meta
            self.meta_trabajo = nueva_meta

        self.posicion_actual = self.posicion_inicial
        self.estacion = self.posicion_inicial
        self.ruta_actual = []
        self._indice_ruta = 0
        self.estado = EstadoAgente.AFK
        self.carga_actual = 0
        self.combustible_actual = self.combustible_max
