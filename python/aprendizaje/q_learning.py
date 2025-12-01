from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, Tuple, List
import random

EstadoQL = Tuple[int, int, int]


class AccionTractor(Enum):
    IR_ESTACION = auto()
    IR_PLANTA_BUENA = auto()
    EXPLORAR = auto()
    MANTENER_META = auto()


@dataclass
class PoliticaQLearning:
    alpha: float = 0.1
    gamma: float = 0.9
    epsilon: float = 0.1
    q_table: Dict[Tuple[EstadoQL, AccionTractor], float] = field(default_factory=dict)

    def acciones_disponibles(self) -> List[AccionTractor]:
        return list(AccionTractor)

    def valor_q(self, estado: EstadoQL, accion: AccionTractor) -> float:
        return self.q_table.get((estado, accion), 0.0)

    def mejor_accion(self, estado: EstadoQL) -> AccionTractor:
        acciones = self.acciones_disponibles()
        mejor = acciones[0]
        mejor_valor = self.valor_q(estado, mejor)
        for a in acciones[1:]:
            v = self.valor_q(estado, a)
            if v > mejor_valor:
                mejor = a
                mejor_valor = v
        return mejor

    def elegir_accion(self, estado: EstadoQL) -> AccionTractor:
        if random.random() < self.epsilon:
            return random.choice(self.acciones_disponibles())
        return self.mejor_accion(estado)

    def actualizar_q(
        self,
        estado: EstadoQL,
        accion: AccionTractor,
        recompensa: float,
        estado_siguiente: EstadoQL,
    ) -> None:
        q_actual = self.valor_q(estado, accion)
        mejor_siguiente = self.valor_q(estado_siguiente, self.mejor_accion(estado_siguiente))
        objetivo = recompensa + self.gamma * mejor_siguiente
        nuevo_q = q_actual + self.alpha * (objetivo - q_actual)
        self.q_table[(estado, accion)] = nuevo_q
