from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, Tuple, List
import random
import numpy as np

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
    epsilon: float = 1.0
    epsilon_min: float = 0.01
    epsilon_decay: float = 0.9995
    q_table: Dict[EstadoQL, np.ndarray] = field(default_factory=dict)

    def acciones_disponibles(self) -> List[AccionTractor]:
        return list(AccionTractor)

    def _ensure_state(self, estado: EstadoQL) -> np.ndarray:
        if estado not in self.q_table:
            self.q_table[estado] = np.zeros(len(self.acciones_disponibles()), dtype=float)
        return self.q_table[estado]

    def _accion_to_index(self, accion: AccionTractor) -> int:
        return self.acciones_disponibles().index(accion)

    def _index_to_accion(self, idx: int) -> AccionTractor:
        return self.acciones_disponibles()[idx]

    def valor_q(self, estado: EstadoQL, accion: AccionTractor) -> float:
        fila = self._ensure_state(estado)
        idx = self._accion_to_index(accion)
        return float(fila[idx])

    def mejor_accion(self, estado: EstadoQL) -> AccionTractor:
        fila = self._ensure_state(estado)
        idx = int(np.argmax(fila))
        return self._index_to_accion(idx)

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
        fila_actual = self._ensure_state(estado)
        fila_siguiente = self._ensure_state(estado_siguiente)
        a_idx = self._accion_to_index(accion)
        max_future_q = float(np.max(fila_siguiente))
        current_q = float(fila_actual[a_idx])
        nuevo_q = current_q + self.alpha * (recompensa + self.gamma * max_future_q - current_q)
        fila_actual[a_idx] = nuevo_q

    def decaer_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
