from typing import Dict, Tuple
from app.core.types import HighLevelAction
from app.agents.interfaces import IHighLevelPolicy
import random
import numpy as np

class QLearningPolicy(IHighLevelPolicy):
    def __init__(self, alpha: float = 0.1, gamma: float = 0.95, epsilon: float = 0.1, seed: int = 42):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q: Dict[Tuple[tuple, HighLevelAction], float] = {}
        self.rng = random.Random(seed)

    def select_action(self, state_key: tuple, available_actions):
        if self.rng.random() < self.epsilon:
            return self.rng.choice(available_actions)
        best = None
        best_q = None
        for a in available_actions:
            qv = self.q.get((state_key, a), 0.0)
            if best is None or qv > best_q:
                best = a
                best_q = qv
        return best

    def update(self, prev_state_key: tuple, action: HighLevelAction, reward: float, next_state_key: tuple) -> None:
        key = (prev_state_key, action)
        old_q = self.q.get(key, 0.0)
        max_next = None
        for a in HighLevelAction:
            qv = self.q.get((next_state_key, a), 0.0)
            if max_next is None or qv > max_next:
                max_next = qv
        if max_next is None:
            max_next = 0.0
        new_q = old_q + self.alpha * (reward + self.gamma * max_next - old_q)
        self.q[key] = new_q

    def decay_exploration(self):
        self.epsilon = max(0.01, self.epsilon * 0.9995)
