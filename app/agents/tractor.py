from typing import List, Optional
from app.core.types import TractorInternalState, HighLevelAction, Coord
from app.agents.interfaces import ITractorAgent, IEnvironment, IBlackboard, IHighLevelPolicy
from app.pathfinding.astar import astar


class TractorAgent(ITractorAgent):
    def __init__(self, internal_state: TractorInternalState, env: IEnvironment, blackboard: IBlackboard, policy: Optional[IHighLevelPolicy], use_qlearning: bool):
        self.state = internal_state
        self.env = env
        self.blackboard = blackboard
        self.policy = policy
        self.use_qlearning = use_qlearning
        self.current_action: Optional[HighLevelAction] = None
        self.current_target: Optional[Coord] = None
        self.current_path: List[Coord] = []
        self.prev_state_key: Optional[tuple] = None
        self.prev_action: Optional[HighLevelAction] = None

    def get_internal_state(self) -> TractorInternalState:
        return self.state

    def _build_state_key(self) -> tuple:
        fuel_ratio = self.state.fuel / self.state.max_fuel
        if fuel_ratio < 0.2:
            fuel_bucket = 0
        elif fuel_ratio < 0.5:
            fuel_bucket = 1
        else:
            fuel_bucket = 2
        load_ratio = self.state.load / self.state.capacity
        if load_ratio < 0.5:
            load_bucket = 0
        elif load_ratio < 0.9:
            load_bucket = 1
        else:
            load_bucket = 2
        has_target = 1 if self.current_target is not None else 0
        return fuel_bucket, load_bucket, has_target

    def _needs_refuel(self) -> bool:
        pos = (self.state.x, self.state.y)
        station = self.env.nearest_station(pos)
        dist = abs(station[0] - pos[0]) + abs(station[1] - pos[1])
        min_step_cost = 1.0
        required = dist * min_step_cost
        safety_factor = 2.0
        return self.state.fuel <= required * safety_factor

    def _choose_action(self) -> HighLevelAction:
        if self._needs_refuel():
            return HighLevelAction.GO_TO_STATION
        if not self.use_qlearning or self.policy is None:
            if self.state.load >= self.state.capacity * 0.9:
                return HighLevelAction.GO_TO_STATION
            if self.current_target is not None:
                return HighLevelAction.GO_TO_PLANT
            return HighLevelAction.EXPLORE
        state_key = self._build_state_key()
        action = self.policy.select_action(state_key, list(HighLevelAction))
        self.prev_state_key = state_key
        self.prev_action = action
        return action

    def _set_target_for_action(self, action: HighLevelAction) -> None:
        if action == HighLevelAction.GO_TO_STATION:
            self.current_target = self.env.nearest_station((self.state.x, self.state.y))
        elif action == HighLevelAction.GO_TO_PLANT:
            pos = (self.state.x, self.state.y)
            if self.current_target is None:
                target = self.blackboard.claim_plant(self.state.id, pos)
                self.current_target = target
        elif action == HighLevelAction.EXPLORE:
            self.current_target = self.env.random_free_cell()
        else:
            self.current_target = None
        if self.current_target is not None:
            path = astar(self.env, (self.state.x, self.state.y), self.current_target)
            self.current_path = path[1:] if path and len(path) > 1 else []

    def _move_one_step(self) -> None:
        if not self.current_path:
            return
        next_pos = self.current_path.pop(0)
        cost = self.env.movement_cost(next_pos)
        if self.state.fuel <= 0:
            return
        self.state.fuel -= cost
        self.state.total_fuel_used += cost
        self.state.x, self.state.y = next_pos

    def _handle_arrival(self) -> float:
        reward = 0.0
        if self.current_target is None:
            return reward
        if (self.state.x, self.state.y) != self.current_target:
            return reward
        if self.env.has_plant(self.current_target):
            good = self.env.harvest_plant(self.current_target)
            if good is not None:
                if good:
                    self.state.harvested_good += 1
                else:
                    self.state.harvested_bad += 1
                self.state.load += 1
                reward += 5.0 if good else 1.0
            self.blackboard.remove_plant(self.current_target)
        cell_is_station = self.env.nearest_station((self.state.x, self.state.y)) == (self.state.x, self.state.y)
        if cell_is_station:
            self.state.fuel = self.state.max_fuel
            self.state.load = 0
            reward += 2.0
        self.current_target = None
        self.current_path = []
        return reward

    def step(self) -> None:
        if self.state.fuel <= 0:
            return
        action = self._choose_action()
        if self.current_target is None:
            self._set_target_for_action(action)
        if self.current_target is not None and not self.current_path:
            path = astar(self.env, (self.state.x, self.state.y), self.current_target)
            self.current_path = path[1:] if path and len(path) > 1 else []
        self._move_one_step()
        reward = self._handle_arrival()
        if self.use_qlearning and self.policy is not None and self.prev_state_key is not None and self.prev_action is not None:
            next_state_key = self._build_state_key()
            reward -= 0.1
            self.policy.update(self.prev_state_key, self.prev_action, reward, next_state_key)
