import random
import numpy as np
import sys
import os
from typing import Tuple

current_dir = os.path.dirname(os.path.abspath(__file__))  
project_root = os.path.dirname(current_dir)  

sys.path.insert(0, project_root)
import campo2d_p2p

class QLearningAgent:
    def __init__(self, n_states, n_actions, learning_rate=0.1, discount_factor = 0.9, exploration_rate=1.0):
        self.q_table=np.zeros((n_states, n_actions))
        self.learning_rate=learning_rate
        self.discount_factor=discount_factor
        self.exploration_rate=exploration_rate
        self.n_actions = n_actions
    
    def chose_action(self,state):
        if random.uniform(0,1) < self.exploration_rate:
            return random.randint(0,self.n_actions-1)
        else:
            return np.argmax(self.q_table[state])
        
    def update_q_value(self,state,action,reward,next_state):
        max_future_q = np.max(self.q_table[next_state])
        current_q = self.q_table[state][action]
        self.q_table[state][action]= current_q + self.learning_rate*(reward + self.discount_factor * max_future_q - current_q)

    def decay_exploration(self):
        self.exploration_rate = max(0.01, self.exploration_rate * 0.9995)


class CampoGymWrapper:
    def __init__(self, pars, controlled_tractor_index=0):
        self.model = campo2d_p2p.CampoModel(pars)
        self.model.setup()
        self.ctrl_idx = controlled_tractor_index
        self.ctrl_id = self.model.tractors[self.ctrl_idx].id
        self.N = self.model.N
        self.current_tick = 0
        self.max_ticks = pars['max_ticks']
        self.pars = pars

    def reset(self):
        self.model = campo2d_p2p.CampoModel(self.pars)
        self.model.setup()
        self.current_tick = 0
        x, y = self.model.tractors[self.ctrl_idx].pos_xy
        return y * self.N + x

    def step(self, action: int) -> Tuple[int, float, bool]:
        ag = self.model.tractors[self.ctrl_idx]
        
        x0, y0 = ag.pos_xy
        prev_visited = bool(self.model.visitado[y0, x0])
        prev_gas = ag.gasolina
        prev_load = ag.load
        
        DIRS = [(0, -1), (1, 0), (0, 1), (-1, 0)]  
        dx, dy = DIRS[action]
        next_pos = (ag.pos_xy[0] + dx, ag.pos_xy[1] + dy)
        
        if not campo2d_p2p.in_bounds(next_pos, self.N):
            reward = -5.0
            x, y = ag.pos_xy
            next_state_idx = y * self.N + x
            return next_state_idx, reward, False
        
        original_ruta = ag.ruta.copy()
        ag.ruta = []
        for i, other_ag in enumerate(self.model.tractors):
            if i != self.ctrl_idx:
                other_ag.intent = ("AFK", None)
        ag.intent = ("INTENT_MOVE", next_pos)
        
        self.model.step()
        ag.ruta = original_ruta
        self.current_tick += 1
        
        x1, y1 = ag.pos_xy
        current_visited = bool(self.model.visitado[y1, x1])
        new_visit = current_visited and not prev_visited
        
        reward = 0.0
        
        if new_visit:
            reward += 20.0

            moved = (x1, y1) != (x0, y0)
            if moved:
                reward += 0.2
            
            plant_type = self.model.plantas[y1, x1]
            if plant_type == 1:  
                reward += 10.0
            elif plant_type == 2:  
                reward += 5.0
            elif plant_type == 3:  
                reward += 2.0
        
        gas_used = prev_gas - ag.gasolina
        reward -= gas_used * 0.05
        
        if ag.pos_xy == ag.estacion and prev_load > 0 and ag.load == 0:
            reward += 20.0
        
        if not new_visit and self.model.plantas[y1, x1] != 4:  
            reward -= 2.0
        
        if ag.gasolina <= 0:
            reward -= 50.0
        
        next_state_idx = y1 * self.N + x1
        
        done = False
        if self.current_tick >= self.max_ticks:
            done = True
        if ag.gasolina <= 0:
            done = True
        if ag.ruta_idx >= len(ag.ruta) and not ag.a_estacion:
            done = True
            reward += 100.0  
        
        cells_visited = np.sum(self.model.visitado)
        total_cells = self.N * self.N - 1 
        completion = cells_visited / total_cells

        if completion >= 0.5:  
            reward += 50.0
        if completion >= 0.9: 
            reward += 200.0
            done = True
        
        return next_state_idx, reward, done
            


pars = {'N': 12, 'T': 4, 'capacity': 144, 'seed': None, 'max_ticks': 1000, 'gasolina_max': 152, 'costo_normal': 10.9/4, 'costo_mojado': (10.9/4) * 3}

env = CampoGymWrapper(pars, controlled_tractor_index=0)
n_states = pars['N'] * pars['N']
n_actions = pars['T']
agent = QLearningAgent(n_states=n_states, n_actions=n_actions)
episodes = 10000
episode_rewards = []
episode_steps = []

for episode in range(episodes):
    state = env.reset()
    done = False
    total_reward = 0
    step = 0

    while not done:
        action = agent.chose_action(state)
        next_state,reward,done= env.step(action)
        agent.update_q_value(state,action,reward,next_state)
        state=next_state
        total_reward += reward
        step += 1
        if step >= 300:
            done = True
        
    agent.decay_exploration()    
    episode_rewards.append(total_reward)
    episode_steps.append(step)

    if (episode + 1) % 100 == 0:
        cells_visited = np.sum(env.model.visitado)
        completion = cells_visited / (pars['N'] * pars['N'] - 1) * 100
        print(f"Ep {episode + 1} | Reward: {total_reward:.1f} | Steps: {step} | Cells: {cells_visited}/143 ({completion:.1f}%) | Explore: {agent.exploration_rate:.3f}")        