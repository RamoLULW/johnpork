import random
import numpy as np
import campo2d_p2p

class QLearningAgent:
    def __init__(self, n_states, n_actions, learning_rate=0.1, discount_factor = 0.9, exploration_rate=0.1):
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


class CampoGymWrapper:
    def __init__(self, pars, controlled_tractor_index=0):
        self.model = campo2d_p2p.CampoModel(pars)
        # inicializar el modelo (agentpy usa setup en run; aquí llamamos setup manualmente)
        self.model.setup()
        # elegir el tractor a controlar (primer tractor)
        self.ctrl_idx = controlled_tractor_index
        self.ctrl_id = self.model.tractors[self.ctrl_idx].id
        self.N = self.model.N

    def reset(self):
        # reinstancia visitado/plantas y recoloca tractores (usa setup)
        self.model = campo2d_p2p.CampoModel(self.model.p)
        self.model.setup()
        # estado codificado: posición del tractor controlado -> idx
        x, y = self.model.tractors[self.ctrl_idx].pos_xy
        return y * self.N + x

    def step(self, action_int):
        """
        action_int: 0..3 move in DIRS
        devuelve: next_state (index), reward (float), done (bool)
        """
        # calcular estado y métricas previas para reward
        ag = self.model.tractors[self.ctrl_idx]
        x0, y0 = ag.pos_xy
        prev_visit = bool(self.model.visitado[y0, x0])
        prev_gas = ag.gasolina

        # pasar acción externa (ag.id es clave)
        external_actions = {self.ctrl_id: action_int}
        self.model.step(external_actions)

        # después del paso, leer nuevo estado
        x1, y1 = ag.pos_xy
        next_state = y1 * self.N + x1
        # reward simple: +1 si ahora visitó una nueva celda; restar gasto de gasolina
        new_visit = bool(self.model.visitado[y1, x1])
        reward = 0.0
        if new_visit and not prev_visit:
            reward += 1.0
        reward += (ag.gasolina - prev_gas)  # notará que esto será negativo por gasto

        # determinar done: usa la lógica del modelo (podrías usar model.history or flags)
        done = False
        # ejemplo: si model.stop() fue llamado el modelo habrá parado; detectamos por tick limit
        if len(self.model.history["tick"]) >= self.model.max_ticks:
            done = True

        return next_state, reward, done

pars = {'N': 12, 'T': 4, 'capacity': (12*12)//(4*4), 'seed': 42, 'max_ticks': 200, 'gasolina_max': 152, 'costo_normal': 10.9/4, 'costo_mojado': (10.9/4) * 3}

env = CampoGymWrapper(pars, controlled_tractor_index=0)
n_states = pars['N'] * pars['N']
n_actions = 4
agent = QLearningAgent(n_states=n_states, n_actions=n_actions)
episodes = 1000

for episode in range(episodes):
    state = env.reset()
    done = False

    while not done:
        action = agent.chose_action(state)
        next_state,reward,done= env.step(action)
        agent.update_q_value(state,action,reward,next_state)
        satte=next_state


        