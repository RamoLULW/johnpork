import random
import numpy as np
import campo2d_p2p

class QLearningAgent:
    def __init__(self, learning_rate=0.1, discount_factor = 0.9, exploration_rate=0.1):
        self.q_table=np.zeros((4,4,4))
        self.learning_rate=learning_rate
        self.discount_factor=discount_factor
        self.exploration_rate=exploration_rate
    
    def chose_action(self,state):
        if random.uniform(0,1) < self.exploration_rate:
            return random.randint(0,3)
        else:
            return np.argmax(self.q_table[state])
        
    def update_q_value(self,state,action,reward,next_state):
        max_future_q = np.max(self.q_table[next_state])
        current_q = self.q_table[state][action]
        self.q_table[state][action]= current_q + self.learning_rate*(reward + self.discount_factor * max_future_q - current_q)



pars = {'N': 12, 'T': 4, 'capacity': (12*12)//(4*4), 'seed': 42, 'max_ticks': 200, 'gasolina_max': 152, 'costo_normal': 10.9/4, 'costo_mojado': (10.9/4) * 3}

env = campo2d_p2p.CampoModel(pars)
agent = QLearningAgent()

episodes = 1000

for episode in range(episodes):
    state = env.reset()
    done = False

    while not done:
        action = agent.chose_action(state)
        next_state,reward,done= env.step(action)
        agent.update_q_value(state,action,reward,next_state)
        satte=next_state


        