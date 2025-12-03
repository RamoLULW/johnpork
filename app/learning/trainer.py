from app.learning.qlearning import QLearningPolicy
from app.core.types import SimulationRequestDTO
from app.simulation.orchestrator import run_simulation

def train_qlearning_policy(
    req: SimulationRequestDTO
):
    num_episodes = 1000
    policy = QLearningPolicy(seed=req.seed)

    best_cost = float('inf')
    best_result = None
    
    for episode in range(num_episodes):
        new_req = req.copy(update={'seed': req.seed + episode})

        result = run_simulation(new_req, policy = policy)
        
        current_cost = result.stats.total_fuel_used / max(1, result.stats.good_harvested)

        
        if current_cost < best_cost and current_cost !=0:
            best_cost = current_cost
            best_result = result
            print(f"Episode {episode}: cost={best_cost:.2f}, fuel={result.stats.total_fuel_used}, harvested={result.stats.good_harvested}")


        policy.decay_exploration()
        
    return best_result