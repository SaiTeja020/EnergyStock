from pydantic import BaseModel

class AgentConfig(BaseModel):
    state_dim: int = 6
    action_dim: int = 3
    max_action: float = 1.0
    
    # SAC (Soft Actor-Critic) Hyperparameters
    lr_actor: float = 3e-4
    lr_critic: float = 3e-4
    lr_alpha: float = 3e-4 # for temperature tuning
    gamma: float = 0.99
    tau: float = 0.005 # Target network update rate
    alpha: float = 0.2  # Initial entropy coefficient
    
    buffer_size: int = 100000
    batch_size: int = 256
    
    # Exploration / Target properties
    exploration_steps: int = 5000 # initial random steps
    policy_freq: int = 2
    target_entropy: float = -3.0 # -action_dim is a good heuristic
    
    # Weights for BESS multi-objective (used if you want to bias the agent)
    # rewards = alpha_ea*r_ea + alpha_fr*r_fr + alpha_ps*r_ps
