from pydantic import BaseModel

class AgentConfig(BaseModel):
    state_dim: int = 6
    action_dim: int = 3
    max_action: float = 1.0
    
    # TDD-ND Hyperparameters
    lr_actor: float = 3e-4
    lr_critic: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005 # Target network update rate
    beta: float = 0.5 # TDD min(Q1, Q2) vs Q3 weight
    
    buffer_size: int = 100000
    batch_size: int = 256
    
    # Noise Decay properties for ND
    exploration_noise_start: float = 0.25
    exploration_noise_end: float = 0.05
    exploration_decay_steps: int = 100000  # in total environment steps (~140 episodes at 719 steps/ep)
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    policy_freq: int = 2
