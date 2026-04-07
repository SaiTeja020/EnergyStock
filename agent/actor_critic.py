import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

class Actor(nn.Module):
    """
    Gaussian Stochastic Actor for SAC.
    """
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        
        self.mu = nn.Linear(256, action_dim)
        self.log_std = nn.Linear(256, action_dim)
        
        self.max_action = max_action
        self.epsilon = 1e-6

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        
        mu = self.mu(a)
        log_std = self.log_std(a)
        log_std = torch.clamp(log_std, min=-20, max=2) # standard clip for SAC
        return mu, log_std

    def sample(self, state):
        mu, log_std = self.forward(state)
        std = log_std.exp()
        
        normal = Normal(mu, std)
        x_t = normal.rsample() # reparameterization trick
        y_t = torch.tanh(x_t)
        action = y_t * self.max_action
        
        # Enforce action bounds (tanh squashing)
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.max_action * (1 - y_t.pow(2)) + self.epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mu = torch.tanh(mu) * self.max_action
        
        return action, log_prob, mu

class Critic(nn.Module):
    """
    Twin Critic architecture to prevent overestimation in SAC.
    """
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        # Q1 Architecture
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

        # Q2 Architecture
        self.l4 = nn.Linear(state_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

class SAC_Agent:
    """
    Soft Actor-Critic agent with automatic temperature (Alpha) tuning.
    """
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.actor = Actor(config.state_dim, config.action_dim, config.max_action).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.lr_actor)

        self.critic = Critic(config.state_dim, config.action_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config.lr_critic)

        # Automatic Entropy Tuning (Alpha)
        self.target_entropy = config.target_entropy
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=config.lr_alpha)

        self.total_it = 0

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        if evaluate:
            _, _, action = self.actor.sample(state)
        else:
            action, _, _ = self.actor.sample(state)
        return action.detach().cpu().numpy().flatten()

    def train(self, replay_buffer):
        self.total_it += 1

        state, action, next_state, reward, not_done = replay_buffer.sample(self.config.batch_size)

        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.actor.sample(next_state)
            qf1_next_target, qf2_next_target = self.critic_target(next_state, next_state_action)
            
            # min(Q1, Q2) - alpha * log_pi
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward + (not_done * self.config.gamma * min_qf_next_target)

        # Critic update
        qf1, qf2 = self.critic(state, action)
        qf1_loss = F.mse_loss(qf1, next_q_value)
        qf2_loss = F.mse_loss(qf2, next_q_value)
        qf_loss = qf1_loss + qf2_loss

        self.critic_optimizer.zero_grad()
        qf_loss.backward()
        self.critic_optimizer.step()

        # Actor update
        pi, log_pi, _ = self.actor.sample(state)
        qf1_pi, qf2_pi = self.critic(state, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        actor_loss = ((self.alpha * log_pi) - min_qf_pi).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Alpha (Temperature) update
        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # Soft update of target networks
        if self.total_it % self.config.policy_freq == 0:
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.config.tau * param.data + (1 - self.config.tau) * target_param.data)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic.pth")
        torch.save(self.actor.state_dict(), filename + "_actor.pth")
        torch.save(self.log_alpha, filename + "_alpha.pth")

    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic.pth"))
        self.critic_target = copy.deepcopy(self.critic)
        self.actor.load_state_dict(torch.load(filename + "_actor.pth"))
        
        if os.path.exists(filename + "_alpha.pth"):
            self.log_alpha = torch.load(filename + "_alpha.pth")

# Alias for backwards compatibility or naming consistency
TDD_ND_Agent = SAC_Agent
