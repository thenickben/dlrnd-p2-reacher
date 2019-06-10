import numpy as np
import random
import copy
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.2, sigma=0.1):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self, epsilon = 1.0):
        """Update internal state and return it as a noise sample."""
        x = self.state
        #dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        dx = epsilon*(self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))]))
        self.state = x + dx
        return self.state

class PSNoise:
    """Parameter Space Noise"""

    def __init__(self, state_size, action_size, seed, sigma=0.1, scalar = 1.01, treshold = 0.3):
      """Initialize parameters and noise process."""
      self.state_size = state_size
      self.action_size = action_size
      self.sigma = sigma
      self.treshold = treshold
      self.scalar = scalar
      self.adapt_step = 0
      self.policy_pert = Actor(self.state_size, self.action_size, seed).to(device)
           
    def reset(self, policy):
      """Perturbate actor's fc layer at the beginning of the episode"""
      with torch.no_grad():
        for param in policy.parameters():
          noise = torch.randn(param.size()) * self.sigma
          param.add_(noise.float().to(device))

        
    def get_distance(self, action_size, policy, states_batch, sigma):
      """Computes distance between perturbed and non-perturbed policies""" 
      actions_no_pert = policy(states_batch).cpu().data.numpy()
        
      #copy weights
      with torch.no_grad():
        for param_nonpert, param_pert in zip(policy.parameters(), self.policy_pert.parameters()):
          param_pert.data.copy_(param_nonpert.data)
      
      
      #perturbed policy
      with torch.no_grad():
        for param_pert in self.policy_pert.parameters():
          noise = torch.randn(param_pert.size()) * self.sigma
          param_pert.add_(noise.float().to(device))
      
      actions_pert = self.policy_pert(states_batch).cpu().data.numpy()
      
      distance = np.sqrt(np.mean((actions_no_pert - actions_pert)**2))
      
      return distance 
      
      
    def adapt(self, distance):
      """Updates sigma"""
      if distance <= self.treshold:
        self.sigma = self.scalar * self.sigma
      else:
        self.sigma = self.sigma / self.scalar
      self.treshold = self.sigma
    
    
    def update_noise(self, policy, states_batch):
        """Adaptative scaling"""
                          
        self.adapt_step = (self.adapt_step + 1) % ADAPT_EVERY
        #change sigma (adapt)
        if self.adapt_step == 0:   
          #compute distance
          distance = self.get_distance(self.action_size, policy, states_batch, sigma = self.sigma)
          
          self.distance = distance
          
          #update sigma
          self.adapt(distance)
          
          #perturbed policy with updated sigma
          with torch.no_grad():
            for param in agent.actor_local.parameters():
              noise = torch.randn(param.size()) * self.sigma
              param.add_(noise.float().to(device))