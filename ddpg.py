import random
import copy
from collections import namedtuple, deque
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from model import Actor, Critic
from noise import OUNoise, PSNoise
from replaybuffers import ReplayBuffer, ReplayBufferPE

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 128       # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 2e-4         # learning rate of the actor 
LR_CRITIC = 1e-3        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay
BETA_INITIAL = 0.0      # Prioritized Experience Replay initial weight decay
BETA_INCREMENT = 1e-6   # Step-wise Beta increment
ALPHA = 0.3             # Prioritized Experience Replay weight sampling importance
LEARN_EVERY = 1         # Learn every LEARN_EVERY steps
STEPS_UPDATE = 1        # Update Actor and Critic nets every STEPS_UPDATE steps
ADAPT_EVERY = 100       # Frequency for adapting Parameter Space Noise weights

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""
    
    def __init__(self, state_size, action_size, random_seed, add_noise = True, PER = False, PSN = True):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)
        self.add_noise = add_noise

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size, random_seed).to(device)
        self.critic_target = Critic(state_size, action_size, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        # Noise process
        self.PSN = PSN
        if self.add_noise:
          if self.PSN:
            self.noise = PSNoise(state_size, action_size, random_seed)
          else:
            self.noise = OUNoise(action_size, random_seed)
          
        # Replay memory
        self.PER = PER
        if self.PER:
          self.memory = ReplayBufferPE(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed, alpha = ALPHA)
          self.beta = BETA_INITIAL
        else:
          self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)
        
        # Initialize learning steps 
        self.learn_step = 0  
    
    def reset(self):
        if self.add_noise:
          if self.PSN:
            self.noise.reset(self.actor_local)
          else:
            self.noise.reset()    
    
    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        self.memory.add(state, action, reward, next_state, done)
        
        if len(self.memory) > BATCH_SIZE:
            # Learn, if enough samples are available in memory for number of timesteps
            for _ in range(STEPS_UPDATE):
              experiences = self.memory.sample()
              self.learn(experiences, GAMMA)
        
        # LEARN_EVERY time steps.
        '''
        self.learn_step = (self.learn_step + 1) % LEARN_EVERY
        if self.learn_step == 0:
            # Learn, if enough samples are available in memory
            if len(self.memory) > BATCH_SIZE:
                for _ in range(STEPS_UPDATE):
                  experiences = self.memory.sample()
                  self.learn(experiences, GAMMA)            
        '''
    def act(self, state, epsilon = 1, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        
        #If add_noise = True:
        if self.add_noise:
          #Add AS or PS noise:
          if self.PSN:
            # Parameter Space Noise
            if len(self.memory) > BATCH_SIZE:
              # PS noise needs to sample from memory to perturbate actor weights
              self.noise.update_noise(self.actor_local, states_batch = self.memory.sample()[0])
            with torch.no_grad():
              action = self.actor_local(state).cpu().data.numpy()
            self.actor_local.train()
          # Action Space Noise
          else:
            with torch.no_grad():
              action = self.actor_local(state).cpu().data.numpy()
            self.actor_local.train()
            action += self.noise.sample()
        #If add_noise = False, no noise is added
        else:
          with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()

        #For all cases, return clipped action      
        return np.clip(action, -1, 1)

    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples."""    
        
        states, actions, rewards, next_states, dones = experiences
        
        #Clip rewards
        #rewards_ = torch.clamp(rewards, min=-1., max=1.)
        
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Compute Q expected
        Q_expected = self.critic_local(states, actions)
        
        # Compute critic loss        
        if self.PER:
          # Update Beta
          self.beta += BETA_INCREMENT
          # Get RB weights
          weights = self.memory.get_weights(self.beta)
          # Clip abs(TD_errors)
          TD_errors = torch.clamp(torch.abs(Q_targets - Q_expected), min=0., max=1.)
          # Update replay buffer with proportional probs
          self.memory.update_priorities(TD_errors)
          #compute weighted mse loss     
          critic_loss = torch.mean(weights * (Q_expected - Q_targets) ** 2)
        else:
          #compute mse loss  
          critic_loss = F.mse_loss(Q_expected, Q_targets)
        
        # Minimize critic loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # Clip critic gradient
        torch.nn.utils.clip_grad_norm(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean() #NEGATIVE: gradiet ascent
        
        # Minimize actor loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft updates for target networks
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
