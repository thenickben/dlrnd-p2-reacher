import numpy as np
from collections import namedtuple, deque
import random
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

class ReplayBufferPE:
    """Fixed-size buffer to store experience tuples and sample with proportional prioritization."""

    def __init__(self, action_size, buffer_size, batch_size, seed, alpha):
        """Initialize a ReplayBuffer object.
        Params
        ======
            TODO
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)      # internal memory for S,A,R,S',done (deque)
        self.priorities = deque(maxlen=buffer_size)  # internal memory priorities (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        self.alpha = max(0., alpha)
        self.max_priority = 1.0**self.alpha
        self.cum_priorities = 0.
        self.epsilon = 1e-6
        self.experiences_idx = []
   
    def add(self, state, action, reward, next_state, done):
          """Add a new experience to memory."""
          #Add experience
          e = self.experience(state, action, reward, next_state, done)
          self.memory.append(e)
          #Update priorities and cum_priorities
          self.priorities.append(self.max_priority)
          # accumulate the priorities
          self.cum_priorities = np.sum(self.priorities)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        probs = np.array(self.priorities) / self.cum_priorities
        len_memory = len(self.memory)
        self.experiences_idx = np.random.choice(len_memory, min(len_memory, self.batch_size), p = probs)
        experiences = [self.memory[i] for i in self.experiences_idx]

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def get_weights(self, beta):
      "Returns Importance Sampling weights wi"
      len_memory = len(self.memory)
      max_weight = (len_memory * min(self.priorities) / self.cum_priorities) ** beta
      weights = np.array([((len_memory * (self.priorities[i] / self.cum_priorities))**(-beta))/max_weight for i in self.experiences_idx])
      return torch.from_numpy(weights).float().to(device)
    
    def update_priorities(self, td_errors):
        """ Update priorities of sampled transitions """

        for i, td_error in zip(self.experiences_idx, td_errors):
            td_error = float(td_error)
            self.cum_priorities -= self.priorities[i]
            self.priorities[i] = ((abs(td_error) + self.epsilon) ** self.alpha)
            self.cum_priorities += self.priorities[i]
        #update max priority value
        self.max_priority = max(self.priorities)
    
    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)