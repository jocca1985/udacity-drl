import numpy as np
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class PrioritizedReplayBuffer(object):
    def __init__(self, capacity, batch_size, prob_alpha=0.6):
        self.prob_alpha = prob_alpha
        self.capacity   = capacity
        self.batch_size = batch_size
        self.buffer     = []
        self.pos        = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)
    
    def add(self, state, action, reward, next_state, done):
        assert state.ndim == next_state.ndim
        # state      = np.expand_dims(state, 0)
        # next_state = np.expand_dims(next_state, 0)
        
        max_prio = self.priorities.max() if self.buffer else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)
        
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity
    
    def sample(self, beta=0.4):
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]
        
        probs  = prios ** self.prob_alpha
        probs /= probs.sum()
        
        indices = np.random.choice(len(self.buffer), self.batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        
        total    = len(self.buffer)
        weights  = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights  = torch.FloatTensor(np.array(weights, dtype=np.float32)).unsqueeze(1).to(device)
        
        batch       = list(zip(*samples))
        
        states      = torch.FloatTensor(batch[0]).to(device)
        actions     = torch.LongTensor(batch[1]).unsqueeze(1).to(device)
        rewards     = torch.FloatTensor(batch[2]).unsqueeze(1).to(device)
        next_states = torch.FloatTensor(batch[3]).to(device)
        dones       = torch.FloatTensor(np.array([x=='TRUE' for x in batch[4]]).astype(np.uint8)).unsqueeze(1).to(device)
        
        return states, actions, rewards, next_states, dones, weights, indices
    
    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio

    def __len__(self):
        return len(self.buffer)