import numpy as np
import random
from collections import namedtuple, deque
from typing import List
from segment_tree import SegmentTree,SumSegmentTree,MinSegmentTree
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class PrioritizedReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, alpha):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.buffer_size = buffer_size
        self.experiences = deque(maxlen=buffer_size) 
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.position = 0
        self.alpha = alpha
        self.max_priority, self.tree_ptr = 1.0, 0
        tree_capacity = 1
        while tree_capacity < self.buffer_size:
            tree_capacity *= 2
        self.sum_tree = SumSegmentTree(tree_capacity)
        self.min_tree = MinSegmentTree(tree_capacity)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.experiences.append(e)
        self.sum_tree[self.tree_ptr] = self.max_priority ** self.alpha
        self.min_tree[self.tree_ptr] = self.max_priority ** self.alpha
        self.tree_ptr = (self.tree_ptr + 1) % self.buffer_size
        
    
    def sample(self, beta):
        """Randomly sample a batch of experiences from memory."""
        indices = self._sample_proportional()
        experiences = []
        for i in indices:
            experiences.append(self.experiences[i])
        states = torch.FloatTensor([experience.state for experience in experiences]).to(device)
        next_states = torch.FloatTensor([experience.next_state for experience in experiences]).to(device)
        actions = torch.LongTensor([experience.action for experience in experiences]).unsqueeze(1).to(device)
        rewards = torch.FloatTensor([experience.reward for experience in experiences]).unsqueeze(1).to(device)
        dones = [experience.done for experience in experiences]
        dones_tensor = torch.FloatTensor(np.array([x=='TRUE' for x in dones]).astype(np.uint8)).unsqueeze(1).to(device)
        weights = torch.FloatTensor(np.array([self._calculate_weight(i, beta) for i in indices])).unsqueeze(1).to(device)
        return (states, actions, rewards, next_states,dones_tensor,weights,indices) 

    def update_priorities(self, indices, priorities):
        """Update priorities of sampled transitions."""
        # assert len(indices) == len(priorities)

        for idx, priority in zip(indices, priorities):

            self.sum_tree[idx] = priority ** self.alpha
            self.min_tree[idx] = priority ** self.alpha

            self.max_priority = max(self.max_priority, priority)
            
    def _sample_proportional(self):
        """Sample indices based on proportions."""
        indices = []
        p_total = self.sum_tree.sum(0, len(self) - 1)
        segment = p_total / self.batch_size
        
        for i in range(self.batch_size):
            a = segment * i
            b = segment * (i + 1)
            upperbound = random.uniform(a, b)
            idx = self.sum_tree.retrieve(upperbound)
            indices.append(idx)
            
        return indices
    
    def _calculate_weight(self, idx, beta):
        """Calculate the weight of the experience at idx."""
        # get max weight
        p_min = self.min_tree.min() / self.sum_tree.sum()
        max_weight = (p_min * len(self)) ** (-beta)
        
        # calculate weights
        p_sample = self.sum_tree[idx] / self.sum_tree.sum()
        weight = (p_sample * len(self)) ** (-beta)
        weight = weight / max_weight
        
        return weight

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.experiences)