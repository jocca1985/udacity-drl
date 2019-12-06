import numpy as np
import torch
from collections import deque
import torch.nn as nn
import torch.nn.functional as F
from unityagents import UnityEnvironment

class EvolutionStrategies(torch.nn.Module):
    def __init__(self, inputs, outputs):
        super(EvolutionStrategies, self).__init__()
        fc1_units = 100
        fc2_units = 100
        self.bn1 = nn.BatchNorm1d(fc1_units) 
        self.linear1 = nn.Linear(inputs, fc1_units)
        self.linear2 = nn.Linear(fc1_units, fc2_units)
        self.linear3 = nn.Linear(fc2_units, outputs)
        

    def forward(self, x):
        x = F.elu(self.bn1(self.linear1(state)))
        x = F.elu(self.linear2(x))
        return F.tanh(self.linear3(x))


def reset():
    env_info = env.reset(train_mode=False)[brain_name]
    return env_info.vector_observations

def step(action):
    env_info = env.step(action)[brain_name]        # send the action to the environment
    next_state = env_info.vector_observations   # get the next state
    reward = env_info.rewards                  # get the reward
    done = env_info.local_done
    return next_state, reward, done

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
env = UnityEnvironment(file_name='./Tennis.app')
# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]
# reset the environment
env_info = env.reset(train_mode=True)[brain_name]
# number of agents
num_agents = len(env_info.agents)
# size of each action
action_size = brain.vector_action_space_size
# examine the state space 
states = env_info.vector_observations
state_size = states.shape[1]

model = EvolutionStrategies(inputs=state_size, outputs=action_size)
model.load_state_dict(torch.load('checkpoint_best.pth'))
model.eval()
for i in range(10):
    state = reset()
    episode_reward = np.zeros(2)
    while True:
        state = torch.from_numpy(state).float().to(device)
        with torch.no_grad():
            action = model(state).cpu().data.numpy().reshape(2,2)
        action = np.clip(action, -1, 1)
        state, reward, done = step(action)
        episode_reward += reward
        if np.any(done):
            break
            print(episode_reward)