import numpy as np
import torch
from collections import deque
import torch.nn as nn
import torch.nn.functional as F
from unityagents import UnityEnvironment
import time
import matplotlib.pyplot as plt

class EvolutionStrategies(torch.nn.Module):
    def __init__(self, inputs, outputs):
        super(EvolutionStrategies, self).__init__()
        fc1_units = 100
        fc2_units = 100
        self.bn1 = nn.BatchNorm1d(fc1_units) 
        self.linear1 = nn.Linear(inputs, fc1_units)
        self.linear2 = nn.Linear(fc1_units, fc2_units)
        self.linear3 = nn.Linear(fc2_units, outputs)
        self.population_size = 80
        self.sigma = 0.25
        self.min_sigma = 0.01
        self.learning_rate = 0.5
        self.min_learning_rate = 0.001
        self.decay = 0.999
        self.counter = 0
        self.rewards = []
        self.score_tracking = deque(maxlen = 100)
        self.master_weights = []

        for param in self.parameters():
            self.master_weights.append(param.data)
        self.populate()

    def forward(self, x):
        x = F.elu(self.bn1(self.linear1(state)))
        x = F.elu(self.linear2(x))
        return F.tanh(self.linear3(x))

    def populate(self):
        self.population = []
        for _ in range(self.population_size):
            x = []
            for param in self.parameters():
                x.append(np.random.randn(*param.data.size()))
            self.population.append(x)

    def add_noise_to_weights(self):
        for i, param in enumerate(self.parameters()):
            noise = torch.from_numpy(self.sigma * self.population[self.counter][i]).float()
            param.data = self.master_weights[i] + noise
        self.counter += 1

    def log_reward(self, reward):
        # When we've got enough rewards, evolve the network and repopulate
        self.rewards.append(reward)

        if len(self.rewards) >= self.population_size:
            self.counter = 0
            self.evolve()
            self.populate()
            self.rewards = []
        self.add_noise_to_weights()

    def evolve(self):
        # Multiply jittered weights by normalised rewards and apply to network
        if np.std(self.rewards) != 0:
            normalized_rewards = (self.rewards - np.mean(self.rewards)) / np.std(self.rewards)
            for index, param in enumerate(self.parameters()):
                A = np.array([individual[index] for individual in self.population])
                rewards_pop = torch.from_numpy(np.dot(A.T, normalized_rewards).T).float()
                param.data = self.master_weights[index] + self.learning_rate / (self.population_size * self.sigma) * rewards_pop

                self.master_weights[index] = param.data

def reset():
    env_info = env.reset(train_mode=True)[brain_name]
    return env_info.vector_observations

def step(action):
    env_info = env.step(action)[brain_name]        # send the action to the environment
    next_state = env_info.vector_observations   # get the next state
    reward = env_info.rewards                  # get the reward
    done = env_info.local_done
    return next_state, reward, done

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
env = UnityEnvironment(file_name='./Tennis.app',no_graphics=True)
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
# model.load_state_dict(torch.load('checkpoint.pth'))
state = env.reset()
episodes = 30000

max_rew = 0
history = []

for episode in range(episodes):
    episode_reward = np.zeros(2) 
    for i in range(100):
        state = reset()
        while True:
            state = torch.from_numpy(state).float().to(device)
            model.eval()
            with torch.no_grad():
                action = model(state).cpu().data.numpy().reshape(2,2)
            model.train()
            action = np.clip(action, -1, 1)
            state, reward, done = step(action)
            episode_reward += reward
            if np.any(done):
                break
    current_reward = max(episode_reward/100)

    if current_reward > max_rew:
        max_rew = current_reward
        history.append(current_reward)
        torch.save(model.state_dict(), 'checkpoint.pth')

    model.log_reward(current_reward)
    if (episode % model.population_size == 0):
        print(episode/model.population_size, max_rew)
    if max_rew > 0.5:
        print(episode/model.population_size, max_rew)
        print("Solved")
        fig = plt.figure(figsize=(16, 12))
        ax = fig.add_subplot(111)
        plt.plot(np.arange(len(history)), history)
        plt.xlabel('Episode number')
        plt.ylabel('Score')
        plt.show()
        break

