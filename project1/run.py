from unityagents import UnityEnvironment
import numpy as np
from collections import deque
from double_dqn_noisy_priority_agent import Agent 
import torch
import matplotlib.pyplot as plt

# please do not modify the line below
env = UnityEnvironment(file_name="./Banana.app")

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents in the environment
print('Number of agents:', len(env_info.agents))

# number of actions
action_size = brain.vector_action_space_size
print('Number of actions:', action_size)

# examine the state space 
state = env_info.vector_observations[0]
print('States look like:', state)
state_size = len(state)
print('States have length:', state_size)

def reset():
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    env_info = env.reset(train_mode=True)[brain_name]
    return env_info.vector_observations[0]

def step(action):
    env_info = env.step(action)[brain_name]        # send the action to the environment
    next_state = env_info.vector_observations[0]   # get the next state
    reward = env_info.rewards[0]                   # get the reward
    done = env_info.local_done[0] 
    return next_state, reward, done

def dqn(n_episodes=1000, max_t=1000):
    """Deep Q-Learning.
    
    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
    """
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    solved = False
    beta = 0.4
    for i_episode in range(1, n_episodes+1):
        beta = max(1, beta+0.001)
        state = reset()
        score = 0
        for t in range(max_t):
            action = agent.act(state)
            next_state, reward, done = step(action)
            agent.step(state, action, reward, next_state, done, beta)
            state = next_state
            score += reward
            if done:
                break 
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if not solved and np.mean(scores_window)>=13.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
            solved = True
    torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
    return scores

agent = Agent(state_size=37, action_size=4, seed=0)
scores = dqn()
fig = plt.figure(figsize=(16, 12))
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores)
plt.xlabel('Episode number')
plt.ylabel('Score')
plt.show()