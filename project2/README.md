# Project : Continous control

## Description 
THis environment consists of 20 agents. Each agent is a double-jointed arm capable of moving around to reach different targets. The goal is for each agent to reach its hand towards a moving target (visually represented as a green dot) and stay on target for as long as possible.

The input state-space for the agents consists of a vector with 33 variables. This includes information such as position, rotation, velocity, and angular velocities of the arm.

Each agent is can act on the world by controlling the torque that is applied to each of the two joints. This is represented as an action-space that consists of a vector of 4 continuous valued variable. Each of these action variables is a value constrained between -1 and 1.

Each agent is awarded a reward of +0.1 for each timestep in which the hand is on the correct target. The environment is considered solved when the agents achieve an average score of +30 (averaged over all 20 agents and across 100 consecutive episodes).


## Files
- `run.py`: Main script used to control and train the agent for experimentation
- `model.py`: Q-network class used to map state to action values
- `prioritized_replay_buffer.py`: Implementation of priority replay buffer
- `segment_tree.py`: SegmentTree data structure implementation
- `es.py`: Evolution strategy for one agent
- `es_20.py`: Evolution strategy for 20 agents
- `model_20.py`: Evolution strategy model for 20 agents
- `model_es.py`: Evolution strategy model for one agent
- `cross_entropy.py`: Genetic algorithm try
- `report.pdf`: Technical report 

## Dependencies
To be able to run this code, you will need an environment with Python 3 and 
the dependencies are listed in the `requirements.txt` file so that you can install them
using the following command: 
```
pip install requirements.txt
``` 

Furthermore, you need to download the environment from one of the links below. You need only to select
the environment that matches your operating system:
- Linux : [link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
- MAC OSX : [link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
- Windows : [link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)

## Running
To run evolution strategies run:
python es.py or python es_20.py

To run DDPG run:
python run.py