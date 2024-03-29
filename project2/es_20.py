import numpy as np
from unityagents import UnityEnvironment
from model_20 import ModelES
import json
from json import encoder
from collections import deque
import time
import keyboard
import termios, fcntl, sys, os
import random
import copy
encoder.FLOAT_REPR = lambda o: format(o, '.16f')

def compute_ranks(x):
  """
  Returns ranks in [0, len(x))
  Note: This is different from scipy.stats.rankdata, which returns ranks in [1, len(x)].
  (https://github.com/openai/evolution-strategies-starter/blob/master/es_distributed/es.py)
  """
  assert x.ndim == 1
  ranks = np.empty(len(x), dtype=int)
  ranks[x.argsort()] = np.arange(len(x))
  return ranks

def compute_centered_ranks(x):
  """
  https://github.com/openai/evolution-strategies-starter/blob/master/es_distributed/es.py
  """
  y = compute_ranks(x.ravel()).reshape(x.shape).astype(np.float32)
  y /= (x.size - 1)
  y -= .5
  return y

def compute_weight_decay(weight_decay, model_param_list):
  model_param_grid = np.array(model_param_list)
  return - weight_decay * np.mean(model_param_grid * model_param_grid, axis=1)

# adopted from:
# https://github.com/openai/evolution-strategies-starter/blob/master/es_distributed/optimizers.py

class Optimizer(object):
  def __init__(self, pi, epsilon=1e-08):
    self.pi = pi
    self.dim = pi.num_params
    self.epsilon = epsilon
    self.t = 0

  def update(self, globalg):
    self.t += 1
    step = self._compute_step(globalg)
    theta = self.pi.mu
    ratio = np.linalg.norm(step) / (np.linalg.norm(theta) + self.epsilon)
    self.pi.mu = theta + step
    return ratio

  def _compute_step(self, globalg):
    raise NotImplementedError


class BasicSGD(Optimizer):
  def __init__(self, pi, stepsize):
    Optimizer.__init__(self, pi)
    self.stepsize = stepsize

  def _compute_step(self, globalg):
    step = -self.stepsize * globalg
    return step

class SGD(Optimizer):
  def __init__(self, pi, stepsize, momentum=0.9):
    Optimizer.__init__(self, pi)
    self.v = np.zeros(self.dim, dtype=np.float32)
    self.stepsize, self.momentum = stepsize, momentum

  def _compute_step(self, globalg):
    self.v = self.momentum * self.v + (1. - self.momentum) * globalg
    step = -self.stepsize * self.v
    return step


class Adam(Optimizer):
  def __init__(self, pi, stepsize, beta1=0.99, beta2=0.999):
    Optimizer.__init__(self, pi)
    self.stepsize = stepsize
    self.beta1 = beta1
    self.beta2 = beta2
    self.m = np.zeros(self.dim, dtype=np.float32)
    self.v = np.zeros(self.dim, dtype=np.float32)

  def _compute_step(self, globalg):
    a = self.stepsize * np.sqrt(1 - self.beta2 ** self.t) / (1 - self.beta1 ** self.t)
    self.m = self.beta1 * self.m + (1 - self.beta1) * globalg
    self.v = self.beta2 * self.v + (1 - self.beta2) * (globalg * globalg)
    step = -a * self.m / (np.sqrt(self.v) + self.epsilon)
    return step

class SimpleGA:
  '''Simple Genetic Algorithm.'''
  def __init__(self, num_params,      # number of model parameters
               sigma_init=0.1,        # initial standard deviation
               sigma_decay=0.999,     # anneal standard deviation
               sigma_limit=0.01,      # stop annealing if less than this
               popsize=256,           # population size
               elite_ratio=0.1,       # percentage of the elites
               forget_best=False,     # forget the historical best elites
               weight_decay=0.01,     # weight decay coefficient
              ):

    self.num_params = num_params
    self.sigma_init = sigma_init
    self.sigma_decay = sigma_decay
    self.sigma_limit = sigma_limit
    self.popsize = popsize

    self.elite_ratio = elite_ratio
    self.elite_popsize = int(self.popsize * self.elite_ratio)

    self.sigma = self.sigma_init
    self.elite_params = np.zeros((self.elite_popsize, self.num_params))
    self.elite_rewards = np.zeros(self.elite_popsize)
    self.best_param = np.zeros(self.num_params)
    self.best_reward = 0
    self.first_iteration = True
    self.forget_best = forget_best
    self.weight_decay = weight_decay

  def rms_stdev(self):
    return self.sigma # same sigma for all parameters.

  def ask(self):
    '''returns a list of parameters'''
    self.epsilon = np.random.randn(self.popsize, self.num_params) * self.sigma
    solutions = []
    
    def mate(a, b):
      c = np.copy(a)
      idx = np.where(np.random.rand((c.size)) > 0.5)
      c[idx] = b[idx]
      return c
    
    elite_range = range(self.elite_popsize)
    for i in range(self.popsize):
      idx_a = np.random.choice(elite_range)
      idx_b = np.random.choice(elite_range)
      child_params = mate(self.elite_params[idx_a], self.elite_params[idx_b])
      solutions.append(child_params + self.epsilon[i])

    solutions = np.array(solutions)
    self.solutions = solutions

    return solutions

  def tell(self, reward_table_result):
    # input must be a numpy float array
    assert(len(reward_table_result) == self.popsize), "Inconsistent reward_table size reported."

    reward_table = np.array(reward_table_result)
    
    if self.weight_decay > 0:
      l2_decay = compute_weight_decay(self.weight_decay, self.solutions)
      reward_table += l2_decay

    if self.forget_best or self.first_iteration:
      reward = reward_table
      solution = self.solutions
    else:
      reward = np.concatenate([reward_table, self.elite_rewards])
      solution = np.concatenate([self.solutions, self.elite_params])

    idx = np.argsort(reward)[::-1][0:self.elite_popsize]

    self.elite_rewards = reward[idx]
    self.elite_params = solution[idx]

    self.curr_best_reward = self.elite_rewards[0]
    
    if self.first_iteration or (self.curr_best_reward > self.best_reward):
      self.first_iteration = False
      self.best_reward = self.elite_rewards[0]
      self.best_param = np.copy(self.elite_params[0])

    if (self.sigma > self.sigma_limit):
      self.sigma *= self.sigma_decay

  def current_param(self):
    return self.elite_params[0]

  def set_mu(self, mu):
    pass

  def best_param(self):
    return self.best_param

  def result(self): # return best params so far, along with historically best reward, curr reward, sigma
    return (self.best_param, self.best_reward, self.curr_best_reward, self.sigma)

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state

class OpenES:
  ''' Basic Version of OpenAI Evolution Strategies.'''
  def __init__(self, num_params,             # number of model parameters
               sigma_init=0.1,               # initial standard deviation
               sigma_decay=1.0,            # anneal standard deviation
               sigma_limit=0.1,             # stop annealing if less than this
               learning_rate=0.03,           # learning rate for standard deviation
               learning_rate_decay = 1.0, # annealing the learning rate
               learning_rate_limit = 0.001,  # stop annealing learning rate
               popsize=256,                  # population size
               antithetic=True,             # whether to use antithetic sampling
               weight_decay=0.00,            # weight decay coefficient
               rank_fitness=False,            # use rank rather than fitness numbers
               forget_best=False):            # forget historical best

    self.num_params = num_params
    self.sigma_decay = sigma_decay
    self.sigma = sigma_init
    self.sigma_init = sigma_init
    self.sigma_limit = sigma_limit
    self.learning_rate = learning_rate
    self.learning_rate_decay = learning_rate_decay
    self.learning_rate_limit = learning_rate_limit
    self.popsize = popsize
    self.antithetic = antithetic
    if self.antithetic:
      assert (self.popsize % 2 == 0), "Population size must be even"
      self.half_popsize = int(self.popsize / 2)

    self.reward = np.zeros(self.popsize)
    self.mu = np.zeros(self.num_params)
    self.best_mu = np.zeros(self.num_params)
    self.best_reward = 0
    self.first_interation = True
    self.forget_best = forget_best
    self.weight_decay = weight_decay
    self.rank_fitness = rank_fitness
    if self.rank_fitness:
      self.forget_best = True # always forget the best one if we rank
    # choose optimizer
    self.optimizer = Adam(self, learning_rate)

  def rms_stdev(self):
    sigma = self.sigma
    return np.mean(np.sqrt(sigma*sigma))

  def ask(self):
    '''returns a list of parameters'''
    # antithetic sampling
    if self.antithetic:
      self.epsilon_half = np.random.randn(self.half_popsize, self.num_params)
      self.epsilon = np.concatenate([self.epsilon_half, - self.epsilon_half])
    else:
      self.epsilon = np.random.randn(self.popsize, self.num_params)

    self.solutions = self.mu.reshape(1, self.num_params) + self.epsilon * self.sigma

    return self.solutions

  def tell(self, reward_table_result):
    # input must be a numpy float array
    assert(len(reward_table_result) == self.popsize), "Inconsistent reward_table size reported."
    
    reward = np.array(reward_table_result)
    
    if self.rank_fitness:
      reward = compute_centered_ranks(reward)
    
    if self.weight_decay > 0:
      l2_decay = compute_weight_decay(self.weight_decay, self.solutions)
      reward += l2_decay

    idx = np.argsort(reward)[::-1]

    best_reward = reward[idx[0]]
    best_mu = self.solutions[idx[0]]

    self.curr_best_reward = best_reward
    self.curr_best_mu = best_mu

    if self.first_interation:
      self.first_interation = False
      self.best_reward = self.curr_best_reward
      self.best_mu = best_mu
    else:
      if self.forget_best or (self.curr_best_reward > self.best_reward):
        self.best_mu = best_mu
        self.best_reward = self.curr_best_reward

    # main bit:
    # standardize the rewards to have a gaussian distribution
    normalized_reward = (reward - np.mean(reward)) / np.std(reward)
    change_mu = 1./(self.popsize*self.sigma)*np.dot(self.epsilon.T, normalized_reward)
    
    # self.mu += self.learning_rate * change_mu

    self.optimizer.stepsize = self.learning_rate
    update_ratio = self.optimizer.update(-change_mu)

    # adjust sigma according to the adaptive sigma calculation
    if (self.sigma > self.sigma_limit):
      self.sigma *= self.sigma_decay

    if (self.learning_rate > self.learning_rate_limit):
      self.learning_rate *= self.learning_rate_decay

  def current_param(self):
    return self.curr_best_mu

  def set_mu(self, mu):
    self.mu = np.array(mu)

  def set_sigma(self, sigma):
    self.sigma = sigma

  def set_pop(self, pop):
    self.popsize = pop

  def set_lr(self, lr):
    self.learning_rate = lr

  def best_param(self):
    return self.best_mu

  def result(self): # return best params so far, along with historically best reward, curr reward, sigma
    return (self.best_mu, self.best_reward, self.curr_best_reward, self.sigma)

env = UnityEnvironment(file_name='./Reacher-2.app',no_graphics=True)
brain_name = env.brain_names[0]
brain = env.brains[brain_name]
env_info = env.reset(train_mode=True)[brain_name]

num_agents = len(env_info.agents)

print('Number of agents:', num_agents)
action_size = brain.vector_action_space_size
state_size = env_info.vector_observations.shape[1]
print(action_size,state_size)
noise = OUNoise(action_size, 0)
model = ModelES(state_size, action_size)
# model.load_model('best_one_smallstep')


NPARAMS = model.param_count        # make this a 100-dimensinal problem.
NPOPULATION = 200    # use population size of 101.
MAX_ITERATION = 4000 # run each solver for 5000 generations.
EPS = 0.0001


def normalize(state):
  return (state + 10)/20

def reset():
    env_info = env.reset(train_mode=True)[brain_name]
    return env_info.vector_observations

def step(action):
    env_info = env.step(action)[brain_name]        # send the action to the environment
    next_state = env_info.vector_observations   # get the next state
    reward = env_info.rewards                  # get the reward
    done = env_info.local_done
    return next_state, reward, done

def fit_func(solution_list, num_steps, GREEDY):
    episode_return = 0.0
    scores_t = np.zeros(20) 
    states = reset()
    noise.reset()
    
    for t in range(num_steps):
        actions = []
        for l in range(20):
          model.set_model_params(solution_list[l])
          normalized_state = normalize(states[l])
          actions.append(np.clip(model.get_action(normalized_state), -1, 1))
        states, rewards, dones = step(actions)
        scores_t += rewards
        # episode_return += reward
        if np.any(dones) and t < 1000:
            print("Finished before max step")

    return scores_t+EPS

def test_solver(solver):
    history = []
    changed = False
    scores_window = deque(maxlen=100)
    best_window = deque(maxlen=100)
    num_steps = 50
    GREEDY = 1
    GREEDY_DECAY = 0.99
    for j in range(MAX_ITERATION):
      # if keyboard.is_pressed('s'):
      #   print("Put value for sigma")
      #   solver.set_sigma(float(sys.stdin.readline().strip()))
      # if keyboard.is_pressed('l'):
      #   print("Put value for lr")
      #   solver.set_lr(float(sys.stdin.readline().strip()))
      # if keyboard.is_pressed('p'):
      #   print("Put value for population")
      #   solver.set_pop(int(sys.stdin.readline().strip()))
      GREEDY = GREEDY*GREEDY_DECAY
      solutions = solver.ask()
      fitness_list = []
      for i in range(10):
        solution_list = []
        for k in range(i*20,i*20+20):
          solution_list.append(solutions[k])
          # model.set_model_params(solutions[i])
        fitness_list.extend(fit_func(solution_list, num_steps, GREEDY))
        print('\rPopulation element {}\t'.format(i), end='\r')
      solver.tell(fitness_list)
      result = solver.result() # first element is the best solution, second element is the best fitness
      history.append(result[2])
      scores_window.append(result[2])
      best_window.append(result[1])
      print(j, "Best candidate", result[1], "Max in generation", result[2], "Total Average", np.mean(scores_window), "Best average", np.mean(best_window))
      # if (not changed and (np.mean(scores_window) > 1.5 or j == 1500)):
      #   changed = True
      #   num_steps = 1001
        # solver.set_lr(0.01)
        # solver.set_pop(20)
        # with open("best_one_smallstep", 'wt') as out:
        #   res = json.dump([np.array(result[0]).round(8).tolist()], out, sort_keys=True, indent=0, separators=(',', ': '))
      if (np.mean(scores_window)>30):
        print("Solved")
        print(history)
        with open("best_one_bigstep", 'wt') as out:
          res = json.dump([np.array(result[0]).round(8).tolist()], out, sort_keys=True, indent=0, separators=(',', ': '))
      # if j % 500 == 0:
        # with open("best_one", 'wt') as out:
        #   res = json.dump([np.array(result[0]).round(8).tolist()], out, sort_keys=True, indent=0, separators=(',', ': '))
        # model.set_model_params(result[0])
        # test(True)
      
    return history

oes = OpenES(NPARAMS,                  # number of model parameters
            popsize=NPOPULATION       # population size
            )
# cmaes = CMAES(NPARAMS,
#               popsize=NPOPULATION,
#               sigma_init = 0.1
#           )
oes_history = test_solver(oes)
print(oes_history)
# print(test())
