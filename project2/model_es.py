import numpy as np
import random
# I implemented Schmidhuber's "Compressed Network Search" but didn't use it.
# ndded for the compress/decompress functions.
#from scipy.fftpack import dct
import json
import sys
import time

from nn import sigmoid, relu, passthru, softmax, sample, RNNModel

class ModelES:
  ''' simple feedforward model '''
  def __init__(self):
    self.layer_1 = 1024
    self.layer_2 = 512
    self.layer_3 = 256
    self.input_size = 33
    self.output_size = 4
    if self.layer_2 > 0:
      self.shapes = [ (self.input_size, self.layer_1),
                      (self.layer_1, self.layer_2),
                      (self.layer_2, self.layer_3),
                      (self.layer_3, self.output_size)]
    elif self.layer_2 == 0:
      self.shapes = [ (self.input_size + self.time_input, self.layer_1),
                      (self.layer_1, self.output_size)]
    else:
      assert False, "invalid layer_2"

    self.sample_output = False
    self.activations = [relu, relu, relu, np.tanh]

    self.weight = []
    self.bias = []
    self.param_count = 0

    idx = 0
    for shape in self.shapes:
      self.weight.append(np.zeros(shape=shape))
      self.bias.append(np.zeros(shape=shape[1]))
      self.param_count += (np.product(shape) + shape[1])
      idx += 1

  def get_action(self, x, mean_mode=False):
    # if mean_mode = True, ignore sampling.
    h = np.array(x).flatten()
  
    num_layers = len(self.weight)
    for i in range(num_layers):
      w = self.weight[i]
      b = self.bias[i]
      h = np.matmul(h, w) + b
      h = self.activations[i](h)

    return h

  def set_model_params(self, model_params):
    pointer = 0
    for i in range(len(self.shapes)):
      w_shape = self.shapes[i]
      b_shape = self.shapes[i][1]
      s_w = np.product(w_shape)
      s = s_w + b_shape
      chunk = np.array(model_params[pointer:pointer+s])
      self.weight[i] = chunk[:s_w].reshape(w_shape)
      self.bias[i] = chunk[s_w:].reshape(b_shape)
      pointer += s

  def load_model(self, filename):
    with open(filename) as f:
      data = json.load(f)
    self.data = data
    model_params = np.array(data[0]) # assuming other stuff is in data
    self.set_model_params(model_params)
