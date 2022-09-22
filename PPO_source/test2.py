import os
import glob
import time
from datetime import datetime

import torch
import numpy as np

import gym
import TurtleBot_v0

env_id = "TurtleBot-v2"

env = gym.make(env_id)
obs = env.reset()
print("Obs shape: ",env.observation_space.shape)
print("obs ", obs.shape)
time.sleep(10000)