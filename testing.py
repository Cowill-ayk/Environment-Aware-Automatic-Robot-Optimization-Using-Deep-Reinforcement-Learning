import gc
from datetime import datetime
import functools
import jax
from jax import numpy as jp
import numpy as np
from typing import Any, Dict, Tuple, Union
from env_model_generator import humanoid_genesis
from train import tRain
from HumanoidEnv import createIt
from base_env import MjxEnv
from jax import numpy as jp
from datetime import datetime
import functools
import jax
from brax import envs
from brax import math
from brax.base import Base, Motion, Transform
from brax.envs.base import Env, State
from brax.training.agents.ppo import train as ppo
from brax.training.agents.ppo import networks as ppo_networks
from brax.io import model
from etils import epath
from flax import struct
from matplotlib import pyplot as plt
import mediapy as media
from ml_collections import config_dict
import mujoco
from mujoco import mjx
import os 
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'default'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'true'
num_evals = 10
plot_name = "38_3_test"

min_y, max_y = 0, 3000
def progress(num_steps, metrics, cmprsn_rewards = None):
    #Array data update
    times.append(datetime.now())
    x_data.append(num_steps)
    y_data.append(metrics['eval/episode_reward'])
    ydataerr.append(metrics['eval/episode_reward_std'])
    #Plot drawing
    plt.xlim([0, train_fn.keywords['num_timesteps'] * 1.25])
    plt.ylim([min_y, max_y])

    plt.xlabel('# environment steps')
    plt.ylabel('reward per episode')
    plt.title(f'y={y_data[-1]:.3f}')

    plt.errorbar(
        x_data, y_data, yerr=ydataerr)
    
    plt.savefig(fname = "plots/"+plot_name+".png", format = "png")
    #Early elimination calculation
    if cmprsn_rewards == None : 
      return False
    else : 
      t = len(y_data) - 1
      r_t = cmprsn_rewards[t]
      if y_data[-1] < rew_lower_limit(r_t,t): 
        return True
      else : 
        return False
rob_env = envs.get_environment(createIt("/home/name/Desktop/Codes/Data/Xml_files/Humanoid_Gen38_Robot3.xml", "38_3")) 
x_data = []
y_data = []
ydataerr = []
times = [datetime.now()]
train_fn = functools.partial(
    tRain, num_timesteps=30_000_000, num_evals=num_evals, reward_scaling=0.1,
    episode_length=1000, normalize_observations=True, action_repeat=1,
    unroll_length=10, num_minibatches=32, num_updates_per_batch=8,
    discounting=0.97, learning_rate=3e-4, entropy_cost=1e-3, num_envs=2048,
    batch_size=1024, seed=0)
make_inference_fn, Params, _,curr_brain = train_fn(environment=rob_env, progress_fn=progress)
print(f'time to jit: {times[1] - times[0]}')
print(f'time to train: {times[-1] - times[1]}')

plot_name = "InitRob_test"
del rob_env
rob_env = envs.get_environment(createIt("/home/name/Desktop/Codes/Data/Xml_files/init.xml", "init")) 
x_data = []
y_data = []
ydataerr = []
times = [datetime.now()]

make_inference_fn, Params, _,curr_brain = train_fn(environment=rob_env, progress_fn=progress)
print(f'time to jit: {times[1] - times[0]}')
print(f'time to train: {times[-1] - times[1]}')