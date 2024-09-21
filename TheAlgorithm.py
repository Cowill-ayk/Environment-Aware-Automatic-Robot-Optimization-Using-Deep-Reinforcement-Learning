#!/usr/bin/env python
# coding: utf-8

# In[1]:
from base_env import MjxEnv
#@title Import MuJoCo, MJX, and Brax

from datetime import datetime
import functools
import jax
from jax import numpy as jp
import numpy as np
from typing import Any, Dict, Tuple, Union

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
import gc

#@title Install

#@title Import packages for plotting and creating graphics
import time
import itertools
import numpy as np
from typing import Callable, NamedTuple, Optional, Union, List

# Graphics and plotting.
import mediapy as media
import matplotlib.pyplot as plt

# More legible printing from numpy.
np.set_printoptions(precision=3, suppress=True, linewidth=100)


#@title Import MuJoCo, MJX, and Brax

import os

# In[ ]:

# In[3]:


import functools
from datetime import datetime

#@title Create Environment With Scaling Variables
import sys

import numpy as np
from typing import Any, Dict, Tuple, Union

import multiprocessing as mp
from multiprocessing import Pool, Process


epsilon = 0.1
init_training_state = None
num_scaling_var = 17 # Must be tuned according to the base model
init_scaling_vars = np.zeros(num_scaling_var)
N = 5
path_to_base_env = "/home/name/Desktop/Codes/ScalingOptimization (1)"
num_iterations_to_go = 500
mutation_limits_wstrt = {
    "max" : [epsilon for i in range(num_scaling_var)],
    "min" : [-epsilon for i in range(num_scaling_var)]
}
mutation_limits_explr = {
    "max" : [2*epsilon for i in range(num_scaling_var)],
    "min" : [-2*epsilon for i in range(num_scaling_var)]
}
num_evals = 5
eelim_factor = 0.21 # Early Elimination Limiting Factor


# In[27]:


#@title Humanoid
os.environ['MUJOCO_GL'] = 'glx'
from HumanoidEnv import createIt
from brax.io import model
from brax import envs
from jax import numpy as jp
rew_lower_limit = lambda r,t: r*jp.exp(eelim_factor*(t-num_evals+1))

plot_name = ""
"""
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
"""


class Robot :
  def __init__(self, scaling_var, model_path, brain): 
    self.scaling_var = scaling_var
    self.model_path = model_path
    self.brain = brain

# In[ ]:
plot_name =  ""

#@title Genetic Algorithm (Early Elimination  & Discretization)
#name = "init"
#path = "/home/name/Desktop/Codes/Data/Xml_files/Humanoid_Gen7_Robot0.xml"

continue_from = 0

#returns Robot, y_data, Params
def robot_process (ancestor_robot, rew_y_data, gen_i, rob_i): 

  os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'default'
  os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'true'
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

  
  name=f"Humanoid_Gen{gen_i}_Robot{rob_i}"
  plot_name = name
  sc_var_change = np.subtract(np.random.randint(5,size=num_scaling_var), np.array([2 for i in range(num_scaling_var)]))
  new_scv = np.add(ancestor_robot.scaling_var,sc_var_change*0.03)
  model_path = humanoid_genesis(new_scv, name)
  
  rob_env = envs.get_environment(createIt(model_path, name)) 
  x_data = []
  y_data = []
  ydataerr = []
  times = [datetime.now()]
  
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
  train_fn = functools.partial(
    tRain, num_timesteps=10_000_000, num_evals=num_evals, reward_scaling=0.1,
    episode_length=1000, normalize_observations=True, action_repeat=1,
    unroll_length=10, num_minibatches=32, num_updates_per_batch=8,
    discounting=0.97, learning_rate=3e-4, entropy_cost=1e-3, num_envs=2048,
    batch_size=1024, seed=0)
  make_inference_fn, Params, _,curr_brain = train_fn(environment=rob_env, progress_fn=progress, transfer_params=ancestor_robot.brain, cmprsn_rewards=rew_y_data)
  curr_Robo = Robot(new_scv, model_path, curr_brain)
  print(f'time to jit: {times[1] - times[0]}')
  print(f'time to train: {times[-1] - times[1]}')
  # if make_inference_fn == None:
  #   return (None,None,[0 for i in range(num_evals)])
  return (curr_Robo, y_data, Params)


def recordVideo(prms, scalingVars, gen_i): 
  from env_model_generator import humanoid_genesis
  from train import tRain
  from HumanoidEnv import createIt
  from base_env import MjxEnv
  from brax import envs
  from brax.envs.base import Env, State
  from brax.training.agents.ppo import networks as ppo_networks
  from brax.training.acme import running_statistics

  renderer = None
  eval_env = envs.get_environment(createIt(humanoid_genesis(scalingVars, "evalEnv"),  "evalEnv")) 
  env = envs.training.wrap(
      eval_env,
      episode_length=1000,
      action_repeat=1,
      randomization_fn=None,
  )

  renderer = mujoco.Renderer(eval_env.model, width = 1920, height = 1080 )
  def get_image(state: State, camera: str) -> np.ndarray:
    """Renders the environment state."""
    d = mujoco.MjData(eval_env.model)
    # write the mjx.Data into an mjData object
    mjx.device_get_into(d, state.pipeline_state)
    mujoco.mj_forward(eval_env.model, d)
    # use the mjData object to update the renderer
    renderer.update_scene(d, camera=camera)
    return renderer.render()

  jit_reset = jax.jit(eval_env.reset)
  jit_step = jax.jit(eval_env.step)
  rng = jax.random.PRNGKey(0)
  state = jit_reset(rng)
  rollout = [state]
  images = [get_image(state, camera='side')]
  
  ppo_network = ppo_networks.make_ppo_networks(
      state.obs.shape[-1],
      env.action_size,
      preprocess_observations_fn=running_statistics.normalize)
  make_policy = ppo_networks.make_inference_fn(ppo_network)

  inference_fn = make_policy(prms)
  jit_inference_fn = jax.jit(inference_fn)
  n_steps = 1000
  render_every = 2
  

  for k in range(n_steps):
    act_rng, rng = jax.random.split(rng)
    ctrl, _ = jit_inference_fn(state.obs, act_rng)
    state = jit_step(state, ctrl)
    rollout.append(state)
    if k % render_every == 0:
      images.append(get_image(state, camera='side'))

    if state.done:
      break
  media.write_video(path = f"/home/name/Desktop/Videos/Running_Genetic/Gen{gen_i}Runner_Selected.mp4", images = images, fps=1.0 / env.dt / render_every)
min_y, max_y = 0, 15000

if __name__ == '__main__':
  mp.set_start_method('spawn')
  path = "/home/name/Desktop/Codes/Data/Xml_files/init.xml"
  init_robot = Robot(np.array([0 for i in range(num_scaling_var)  ]),
                    path,
                    None
                    )
  glob_y_data, robs, Y, Params = [0],None,None,None
  rews,index = None, None
  selectedRob = init_robot
  selectedParams = None 
  f, pol_name, vidP = None,None,None
  Robs,Y,Params = None,None,None
  rob,y,params = None,None,None 
  for i in range(continue_from,num_iterations_to_go): 
    del Robs
    del Y
    del Params
    Robs =[selectedRob]
    Y=[glob_y_data]
    Params=[selectedParams]
    del glob_y_data 
    glob_y_data = None
    
    
    for k in range(N-1): 
      del rob
      del y 
      del params
      with Pool(1) as p: 
        [(rob,y,params)] = p.starmap(robot_process, [
          (selectedRob, glob_y_data, i, k),
        ])
        if glob_y_data == None : 
          glob_y_data = y
        else : 
          if glob_y_data[-1] < y[-1]: 
            glob_y_data = y
        
        p.close()
        Robs.append(rob)
        Y.append(y)
        Params.append(params)
    del rews
    del index
    del glob_y_data
    del selectedRob
    del selectedParams 
    rews = np.array([y[-1] for y in Y])
    index = np.argmax(rews)
    glob_y_data = Y[index]
    selectedRob =Robs[index]
    selectedParams = Params[index]
    
    del pol_name
    pol_name = f"policy-Gen{i}"
    model_path = f'/home/name/Desktop/Codes/ScalingOptimization (1)/policies/{pol_name}'
    model.save_params(model_path, selectedParams)
    del f 
    f = open("/home/name/Desktop/Codes/ScalingOptimization (1)/logofselecteds.txt", "a")
    f.write(f"{i}. Generation Model:{model_path}\n{i}. Generation Scaling Variables: {selectedRob.scaling_var}\n")
    f.close()
    del vidP
    vidP = Process(target = recordVideo, args=(selectedParams, selectedRob.scaling_var, i,))
    vidP.start()
    vidP.join()
    gc.collect()



    


"""
Candidates = [init_robot for i in range(N)]
FitnessScores = [0 for i in range(N)]
FitnessScores = np.array(FitnessScores)
Params = [0 for i in range(N)]
selected = init_robot
Envs = [0 for i in range(N)]

for i in range(continue_from, num_iterations_to_go):
  
  cmprsn_rewards = None
  print(i)
  #Sampling
  for j in range(2): # ADJUST THE HEALTHY Z RANGE, OBTAINED FROM GENESIS
      sc_var_change = np.subtract(np.random.randint(5,size=num_scaling_var), np.array([2 for i in range(num_scaling_var)]))
      new_scv = np.add(selected.scaling_var,sc_var_change*0.1)
      name=f"Humanoid_Gen{i}_Robot{j}"
      Candidates[j] = Robot(new_scv, humanoid_genesis(new_scv, name), selected.brain)
  for j in range(2):
      sc_var_change = np.subtract(np.random.randint(5,size=num_scaling_var), np.array([2 for i in range(num_scaling_var)]))
      new_scv = np.add(selected.scaling_var,sc_var_change*0.1)
      name=f"Humanoid_Gen{i}_Robot{j+2}"
      Candidates[j+2] = Robot(new_scv, humanoid_genesis(new_scv, name), selected.brain)

  #Fitness scores(perhaps can be taken from y_datas)
  for k in range(N-1)
    name=f"Humanoid_Gen{i}_Robot{k}"
    plot_name = name
    rob_env = envs.get_environment (createIt(Candidates[k].model_path, name))
    Envs[k] = rob_env
    
    x_data = []
    y_data = []
    ydataerr = []
    times = [datetime.now()]
    print(f"\n--------\n\nIteration{i}-Robot{k}\n\n--------\n")
    a,b,c,d= train_fn(environment=rob_env, progress_fn=progress, transfer_params=Candidates[k].brain, cmprsn_rewards=cmprsn_rewards)
    #make_inference_fn, Params[k], _,rob_new_training_state = train_fn(environment=rob_env, progress_fn=progress)
    if a == None:
        continue
    make_inference_fn, Params[k], _,rob_new_training_state = a,b,c,d
    del a,b,c,d
    print(f'time to jit: {times[1] - times[0]}')
    print(f'time to train: {times[-1] - times[1]}')
    FitnessScores[k] = y_data[-1]
    new_rob = Robot(Candidates[k].scaling_var, Candidates[k].model_path, rob_new_training_state)
    Candidates[k] = new_rob
    if cmprsn_rewards == None: 
      cmprsn_rewards = y_data
    elif cmprsn_rewards[-1]<y_data[-1] :
      cmprsn_rewards = y_data
    gc.collect()
        
  #Selection and Recording power limits established in your graphics card’s VBIOS, and you also need to re-enable the settings after a Windows reboot. However, it does offer you more granular options than Afterburner, pa
  index = np.argmax(FitnessScores)
  selected = Candidates[index]
  selected_params = Params[index]
  name = f"policy-Gen{i}"
  model_path = f'/home/name/Desktop/Codes/ScalingOptimization (1)/policies/{name}'
  model.save_params(model_path, selected_params)
  f = open("/home/name/Desktop/Codes/ScalingOptimization (1)/logofselecteds.txt", "a")
  f.write(f"{i}. Generation Model:{model_path}\n{i}. Generation Scaling Variables: {selected.scaling_var}\n")
  f.close()

  inference_fn = make_inference_fn(selected_params)
  jit_inference_fn = jax.jit(inference_fn)
  eval_env = Envs[index]
  jit_reset = jax.jit(eval_env.reset)
  jit_step = jax.jit(eval_env.step)
  # initialize the state
  renderer = None
  renderer = mujoco.Renderer(eval_env.model, width = 1920, height = 1080 )
  def get_image(state: State, camera: str) -> np.ndarray:
    #Renders the environment state.
    d = mujoco.MjData(eval_env.model)
    # write the mjx.Data into an mjData object
    mjx.device_get_into(d, state.pipeline_state)
    mujoco.mj_forward(eval_env.model, d)
    # use the mjData object to update the renderer
    renderer.update_scene(d, camera=camera)
    return renderer.render()
  rng = jax.random.PRNGKey(0)
  state = jit_reset(rng)
  rollout = [state]
  images = [get_image(state, camera='side')]

  n_steps = 5000
  render_every = 2

  for k in range(n_steps):
    act_rng, rng = jax.random.split(rng)
    ctrl, _ = jit_inference_fn(state.obs, act_rng)
    state = jit_step(state, ctrl)
    rollout.append(state)
    if k % render_every == 0:
      images.append(get_image(state, camera='side'))

    if state.done:
      break

  media.write_video(path = f"/home/name/Desktop/Videos/Running_Genetic/Gen{i}Runner_Selected.mp4", images = images, fps=1.0 / eval_env.dt / render_every)
  del images
  Candidates[4] = selected
  FitnessScores[4] = FitnessScores[index]
  Params[4] = selected_params
  Envs[4] = eval_env
  gc.collect()

"""