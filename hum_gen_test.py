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
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks
  
min_y, max_y = 0, 15000

plot_name = "101010"
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


if __name__ == "__main__": 
    name = "ResultRobTest"
    model_path = "/home/name/Desktop/Codes/Data-38GEN/Xml_files/Humanoid_Gen38_Robot3.xml"
    rob_env = envs.get_environment(createIt(model_path, name)) 
    x_data = []
    y_data = []
    ydataerr = []
    times = [datetime.now()]

    train_fn = functools.partial(
    tRain, num_timesteps=70_000_000, num_evals=10, reward_scaling=0.1,
    episode_length=1000, normalize_observations=True, action_repeat=1,
    unroll_length=10, num_minibatches=32, num_updates_per_batch=8,
    discounting=0.97, learning_rate=3e-4, entropy_cost=1e-3, num_envs=2048,
    batch_size=1024, seed=0)
    make_inference_fn, Params, _,curr_brain = train_fn(environment=rob_env, progress_fn=progress)

    renderer = None
    eval_env = rob_env
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

    inference_fn = make_policy(Params)
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
    media.write_video(path = f"/home/name/Desktop/Videos/Running_Genetic/Gen{101010}Runner_Selected.mp4", images = images, fps=1.0 / env.dt / render_every)
