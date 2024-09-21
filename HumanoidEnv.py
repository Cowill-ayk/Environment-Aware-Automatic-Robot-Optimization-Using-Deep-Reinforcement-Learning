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

#@title Humanoid Env
def createIt(path, name, z_range=(0.5,3.0)) : 
  class Humanoid(MjxEnv):

    def __init__(
        self,
        forward_reward_weight=1.25,
        ctrl_cost_weight=0.1,
        healthy_reward=5.0,
        terminate_when_unhealthy=True,
        healthy_z_range=z_range,
        reset_noise_scale=1e-2,
        exclude_current_positions_from_observation=True,
        **kwargs,
    ):
      mj_model = mujoco.MjModel.from_xml_path(path)
      mj_model.opt.solver = mujoco.mjtSolver.mjSOL_CG
      mj_model.opt.iterations = 6
      mj_model.opt.ls_iterations = 6

      physics_steps_per_control_step = 5
      kwargs['physics_steps_per_control_step'] = kwargs.get(
          'physics_steps_per_control_step', physics_steps_per_control_step)

      super().__init__(mj_model=mj_model, **kwargs)

      self._forward_reward_weight = forward_reward_weight
      self._ctrl_cost_weight = ctrl_cost_weight
      self._healthy_reward = healthy_reward
      self._terminate_when_unhealthy = terminate_when_unhealthy
      self._healthy_z_range = healthy_z_range
      self._reset_noise_scale = reset_noise_scale
      self._exclude_current_positions_from_observation = (
          exclude_current_positions_from_observation
      )

    def reset(self, rng: jp.ndarray) -> State:
      """Resets the environment to an initial state."""
      rng, rng1, rng2 = jax.random.split(rng, 3)

      low, hi = -self._reset_noise_scale, self._reset_noise_scale
      qpos = self.sys.qpos0 + jax.random.uniform(
          rng1, (self.sys.nq,), minval=low, maxval=hi
      )
      qvel = jax.random.uniform(
          rng2, (self.sys.nv,), minval=low, maxval=hi
      )

      data = self.pipeline_init(qpos, qvel)

      obs = self._get_obs(data, jp.zeros(self.sys.nu))
      reward, done, zero = jp.zeros(3)
      metrics = {
          'forward_reward': zero,
          'reward_linvel': zero,
          'reward_quadctrl': zero,
          'reward_alive': zero,
          'x_position': zero,
          'y_position': zero,
          'distance_from_origin': zero,
          'x_velocity': zero,
          'y_velocity': zero,
      }
      return State(data, obs, reward, done, metrics)

    def step(self, state: State, action: jp.ndarray) -> State:
      """Runs one timestep of the environment's dynamics."""
      data0 = state.pipeline_state
      data = self.pipeline_step(data0, action)

      com_before = data0.subtree_com[1]
      com_after = data.subtree_com[1]
      velocity = (com_after - com_before) / self.dt
      forward_reward = self._forward_reward_weight * velocity[0]

      min_z, max_z = self._healthy_z_range
      is_healthy = jp.where(data.qpos[2] < min_z, x=0.0, y=1.0)
      is_healthy = jp.where(
          data.qpos[2] > max_z, x=0.0, y=is_healthy
      )
      if self._terminate_when_unhealthy:
        healthy_reward = self._healthy_reward
      else:
        healthy_reward = self._healthy_reward * is_healthy

      ctrl_cost = self._ctrl_cost_weight * jp.sum(jp.square(action))

      obs = self._get_obs(data, action)
      reward = forward_reward + healthy_reward - ctrl_cost
      done = 1.0 - is_healthy if self._terminate_when_unhealthy else 0.0
      state.metrics.update(
          forward_reward=forward_reward,
          reward_linvel=forward_reward,
          reward_quadctrl=-ctrl_cost,
          reward_alive=healthy_reward,
          x_position=com_after[0],
          y_position=com_after[1],
          distance_from_origin=jp.linalg.norm(com_after),
          x_velocity=velocity[0],
          y_velocity=velocity[1],
      )

      return state.replace(
          pipeline_state=data, obs=obs, reward=reward, done=done
      )

    def _get_obs(
        self, data: mjx.Data, action: jp.ndarray
    ) -> jp.ndarray:
      """Observes humanoid body position, velocities, and angles."""
      position = data.qpos
      if self._exclude_current_positions_from_observation:
        position = position[2:]

      # external_contact_forces are excluded
      return jp.concatenate([
          position,
          data.qvel,
          data.cinert[1:].ravel(),
          data.cvel[1:].ravel(),
          data.qfrc_actuator,
      ])


  envs.register_environment(name, Humanoid)
  return name