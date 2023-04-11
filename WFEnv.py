# TODO
#  Test Single-Agent Environment
#  Run this for stochastic wind fields with PPO
#  Test Multi-Agent Environment
#  Run with MultAgen algorithm
#  Run with baseline
#  Integrate platform location actuation

# from ray.rllib.algorithms.qmix import QMix, QMixConfig
# from ray.rllib.env.env_context import EnvContext
# from ray import tune
import numpy as np
from floridyn import tools as wfct
from floridyn.utilities import cosd
import os
import random
import gymnasium as gym
from gymnasium.spaces import Dict, MultiDiscrete, MultiBinary, Box, Discrete, Tuple
from gymnasium import Env
from gymnasium.envs.registration import EnvSpec
from collections import deque

# TODO consider preview of reference power
# TODO QUESTION to include power in observations if it is part of the reward or not - is it possible to 'overflow' observations unecessarily

# from ray.rllib.env.multi_agent_env import (
#     MultiAgentEnv,
#     ENV_STATE,
#     MultiAgentDict,
#     MultiEnvDict,
# )
# from collections import OrderedDict

DT = 1.0  # discrete-time step for wind farm control
EPISODE_LEN = int(10 * 60 // DT)  # 10 minute episode length
WIND_SPEED_RANGE = (8, 16)
WIND_DIR_RANGE = (250, 290)
# RHO = 1

AX_IND_FACTORS = np.linspace(0.2, 0.4, 11, endpoint=True)
EPS = 0.0 # substitute for zero axial-ind factor
YAW_CHANGES = np.array([-1, 0, 1]) # np.array([-20, -15, -10, -5, 0, 5, 10, 15, 20])
YAW_RATE = 0.5 # degrees per second
DELTA_YAW = DT * YAW_RATE
PTFM_RNG = 200  # 200 of platform relocation range

YAW_ACTUATION = True
AX_IND_ACTUATION = False
PTFM_ACTUATION = False

MAX_YAW_TRAVEL_THR = 100 # 100 degrees
MAX_YAW_TRAVEL_TIME = 600 # 10 minutes

WORKING_DIR = os.getcwd()
ENV_CONFIG = {  # EnvSpec("wf_env", max_episode_steps=int(24*3600//DT), kwargs={
		"floris_input_file": os.path.join(WORKING_DIR, "9turb_floris_input.json"),
		"turbine_layout_std": 1,
		"offline_probability": 0.1, # probability of any given turbine going offline at each time-step
		"wind_change_probability": 0.1, # probability of wind speed/direction changing (1/2 for increase, 1/2 for decrease)
		"wind_speed_var": 0.5, # step change in m/s of wind speed
		"wind_dir_var": 5, # step change in degrees of wind direction,
		"wind_speed_turb_std": 0,  # 0.5, # standard deviation of normal turbulence of wind speed, set to 0 for no turbulence
		"wind_dir_turb_std": 0,  # 5, # standard deviation of normal turbulence  of wind direction, set to 0 for no turbulence
		"max_yaw_travel_thr": MAX_YAW_TRAVEL_THR,
		"max_yaw_travel_time": MAX_YAW_TRAVEL_TIME
}

ALPHA_POWER = 1e-2 # TODO the choice of alpha depends on how much leverage axindfactor has to fine-tune for firm power
ALPHA_THRUST = 1e-2
WEIGHTING_POWER = 10
WEIGHTING_THRUST = 0

class WFEnv(Env):
	def __init__(self, **config: dict):
		super().__init__()
		
		self.episode_time_step = None
		self.offline_probability = (
			config["offline_probability"] if "offline_probability" in config else 0.001
		)
		
		self.floris_input_file = (
			config["floris_input_file"]
			if "floris_input_file" in config
			else "./9turb_floris_input.json"
		)
		
		self.wind_farm = wfct.floris_interface.FlorisInterface(self.floris_input_file)
		
		self.x_bounds = (min(self.wind_farm.layout_x), max(self.wind_farm.layout_x))
		self.y_bounds = (min(self.wind_farm.layout_y), max(self.wind_farm.layout_y))
		
		self.x_nom = self.wind_farm.layout_x
		self.y_nom = self.wind_farm.layout_y
		
		self.n_turbines = len(self.wind_farm.floris.farm.turbines)
		
		# each turbine can take take two actions: yaw angle and axial induction factor
		turbine_action_space = {}
		if YAW_ACTUATION:
			turbine_action_space = {**turbine_action_space, **{'yaw_angle': Discrete(len(YAW_CHANGES))}}
		
		if AX_IND_ACTUATION:
			turbine_action_space = {**turbine_action_space, **{'ai_factor': Box(low=0, high=1/3, shape=())}}
			
		if PTFM_ACTUATION:
			turbine_action_space = {**turbine_action_space, **{'loc_mag': Box(low=0, high=PTFM_RNG, shape=()),
			                                                   'loc_dir': Box(low=0, high=2 * np.pi, shape=())}}
			
		turbine_action_space = Dict(turbine_action_space)  # MultiDiscrete([len(AX_IND_FACTORS), len(YAW_CHANGES)]) # Discrete(len(AX_IND_FACTORS) * len(YAW_CHANGES))
		# format of valid actions
		self._turbine_ids = set(list(range(self.n_turbines)))
		self.action_space = Dict({str(k): turbine_action_space for k in self._turbine_ids})
		self.n_actions = self.n_turbines * len(turbine_action_space)
		
		self.state = None
		
		self._skip_env_checking = False
		
		# agent (single FOWF controller) can see all turbine locations, axial induction factors, yaw angles and online status
		turbine_observation_space = {'online_bool': Discrete(2)} #, 'power': Box(low=0, high=np.infty, shape=())}
		
		if YAW_ACTUATION:
			turbine_observation_space = {**turbine_observation_space,
			                             **{'yaw_angle': Box(low=-20, high=20, shape=()),
			                                'yaw_travel': Box(low=0, high=np.infty, shape=(), dtype=int)}} # Discrete(len(YAW_CHANGES)),
		
		if AX_IND_ACTUATION:
			# TODO normalize to 1 for NNs
			turbine_observation_space = {**turbine_observation_space, **{'ai_factor': Box(low=0, high=1/3, shape=())}} # Discrete(len(AX_IND_FACTORS))
		
		if PTFM_ACTUATION:
			turbine_observation_space = {**turbine_observation_space, **{
				'loc_mag': Box(low=0, high=PTFM_RNG, shape=()),
				'loc_dir': Box(low=0, high=2 * np.pi, shape=())}}
		
		turbine_observation_space = Dict(turbine_observation_space)
		self.observation_space = Dict(
			{
				"fs_wind_speed": Box(low=0, high=np.infty, shape=()),
				"fs_wind_dir": Box(low=180, high=360, shape=()),
				"turbines": Dict({str(k): turbine_observation_space for k in self._turbine_ids})
			}
		)
		self.n_observations = 2 + self.n_turbines * len(turbine_observation_space)
		
		self.mean_layout = [
			(turbine_coords.x1, turbine_coords.x2)
			for turbine_coords in self.wind_farm.floris.farm.turbine_map.coords
		]
		
		turbine_layout_std = (
			config["turbine_layout_std"] if "turbine_layout_std" in config else 1.0
		)
		self.var_layout = turbine_layout_std ** 2 * np.eye(2)  # for x and y coordinate
		
		# set wind speed/dir change probabilities and variability parameters
		self.wind_change_probability = config["wind_change_probability"] # 0.1
		self.wind_speed_var = config["wind_speed_var"] # 0.5
		self.wind_dir_var = config["wind_dir_var"] # 5.0
		self.wind_speed_turb_std = config["wind_speed_turb_std"]  # 0.5
		self.wind_dir_turb_std = config["wind_dir_turb_std"]  # 5.0
		self.max_yaw_travel_thr = config["max_yaw_travel_thr"]
		self.max_yaw_travel_time = config["max_yaw_travel_time"]
		
		self.mean_wind_speed = None
		self.mean_wind_dir = None
		
		self.yaw_travel = [deque(maxlen=int(self.max_yaw_travel_time // DT)) for k in self._turbine_ids]
	
	def _new_layout(self):
		new_layout = np.vstack(
			[
				np.random.multivariate_normal(self.mean_layout[k], self.var_layout)
				for k in self._turbine_ids
			]
		).T
		
		# clip the x coords
		new_layout[0, :] = np.clip(
			new_layout[0, :],
			self.x_bounds[0], self.x_bounds[1]
		)
		
		# clip the y coords
		new_layout[1, :] = np.clip(
			new_layout[1, :],
			self.y_bounds[0], self.y_bounds[1]
		)
		
		new_mag = [
			((new_layout[0, k] - self.x_nom[k]) ** 2 + (new_layout[1, k] - self.y_nom[k]) ** 2) ** 0.5 for k in
			self._turbine_ids]
		new_dir = [np.arctan((new_layout[1, k] - self.y_nom[k]) / (new_layout[0, k] - self.x_nom[k])) for
		                        k in self._turbine_ids]
		for i, d in enumerate(new_dir):
			if d < 0:
				new_dir[i] = 2*np.pi + d
			if np.isnan(d): # if mag == 0
				new_dir[i] = 0
				
		return new_layout, new_mag, new_dir
	
	def _new_online_bools(self):
		return [
			np.random.choice(
				[0, 1], p=[self.offline_probability, 1 - self.offline_probability]
			)
			for _ in range(self.n_turbines)
		]
	
	def _new_wind_speed(self):
		
		# randomly increase or decrease mean wind speed or keep static
		self.mean_wind_speed = self.mean_wind_speed + np.random.choice(
			[-self.wind_speed_var, 0, self.wind_speed_var],
			p=[
				self.wind_change_probability / 2,
				1 - self.wind_change_probability,
				self.wind_change_probability / 2,
			],
		)
		
		# bound the wind speed to given range
		self.mean_wind_speed = np.clip(
			self.mean_wind_speed, WIND_SPEED_RANGE[0], WIND_SPEED_RANGE[1]
		)
		
		return self.mean_wind_speed + np.random.normal(scale=self.wind_speed_turb_std)
	
	def _new_wind_dir(self):
		
		# randomly increase or decrease mean wind direction or keep static
		self.mean_wind_dir = self.mean_wind_dir + np.random.choice(
			[-self.wind_dir_var, 0, self.wind_dir_var],
			p=[
				self.wind_change_probability / 2,
				1 - self.wind_change_probability,
				self.wind_change_probability / 2,
			],
		)
		
		# bound the wind direction to given range
		self.mean_wind_dir = np.clip(
			self.mean_wind_dir, WIND_DIR_RANGE[0], WIND_DIR_RANGE[1]
		)
		return self.mean_wind_dir + np.random.normal(scale=self.wind_dir_turb_std)
	
	def _new_power_ref(self):
		# TODO fetch from file then loop
		return 30 * 1e6
		
	def reset(self, seed=None, options={}):
		self.seed(seed)
		
		# initialize aat random wind speed and direction
		self.mean_wind_speed = np.random.choice(
			np.arange(WIND_SPEED_RANGE[0], WIND_SPEED_RANGE[1], self.wind_speed_var)
		)
		self.mean_wind_dir = np.random.choice(
			np.arange(WIND_DIR_RANGE[0], WIND_DIR_RANGE[1], self.wind_dir_var)
		)
		
		# sample from action space for initial action
		init_action_dict = self.action_space.sample()
		
		# reset layout to original
		init_layout, init_mag, init_dir = self._new_layout()
		
		# randomly sample online bools
		init_online_bools = self._new_online_bools()
		
		# initialize at steady-state
		self.wind_farm = wfct.floris_interface.FlorisInterface(self.floris_input_file)
		self.wind_farm.floris.farm.flow_field.mean_wind_speed = self.mean_wind_speed
		self.wind_farm.reinitialize_flow_field(
			wind_speed=self.mean_wind_speed,
			wind_direction=self.mean_wind_dir,
			layout_array=init_layout,
		)
		
		(
			init_yaw_angles,
			set_init_ax_ind_factors,
			effective_init_ax_ind_factors,
		) = self.get_action_values(init_action_dict, init_online_bools)
		
		for k in self._turbine_ids:
			self.yaw_travel[k].append(abs(init_yaw_angles[k]))
			self.wind_farm.floris.farm.turbines[k].area = self.wind_farm.floris.farm.turbines[k].rotor_radius ** 2 * np.pi
		
		# self.current_ax_ind_factors = set_init_ax_ind_factors
		# self.current_yaw_angles = init_yaw_angles
		# self.current_loc_mag = init_mag
		# self.current_loc_dir = init_dir
		
		self.wind_farm.calculate_wake(
			yaw_angles=init_yaw_angles, axial_induction=effective_init_ax_ind_factors
		)
		
		self.episode_time_step = 0
		
		obs = self._obs(self.mean_wind_speed, self.mean_wind_dir, init_mag, init_dir, set_init_ax_ind_factors, init_yaw_angles, init_online_bools)
		info = {}
		return obs, info
	
	def get_action_values(self, action_dict, online_bools):
		import pdb
		# pdb.set_trace()
		actions = {}
		
		if YAW_ACTUATION:
			yaw_angle_change = np.array(
				[action_dict[str(k)]['yaw_angle'] for k in self._turbine_ids]
			)
			
			yaw_angles = np.array(self.wind_farm.get_yaw_angles()) + (DELTA_YAW * yaw_angle_change)
		
		if AX_IND_ACTUATION:
			set_ax_ind_factors = np.array(
				[action_dict[str(k)]['ai_factor'] for k in self._turbine_ids]
			)
		else:
			thrust_coeffs = self.wind_farm.get_turbine_ct()
			# TODO QUESTION MISHA does thrust coeff need to be multipled by cosd twice? If not there is an error in floris axial_induction
			set_ax_ind_factors = [0.5 / cosd(yaw_angles[k]) * (1 - np.sqrt(1 - thrust_coeffs[k] * cosd(yaw_angles[k])))
			                      for k in self._turbine_ids]
			
		effective_ax_ind_factors = np.array(
			[
				a if online_bool else EPS
				for a, online_bool in zip(set_ax_ind_factors, online_bools)
			]
		)
			
		return yaw_angles, set_ax_ind_factors, effective_ax_ind_factors
	
	def step(self, action_dict):
		"""
        Given the yaw-angle and axial induction factor setting for each wind turbine (action) and the freestream wind speed (disturbance).
        Take a single step (one time-step) in the current episode
        Set the stochastic turbine (x, y) coordinates and online/offline binary variables.
        Get the effective wind speed at each wind turbine.
        Get the power output of each wind turbine and compute the overall rewards.

        """
		
		new_wind_speed = self._new_wind_speed()
		new_wind_dir = self._new_wind_dir()
		
		# TODO how to use this preview information QUESTION ALESSANDRO - include in state? how to include in predicted reward computation?
		self.power_ref = self._new_power_ref()
		
		# Make list of turbine x, y coordinates samples from Gaussian distributions
		new_layout, new_mag, new_dir = self._new_layout()
		
		# Make list of turbine online/offline booleans, offline with some small probability p,
		# if a turbine is offline, set its axial induction factor to 0
		new_online_bools = self._new_online_bools()
		
		(
			new_yaw_angles,
			set_new_ax_ind_factors,
			effective_new_ax_ind_factors,
		) = self.get_action_values(action_dict, new_online_bools)
		# self.current_ax_ind_factors = set_new_ax_ind_factors
		# self.current_yaw_angles = new_yaw_angles
		
		for k in self._turbine_ids:
			self.yaw_travel[k].append(abs(new_yaw_angles[k]))
			
		self.wind_farm.floris.farm.flow_field.mean_wind_speed = self.mean_wind_speed
		self.wind_farm.reinitialize_flow_field(
			wind_speed=new_wind_speed,
			wind_direction=new_wind_dir,
			layout_array=new_layout,
			sim_time=self.episode_time_step,
		)
		self.wind_farm.calculate_wake(
			yaw_angles=new_yaw_angles,
			axial_induction=effective_new_ax_ind_factors,
			sim_time=self.episode_time_step,
		)
		
		# global_reward = self.wind_farm.get_farm_power()
		# turbine_power = {k: self.wind_farm.get_turbine_power()[k] for k in self._turbine_ids}
		
		# Set `done` flag after EPISODE_LEN steps.
		self.episode_time_step += 1
		dones = {"__all__": self.episode_time_step >= EPISODE_LEN}
		dones = dones["__all__"]
		
		# TODO this should be a wide area, just coarsely following power reference
		# TODO QUESTION ALESSANDRO constrain yaw travel by penalizing moving sum in reward - will agent learn from this?
		# yaw_travel
		# scale power to MWs
		# include thrust force as proxy for loading in penalty TODO check computation, integrate with weightings
		turbine_thrust = 0.5 * self.wind_farm.floris.farm.air_density * np.array(self.wind_farm.get_turbine_ct()) * \
		                 np.array([self.wind_farm.floris.farm.turbines[k].area * self.wind_farm.floris.farm.turbines[k].average_velocity for k in self._turbine_ids])
		reward = WEIGHTING_POWER * np.exp(-ALPHA_POWER * ((self.wind_farm.get_farm_power() - self.power_ref) * 1e-6)*2) - WEIGHTING_THRUST * np.exp(-ALPHA_THRUST * (max(turbine_thrust) * 1e-6)*2)
		if np.any([sum(self.yaw_travel[k]) > self.max_yaw_travel_thr for k in self._turbine_ids]):
			reward = 0
			
		# Update observation
		obs = self._obs(new_wind_speed, new_wind_dir, new_mag, new_dir, set_new_ax_ind_factors, new_yaw_angles, online_bools=new_online_bools)
		
		# TODO terminate if yaw travel exceeds allowed amount or power deviates too far from reference
		terminated = False
		truncated = self.episode_time_step == (24 * 3600)
		
		return obs, reward, terminated, truncated, {}
	
	def _obs(self, wind_speed, wind_dir, loc_mag, loc_dir, set_ax_ind_factors, yaw_angles, online_bools):
		# return {k: self._turbine_obs(k, wind_speed, wind_dir, online_bools) for k in self._turbine_ids}
		try:
			assert all(
				turbine.ai_set in AX_IND_FACTORS or turbine.ai_set == 0
				for turbine in self.wind_farm.floris.farm.turbines
			) and all(
				turbine.yaw_angle in YAW_CHANGES
				for turbine in self.wind_farm.floris.farm.turbines
			)
		except Exception as e:
			print(e)
		
		turbine_obs = {}
		for k in self._turbine_ids:
			turbine_k_obs = {'online_bool': online_bools[k]} # 'power': self.wind_farm.get_turbine_power()[k]}
			if YAW_ACTUATION:
				turbine_k_obs = {**turbine_k_obs,
				                 **{'yaw_angle': yaw_angles[k],
				                    'yaw_travel': np.sum(self.yaw_travel[k])}} # np.where(YAW_CHANGES == yaw_angles[k])[0][0],
				
			if AX_IND_ACTUATION:
				turbine_k_obs = {**turbine_k_obs, **{'ai_factor': set_ax_ind_factors[k]}} # np.where(AX_IND_FACTORS == set_ax_ind_factors[k])[0][0],
				
			if PTFM_ACTUATION:
				turbine_k_obs = {**turbine_k_obs, **{'loc_mag': loc_mag[k], 'loc_dir': loc_dir[k]}}
			
			turbine_obs[str(k)] = turbine_k_obs
			
		return {
			"fs_wind_speed": wind_speed,
			"fs_wind_dir": wind_dir,
			"turbines": turbine_obs
		}
	
	def seed(self, seed=None):
		random.seed(seed)


if __name__ == '__main__':
	
	# register the environment
	gym.register('wf_env', WFEnv, kwargs=ENV_CONFIG)
	
	# create the environment
	wf_env = gym.make("wf_env")  # WFEnv(env_config)
	
	# reset the environment to get the first observation of the environment
	observation, info = wf_env.reset(seed=1, options={})
	
	for _ in range(50):
		
		action = wf_env.action_space.sample()  # agent policy that uses the observation and info
		
		# agent performs an action in the environment
		# agent receives a new observation from the updated environment along with a reward for taking the action.
		# One such action-observation exchange is referred to as a timestep.
		observation, reward, terminated, truncated, info = wf_env.step(action)
		# [observation['turbines'][k]['loc_dir'] for k in range(9)]
		if terminated or truncated:
			observation, info = wf_env.reset()
	
	wf_env.close()
