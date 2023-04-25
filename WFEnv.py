import functools

import numpy as np
from floridyn import tools as wfct
from floridyn.utilities import cosd
import os
import random
import gymnasium as gym
from pettingzoo import ParallelEnv
from gymnasium import Env
from collections import deque
from Agent import YawAgent, AxIndFactorAgent
from constants import DT, EPISODE_LEN, WIND_SPEED_RANGE, WIND_DIR_RANGE, \
					  EPS, YAW_CHANGES, DELTA_YAW, ACTION_MAPPING, YAW_LIMITS, OBSERVATION_LIMITS, AI_FACTOR_LIMITS, \
					  YAW_ACTUATION, AI_FACTOR_ACTUATION, PTFM_ACTUATION, \
					  ENV_CONFIG, \
					  ALPHA, WEIGHTING, IS_DYNAMIC, N_PREVIEW_TIME_STEPS, N_PREVIOUS_TIME_STEPS
from results import plot_wind_farm
from itertools import count

# TODO consider preview of reference power
# TODO QUESTION to include power in observations if it is part of the reward or not - is it possible to 'overflow' observations unecessarily

# TODO find realistic values for yaw travel limit, yaw rate of change, power tracking reference

class WFEnv(ParallelEnv):
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
		self.n_turbines = len(self.wind_farm.floris.farm.turbines)
		self.wind_farm.turbine_indices = list(range(self.n_turbines))
		
		self.x_bounds = (min(self.wind_farm.layout_x), max(self.wind_farm.layout_x))
		self.y_bounds = (min(self.wind_farm.layout_y), max(self.wind_farm.layout_y))
		
		self.x_nom = self.wind_farm.layout_x
		self.y_nom = self.wind_farm.layout_y
		
		self.agent_ids = []
		self.agents = {}
		if YAW_ACTUATION:
			self.agent_ids += ['yaw_angle']
			self.agents['yaw_angle'] = YawAgent(self.n_turbines)
		if AI_FACTOR_ACTUATION:
			self.agent_ids += ['ai_factor']
			self.agents['ai_factor'] = AxIndFactorAgent(self.n_turbines)
		
		self._action_spaces = {agent_id: agent.action_space for agent_id, agent in self.agents.items()}
		self._observation_spaces = {agent_id: agent.observation_space for agent_id, agent in self.agents.items()}
		
		self.n_actions = {agent_id: agent.n_actions for agent_id, agent in self.agents.items()}
		self.n_observations = {agent_id: agent.n_observations for agent_id, agent in self.agents.items()}

		# format of valid actions
		self._turbine_ids = set(list(range(self.n_turbines)))
		
		self.state = None
		
		self._skip_env_checking = False
			
		self.mean_layout = [
			(turbine_coords.x1, turbine_coords.x2)
			for turbine_coords in self.wind_farm.floris.farm.turbine_map.coords
		]
		
		turbine_layout_std = (
			config["turbine_layout_std"] if "turbine_layout_std" in config else 1.0
		)
		self.var_layout = turbine_layout_std ** 2 * np.eye(2)  # for x and y coordinate
		
		# set wind speed/dir change probabilities and variability parameters
		self.wind_speed_change_probability = config["wind_speed_change_probability"] # 0.1
		self.wind_dir_change_probability = config["wind_dir_change_probability"] # 0.1
		self.wind_speed_var = config["wind_speed_var"] # 0.5
		self.wind_dir_var = config["wind_dir_var"] # 5.0
		self.wind_speed_turb_std = config["wind_speed_turb_std"]  # 0.5
		self.wind_dir_turb_std = config["wind_dir_turb_std"]  # 5.0
		self.max_yaw_travel_thr = config["max_yaw_travel_thr"]
		self.max_yaw_travel_time = config["max_yaw_travel_time"]
		self.max_episode_time_step = config["max_episode_time_step"]
		
		self.mean_wind_speed = None
		self.mean_wind_dir = None
		
		self.yaw_buffer = [deque(maxlen=int(self.max_yaw_travel_time // DT)) for k in self._turbine_ids]
	
	# this cache ensures that same space object is returned for the same agent
	# allows action space seeding to work as expected
	@functools.lru_cache(maxsize=None)
	def observation_space(self, agent_id):
		return self._observation_spaces[agent_id]

	@functools.lru_cache(maxsize=None)
	def action_space(self, agent_id):
		return self._action_spaces[agent_id]
	
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
				self.wind_speed_change_probability / 2,
				1 - self.wind_speed_change_probability,
				self.wind_speed_change_probability / 2,
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
				self.wind_dir_change_probability / 2,
				1 - self.wind_dir_change_probability,
				self.wind_dir_change_probability / 2,
			],
		)
		
		# bound the wind direction to given range
		self.mean_wind_dir = np.clip(
			self.mean_wind_dir, WIND_DIR_RANGE[0], WIND_DIR_RANGE[1]
		)
		return self.mean_wind_dir + np.random.normal(scale=self.wind_dir_turb_std)
	
	# def _new_power_ref(self):
	# 	# TODO fetch from file then preview
	# 	return 30 * 1e6
		
	def reset(self, seed=None, options={}):
		self.seed(seed)
		
		# initialize aat random wind speed and direction
		self.mean_wind_speed = np.random.choice(
			np.arange(WIND_SPEED_RANGE[0], WIND_SPEED_RANGE[1], self.wind_speed_var)
		)
		self.mean_wind_dir = np.random.choice(
			np.arange(WIND_DIR_RANGE[0], WIND_DIR_RANGE[1], self.wind_dir_var)
		)
		
		# reset layout to original
		# init_layout, init_mag, init_dir = self._new_layout()
		
		# initialize at steady-state
		self.wind_farm = wfct.floris_interface.FlorisInterface(self.floris_input_file)
		self.wind_farm.turbine_indices = list(range(self.n_turbines))
		self.wind_farm.floris.farm.flow_field.mean_wind_speed = self.mean_wind_speed
		self.wind_farm.reinitialize_flow_field(
			wind_speed=self.mean_wind_speed,
			wind_direction=self.mean_wind_dir,
			# layout_array=init_layout,
		)
		
		# randomly sample online bools
		init_online_bools = self._new_online_bools()
		
		# sample from action space for initial action for both agents
		init_action_dict = {'yaw_angle': self.agents['yaw_angle'].action_space.sample() if YAW_ACTUATION else None,
		                    'ai_factor': self.agents['ai_factor'].action_space.sample() if AI_FACTOR_ACTUATION else None
		                    }
		
		(
			init_yaw_angles,
			set_init_ax_ind_factors,
			effective_init_ax_ind_factors,
		) = self.get_action_values(init_action_dict, init_online_bools)
		
		for k in self._turbine_ids:
			self.yaw_buffer[k].append(init_yaw_angles[k])
			self.wind_farm.floris.farm.turbines[k].area = self.wind_farm.floris.farm.turbines[k].rotor_radius ** 2 * np.pi
		
		# initialize world
		self.wind_farm.calculate_wake(
			yaw_angles=init_yaw_angles, axial_induction=effective_init_ax_ind_factors
		)
		
		self.episode_time_step = 0

		# initialize previous states - buffer of states over last X time-steps for each turbine
		self.previous_observations = {agent_id:
			{'turbines':
			{str(k): {
				'online_bool': deque([init_online_bools[k]] *  N_PREVIOUS_TIME_STEPS, maxlen=N_PREVIOUS_TIME_STEPS),
				'wind_speed': deque([self.wind_farm.floris.farm.wind_speed[k]] *  N_PREVIOUS_TIME_STEPS, maxlen=N_PREVIOUS_TIME_STEPS),
				'wind_dir': deque([self.wind_farm.floris.farm.wind_direction[k]] *  N_PREVIOUS_TIME_STEPS, maxlen=N_PREVIOUS_TIME_STEPS),
				'ai_factor': deque([set_init_ax_ind_factors] * N_PREVIOUS_TIME_STEPS, maxlen=N_PREVIOUS_TIME_STEPS),
				'yaw_angle': deque([init_yaw_angles] * N_PREVIOUS_TIME_STEPS, maxlen=N_PREVIOUS_TIME_STEPS)
			} for k in self._turbine_ids}
		} for agent_id in self.agents}
		
		# yaw_buffer is a list of deque objects containing the yaw values for the last X time-steps for each turbine
		self.yaw_buffer = [deque([0] * int(self.max_yaw_travel_time // DT), maxlen=int(self.max_yaw_travel_time // DT)) for k in self._turbine_ids]
		
		self.power_ref_preview = options['power_ref_preview']
		
		rotor_thrust = 0.5 * self.wind_farm.floris.farm.air_density * np.array(self.wind_farm.get_turbine_ct()) * \
		               np.array([self.wind_farm.floris.farm.turbines[k].area * self.wind_farm.floris.farm.turbines[
			               k].average_velocity for k in self._turbine_ids])
		turbine_power = self.wind_farm.get_turbine_power()
		
		# yaw_travel is a list of total yaw travel over the last X time-steps for each turbine
		yaw_travel = [sum(abs(np.diff(self.yaw_buffer[k]))) for k in self._turbine_ids]
		
		obs = self._obs(self.wind_farm.floris.farm.wind_speed, self.wind_farm.floris.farm.wind_direction,
		                set_init_ax_ind_factors, init_yaw_angles, init_online_bools,
		                power=turbine_power, rotor_thrust=rotor_thrust, yaw_travel=yaw_travel, sampling_time=options['sampling_time'])
		info = {}
		return obs, info
	
	def get_action_values(self, action_dict, online_bools):
		import pdb
		# pdb.set_trace()
		actions = {}
		
		# if actions sampled from yaw agent are None (no yaw actuation), consider default, facing wind
		
		# if we are using a yaw actuation agent, get the commanded yaw angles
		# If it is not the yaw sampling time, this will be None
		# if we are not using the yaw actuation agent, deafault is to face wind
		# positive yaw angle is measured from westerly direction CCW
		# wind direction 270 is westerly, 180 northerly
		if YAW_ACTUATION:
			# if it is not the right sampling time for a yaw angle change
			if action_dict['yaw_angle'] is None:
				yaw_angles = self.wind_farm.get_yaw_angles()
			else:
				yaw_angle_change = [ACTION_MAPPING['yaw_angle'](x) for x in action_dict['yaw_angle']]
				yaw_angles = np.array(self.wind_farm.get_yaw_angles()) + (DELTA_YAW * np.array(yaw_angle_change))
				
		else:
			yaw_angles = 270 - self.wind_farm.floris.farm.wind_direction
		
		# QUESTION is this the best way to clip... RL agent won't know not to increase at boundaries ...
		yaw_angles = np.clip(yaw_angles, YAW_LIMITS[0], YAW_LIMITS[1])
		# self.wind_farm.reinitialize_flow_field(
		# 	wind_speed=self.mean_wind_speed,
		# 	wind_direction=180
		# )
		# self.wind_farm.calculate_wake(
		# 	yaw_angles=[0] * 9
		# )
		# fig = plot_wind_farm(self.wind_farm)
		
		# if we are controlling ai_factor with a RL agent
		if AI_FACTOR_ACTUATION:
			# if it is not the time-step for the ai_factor execution
			if action_dict['ai_factor'] is None:
				set_ax_ind_factors = [self.wind_farm.floris.farm.turbines[k].aI_set for k in self._turbine_ids]
			else:
				set_ax_ind_factors = [ACTION_MAPPING['ai_factor'](x) for x in action_dict['ai_factor']]
		else:
			# optimize based on wind speed
			thrust_coeffs = self.wind_farm.get_turbine_ct()
			# TODO QUESTION MISHA does thrust coeff need to be multipled by cosd twice? If not there is an error in floris axial_induction
			# set_ax_ind_factors = [0.5 / cosd(yaw_angles[k]) * (1 - np.sqrt(1 - thrust_coeffs[k] * cosd(yaw_angles[k])))
			#                       for k in self._turbine_ids]
			set_ax_ind_factors = [0.5 / cosd(yaw_angles[k]) * (1 - np.sqrt(1 - thrust_coeffs[k])) for k in self._turbine_ids]
		
		if np.any(np.array(set_ax_ind_factors) < 0):
			print('oh no')
		
		
		effective_ax_ind_factors = np.array(
			[
				a if online_bool else EPS
				for a, online_bool in zip(set_ax_ind_factors, online_bools)
			]
		)
			
		return yaw_angles, set_ax_ind_factors, effective_ax_ind_factors
		
	def step(self, action_dict, t, sampling_time):
		"""
        Given the yaw-angle and axial induction factor setting for each wind turbine (action) and the freestream wind speed (disturbance).
        Take a single step (one time-step) in the current episode
        Set the stochastic turbine (x, y) coordinates and online/offline binary variables.
        Get the effective wind speed at each wind turbine.
        Get the power output of each wind turbine and compute the overall rewards.
        """
		
		new_wind_speed = self._new_wind_speed()
		new_wind_dir = self._new_wind_dir()
		
		# TODO how to use this power preview information QUESTION ALESSANDRO - include in state? how to include in predicted reward computation? - do we try to optimize yaw actuation over a future time-interval?
		
		# Make list of turbine x, y coordinates samples from Gaussian distributions
		# new_layout, new_mag, new_dir = self._new_layout()
		
		# Make list of turbine online/offline booleans, offline with some small probability p,
		# if a turbine is offline, set its axial induction factor to 0
		new_online_bools = self._new_online_bools()
		
		# set action for each agent
		(
			new_yaw_angles,
			set_new_ax_ind_factors,
			effective_new_ax_ind_factors,
		) = self.get_action_values(action_dict, new_online_bools)
		
		# update previous states
		for agent_id in self.previous_observations:
			for k in self._turbine_ids:
				self.previous_observations[agent_id]['turbines'][str(k)]['online_bool'].append(new_online_bools[k])
				self.previous_observations[agent_id]['turbines'][str(k)]['wind_speed'].append(
					self.wind_farm.floris.farm.wind_speed[k])
				self.previous_observations[agent_id]['turbines'][str(k)]['wind_dir'].append(
					self.wind_farm.floris.farm.wind_direction[k])
				
				# if the sampling time for ai_factor has passed, reglardless of whether ai_factor actuation is being used or not
				if t % sampling_time['ai_factor'] == 0:
					self.previous_observations[agent_id]['turbines'][str(k)]['ai_factor'].append(set_new_ax_ind_factors[k])
				
				# if the sampling time for yaw angle has passed, reglardless of whether yaw angle  actuation is being used or not
				if t % sampling_time['yaw_angle'] == 0:
					self.previous_observations[agent_id]['turbines'][str(k)]['yaw_angle'].append(new_yaw_angles[k])
		
		
		# for agent_id, agent in self.agents.items():
		# 	self._set_action(action[agent_id], agent, self.action_space[agent_id])
		
		# advance world state
		self.wind_farm.floris.farm.flow_field.mean_wind_speed = self.mean_wind_speed
		self.wind_farm.reinitialize_flow_field(
			wind_speed=new_wind_speed,
			wind_direction=new_wind_dir,
			# layout_array=new_layout,
			sim_time=self.episode_time_step if IS_DYNAMIC else None,
		)
		self.yaw_angles = new_yaw_angles
		self.eff_ai_factors = effective_new_ax_ind_factors
		self.ai_factors = set_new_ax_ind_factors
		self.wind_farm.calculate_wake(
			yaw_angles=new_yaw_angles,
			axial_induction=effective_new_ax_ind_factors,
			sim_time=self.episode_time_step if IS_DYNAMIC else None,
		)
		
		for k in self._turbine_ids:
			self.yaw_buffer[k].append(new_yaw_angles[k])
		
		# global_reward = self.wind_farm.get_farm_power()
		# turbine_power = {k: self.wind_farm.get_turbine_power()[k] for k in self._turbine_ids}
		
		# Update world observation, and for each agent
		self.rotor_thrust = 0.5 * self.wind_farm.floris.farm.air_density * np.array(self.wind_farm.get_turbine_ct()) * \
		                 np.array([self.wind_farm.floris.farm.turbines[k].area * self.wind_farm.floris.farm.turbines[
			                 k].average_velocity for k in self._turbine_ids])
		self.turbine_power = self.wind_farm.get_turbine_power()
		self.yaw_travel = np.array([sum(abs(np.diff(self.yaw_buffer[k]))) for k in self._turbine_ids])
		
		obs = self._obs(self.wind_farm.floris.farm.wind_speed, self.wind_farm.floris.farm.wind_direction,
		                set_new_ax_ind_factors, new_yaw_angles,
		                online_bools=new_online_bools, power=self.turbine_power,
		                rotor_thrust=self.rotor_thrust, yaw_travel=self.yaw_travel, sampling_time=sampling_time)
				
		# compute reward for each agent
		
		# yaw_travel
		# scale power to MWs
		# include thrust force as proxy for loading in penalty
		
		# TODO fix scaling
		# TODO question, if yaw angle goes out of bounds. end episode or just penalize?
		self.farm_power = self.wind_farm.get_farm_power()
		self.power_tracking_error = ((self.farm_power - self.power_ref_preview[self.episode_time_step]) * 1e-6)**2
		rotor_thrust_error = (max(self.rotor_thrust) * 1e-6)**2
		yaw_travel_error = (max(self.yaw_travel) / self.max_yaw_travel_thr)**2
		
		# TODO QUESTION how to distinguish which action led to which reward..
		reward = {agent_id: (WEIGHTING[agent_id]['power'] * np.exp(-ALPHA[agent_id]['power'] * self.power_tracking_error)
		         - WEIGHTING[agent_id]['rotor_thrust'] * np.exp(-ALPHA[agent_id]['rotor_thrust'] * rotor_thrust_error)
		         - WEIGHTING[agent_id]['yaw_travel'] * np.exp(-ALPHA[agent_id]['yaw_travel'] * yaw_travel_error))
		                     if action_dict[agent_id] is not None else None for agent_id in self.agents}
		
		# all agents get total reward in cooperative case
		# reward = np.sum(reward_n)
		# if self.shared_reward:
		# 	reward_n = [reward] * self.n
		
		# Set `done` flag after EPISODE_LEN steps.
		self.episode_time_step += 1
		terminated = {agent_id: False for agent_id in self.agent_ids}
		truncated = {agent_id: self.episode_time_step == self.max_episode_time_step for agent_id in self.agent_ids}
		
		return obs, reward, terminated, truncated, {agent_id: {} for agent_id in self.agent_ids}
	
	def _obs(self, wind_speed, wind_dir, set_ax_ind_factors, yaw_angles, online_bools, power, rotor_thrust, yaw_travel, sampling_time, loc_mag=None, loc_dir=None):
		
		turbine_observation = {agent_id: {str(k): {
			'online_bool': tuple(list(self.previous_observations[agent_id]['turbines'][str(k)]['online_bool']) + [online_bools[k]]),
			'power': power[k] / OBSERVATION_LIMITS['turbine_power'][1],
			'wind_speed': tuple(list(self.previous_observations[agent_id]['turbines'][str(k)]['wind_speed'])
			                    + [wind_speed[k] / OBSERVATION_LIMITS['wind_speed'][1]]),
			'wind_dir': tuple(list(self.previous_observations[agent_id]['turbines'][str(k)]['wind_dir'])
			                  + [wind_dir[k] / OBSERVATION_LIMITS['wind_direction'][1]]),
			'yaw_angle': tuple(list(self.previous_observations[agent_id]['turbines'][str(k)]['yaw_angle'])
			                   + [yaw_angles[k] / YAW_LIMITS[1]]),
			'yaw_travel': yaw_travel[k] / YAW_LIMITS[1],
			'ai_factor': tuple(list(self.previous_observations[agent_id]['turbines'][str(k)]['ai_factor'])
			                   + [set_ax_ind_factors[k] / AI_FACTOR_LIMITS[1]]),
			'rotor_thrust': rotor_thrust[k] / OBSERVATION_LIMITS['rotor_thrust'][1]
		} for k in self._turbine_ids} for agent_id in self.agents}
		
		return {agent_id: {
			"power_ref_preview": tuple(
				np.array(self.power_ref_preview[
				self.episode_time_step:self.episode_time_step + (N_PREVIEW_TIME_STEPS * sampling_time[agent_id]) + 1:sampling_time[agent_id]])
				/ OBSERVATION_LIMITS['turbine_power'][1]),
			"turbines": {str(k): turbine_observation[agent_id][str(k)] for k in self._turbine_ids}
		} for agent_id in self.agents}
	
	def seed(self, seed=None):
		random.seed(seed)
	
	def close(self):
		pass


if __name__ == '__main__':
	
	# register the environment
	# gym.register('wf_env', WFEnv, kwargs=ENV_CONFIG)
	# gym.register('wf_env', WFEnv, kwargs=ENV_CONFIG)

	# create the environment
	# wf_env = gym.make("wf_env")  # WFEnv(env_config)
	wf_env = WFEnv(**ENV_CONFIG)
	
	# load power reference signal
	power_ref_preview = [30 * 1e6] * int(24 * 3600 // DT)
	
	# reset the environment to get the first observation of the environment
	observation, info = wf_env.reset(seed=1, options={'power_ref_preview': power_ref_preview})
	
	for dt in range(50): # count():
		
		action = {agent_id: wf_env.action_space(agent_id).sample() for agent_id in wf_env.agent_ids}  # agent policy that uses the observation and info
		
		# agent performs an action in the environment
		# agent receives a new observation from the updated environment along with a reward for taking the action.
		# One such action-observation exchange is referred to as a timestep.
		observation, reward, terminated, truncated, info = wf_env.step(action)
		# [observation['turbines'][k]['loc_dir'] for k in range(9)]
		if terminated or truncated:
			observation, info = wf_env.reset(seed=1, options={'power_ref_preview': power_ref_preview})
	
	wf_env.close()
