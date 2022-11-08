import gym
from ray.rllib.algorithms.qmix import QMixConfig
from ray import tune
import numpy as np
from floridyn import tools as wfct
import os
import multiprocessing as mp

DT = 1.0 # discrete-time step for wind farm control
EPISODE_LEN = int(10 * 60 // DT) # 10 minute episode length

class RLALgo():
	def __init__(self, env):
		self.env = env

	def configure_rl_algo(self):
		# Configure the algorithm.
		config = (QMixConfig()
				  .environment(env=self.env)
				  .rollouts(num_workers=mp.cpu_count())
				  .framework('torch')
				  .training(mixer='qmix', double_q=True)
				  .exploration(
					exploration_config={
						"final_epsilon": 0.0,
					}
					)
				  .evaluation(evaluation_num_workers=1)
				  .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")))
				  )

		# Create our RLlib Trainer from the config object.
		# self.algo = config.build()

		# OR, train viar Ray Tune to tune hyperparameters
		tune.run('QMIX', config=config.to_dict())

	def train(self, n_iters):
		for _ in range(n_iters):
			self.algo.train()

	def evaluate(self):
		self.algo.evaluate()

def step_wind_field(wind_speed_mean, wind_dir_mean, wind_speed_TI, wind_dir_TI):
	ws = np.random.normal(loc=wind_speed_mean, scale=(wind_speed_TI / 100) * wind_speed_mean)[0]  # np.random.uniform(low=8, high=8.3)
	wd = np.random.normal(loc=wind_dir_mean, scale=(wind_dir_TI / 100) * wind_dir_mean)[0]
	return ws, wd

class FOWFEnv(gym.Env):
	def __init__(self,
				 floris_input_file="./9turb_floris_input.json",
				 turbine_layout_std=1.,
				 offline_probability=0.001):
		# TODO use gym's Discrete object for action space
		self.action_space = {
			'ax_ind_factor_set': [0.11, 0.22, 0.33],
			'yaw_angle_set': [-15, -10, -5, 0, 5, 10, 15]
		}

		self.state_space = {
			'layout': [],
			'ax_ind_factors_actual': [],
			'yaw_angles_actual': [],
			'online_bool': []
		}

		self.observation_space = self.state_space

		self.current_observation = {'layout': [],
									'ax_ind_factors': [],
									'yaw_angles': [],
									'online_bool': []}
		self.episode_time_step = None
		self.offline_probability = offline_probability

		self.floris_input_file = floris_input_file
		self.wind_farm = wfct.floris_interface.FlorisInterface(self.floris_input_file)
		self.n_turbines = len(self.wind_farm.floris.farm.turbines)

		self.mean_layout = [(turbine_coords.x1, turbine_coords.x2) for turbine_coords in self.wind_farm.floris.farm.turbine_map.coords]
		self.var_layout = turbine_layout_std**2 * np.eye(2) # for x and y coordinate

	def reset(self, init_action, init_disturbance):

		new_layout = np.vstack([np.random.multivariate_normal(self.mean_layout[t_idx], self.var_layout)
					  for t_idx in range(self.n_turbines)]).T

		new_online_bools = [np.random.choice([0, 1], p=[self.offline_probability, 1 - self.offline_probability])
							for t_idx in range(self.n_turbines)]

		# initialize at steady-state
		self.wind_farm = wfct.floris_interface.FlorisInterface(self.floris_input_file)
		self.wind_farm.floris.farm.flow_field.mean_wind_speed = init_disturbance['wind_speed']
		self.wind_farm.reinitialize_flow_field(wind_speed=init_disturbance['wind_speed'],
											   wind_direction=init_disturbance['wind_dir'],
											   layout_array=new_layout)
		self.wind_farm.calculate_wake(yaw_angles=init_action['yaw_angle_set'],
									  axial_induction=[a * online_bool
													   for a, online_bool in
													   zip(init_action['ax_ind_factor_set'], new_online_bools)])


		self.episode_time_step = 0

		self.current_observation = {'layout':
										[[turbine.coords.x1 for turbine in self.wind_farm.floris.farm.turbine_map],
										 [turbine.coords.x2 for turbine in self.wind_farm.floris.farm.turbine_map]],
									'ax_ind_factors': [turbine.aI for turbine in self.wind_farm.floris.farm.turbines],
									'yaw_angles': [turbine.yaw_angle for turbine in self.wind_farm.floris.farm.turbines],
									'online_bool': new_online_bools}

		return self.current_observation

	def step(self, action, disturbance):
		"""
		Given the yaw-angle and axial induction factor setting for each wind turbine (action) and the freestream wind speed (disturbance).
		Take a single step (one time-step) in the current episode
		Set the stochastic turbine (x, y) coordinates and online/offline binary variables.
		Get the effective wind speed at each wind turbine.
		Get the power output of each wind turbine and compute the overall rewards.

		"""


		# Make list of turbine x, y coordinates samples from Gaussian distributions
		new_layout = np.vstack([np.random.multivariate_normal(self.mean_layout[t_idx], self.var_layout)
								for t_idx in range(self.n_turbines)]).T

		# Make list of turbine online/offline booleans, offline with some small probability p,
		# if a turbine is offline, set its axial induction factor to 0
		new_online_bools = [np.random.choice([0, 1], p=[self.offline_probability, 1 - self.offline_probability])
							for t_idx in range(self.n_turbines)]

		self.wind_farm.floris.farm.flow_field.mean_wind_speed = disturbance['wind_speed']
		self.wind_farm.reinitialize_flow_field(wind_speed=disturbance['wind_speed'],
											   wind_direction=disturbance['wind_dir'],
											   layout_array=new_layout,
											   sim_time=self.episode_time_step)
		self.wind_farm.calculate_wake(yaw_angles=action['yaw_angle_set'],
									  axial_induction=[a * online_bool
													   for a, online_bool in
													   zip(action['ax_ind_factor_set'], new_online_bools)],
									  sim_time=self.episode_time_step)

		reward = self.wind_farm.get_farm_power()
		# for i, turbine in enumerate(fi.floris.farm.turbines):
		# 	turbine_powers[i].append(turbine.power / 1e6)

		# Set `done` flag after EPISODE_LEN steps.
		self.episode_time_step += 1
		done = self.episode_time_step >= EPISODE_LEN

		# Update observation
		self.current_observation = {'layout': [],
									'ax_ind_factors': [],
									'yaw_angles': [],
									'online_bool': []}

		return self.current_observation, reward, done
