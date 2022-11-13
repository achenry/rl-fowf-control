import gym
from gym.core import Optional
from ray.rllib.algorithms.qmix import QMixConfig
from ray.rllib.env.env_context import EnvContext
from ray import tune, air
import numpy as np
from floridyn import tools as wfct
import os
import multiprocessing as mp
import random

DT = 1.0 # discrete-time step for wind farm control
EPISODE_LEN = int(10 * 60 // DT) # 10 minute episode length

# TODO combine actions into single vector
# TODO integrate wind speed/direction variation in logic => some probability of mean wind speed increasing/decreasing by 1 m/s, some turbulence added on; same for wind direction AOIFE

class RLALgo():
	def __init__(self, env):
		self.env = env

	def configure_rl_algo(self):
		# Configure the algorithm.
		config = (QMixConfig()
				  .environment(env=self.env)
				  .rollouts()
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
	# def __init__(self,
	# 			 floris_input_file="./9turb_floris_input.json",
	# 			 turbine_layout_std=1.,
	# 			 offline_probability=0.001):
	
	def __init__(self, config: EnvContext):
		
		
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
		self.offline_probability = config['offline_probability'] if 'offline_probability' in config else 0.001

		self.floris_input_file = config['floris_input_file'] if 'floris_input_file' in config else "./9turb_floris_input.json"
		self.wind_farm = wfct.floris_interface.FlorisInterface(self.floris_input_file)
		self.n_turbines = len(self.wind_farm.floris.farm.turbines)
		
		self.ax_ind_factor_indices = np.arange(0, n_turbines)
		self.yaw_angle_indices = np.arange(n_turbines, 2 * n_turbines)

		self.mean_layout = [(turbine_coords.x1, turbine_coords.x2) for turbine_coords in self.wind_farm.floris.farm.turbine_map.coords]
		
		turbine_layout_std = config['turbine_layout_std'] if 'turbine_layout_std' in config else 1.
		self.var_layout = turbine_layout_std**2 * np.eye(2) # for x and y coordinate
		
		self.disturbance = None
		
		# set wind speed/dir change probabilities and variability parameters
		self.wind_change_probability = 0.1
		self.wind_speed_var = 0.5
		self.wind_dir_var = 5.0
		
		self.mean_wind_speed = None
		self.mean_wind_dir = None

	# def get_disturbance(self):
	# 	wind_speed = None
	# 	wind_dir = None
	#
	# 	return wind_speed, wind_dir
	
	def reset(self, options: Optional[dict]):
		
		init_ax_ind_factors = options['init_action'][self.ax_ind_factor_indices]
		init_yaw_angles = options['init_action'][self.yaw_angle_indices]
		self.mean_wind_speed = options['mean_wind_speed']
		self.mean_wind_dir = options['mean_wind_dir']

		new_layout = np.vstack([np.random.multivariate_normal(self.mean_layout[t_idx], self.var_layout)
					  for t_idx in range(self.n_turbines)]).T

		new_online_bools = [np.random.choice([0, 1], p=[self.offline_probability, 1 - self.offline_probability])
							for _ in range(self.n_turbines)]

		# initialize at steady-state
		self.wind_farm = wfct.floris_interface.FlorisInterface(self.floris_input_file)
		self.wind_farm.floris.farm.flow_field.mean_wind_speed = self.mean_wind_speed
		self.wind_farm.reinitialize_flow_field(wind_speed=self.mean_wind_speed,
											   wind_direction=self.mean_wind_dir,
											   layout_array=new_layout)
		self.wind_farm.calculate_wake(yaw_angles=init_yaw_angles,
									  axial_induction=[a * online_bool
													   for a, online_bool in
													   zip(init_ax_ind_factors, new_online_bools)])


		self.episode_time_step = 0

		self.current_observation = {'layout':
										[[coords.x1 for coords in self.wind_farm.floris.farm.turbine_map.coords],
										 [coords.x2 for coords in self.wind_farm.floris.farm.turbine_map.coords]],
									'ax_ind_factors': [turbine.aI for turbine in self.wind_farm.floris.farm.turbines],
									'yaw_angles': [turbine.yaw_angle for turbine in self.wind_farm.floris.farm.turbines],
									'online_bool': new_online_bools,
		                            'wind_speed': self.mean_wind_speed,
		                            'wind_direction': self.mean_wind_dir}

		return self.current_observation

	def step(self, action):
		
		"""
		Given the yaw-angle and axial induction factor setting for each wind turbine (action) and the freestream wind speed (disturbance).
		Take a single step (one time-step) in the current episode
		Set the stochastic turbine (x, y) coordinates and online/offline binary variables.
		Get the effective wind speed at each wind turbine.
		Get the power output of each wind turbine and compute the overall rewards.

		"""
		
		self.mean_wind_speed = self.mean_wind_speed \
		                       + np.random.choice([-self.wind_speed_var, 0, self.wind_speed_var],
		                                    p=[self.wind_change_probability / 2,
		                                       1 - self.wind_change_probability,
		                                       self.wind_change_probability / 2])
		self.mean_wind_dir = self.mean_wind_dir \
		                       + np.random.choice([-self.wind_dir_var, 0, self.wind_dir_var],
		                                          p=[self.wind_change_probability / 2,
		                                             1 - self.wind_change_probability,
		                                             self.wind_change_probability / 2])
		
		dev_wind_speed = np.random.normal(scale=self.wind_speed_var)
		dev_wind_dir = np.random.normal(scale=self.wind_dir_var)
		
		new_wind_speed = self.mean_wind_speed + np.random.normal(scale=self.wind_speed_var)
		new_wind_dir = self.mean_wind_dir + np.random.normal(scale=self.wind_dir_var)
		
		# Make list of turbine x, y coordinates samples from Gaussian distributions
		new_layout = np.vstack([np.random.multivariate_normal(self.mean_layout[t_idx], self.var_layout)
		                        for t_idx in range(self.n_turbines)]).T
		
		# Make list of turbine online/offline booleans, offline with some small probability p,
		# if a turbine is offline, set its axial induction factor to 0
		new_online_bools = [np.random.choice([0, 1], p=[self.offline_probability, 1 - self.offline_probability])
		                    for t_idx in range(self.n_turbines)]
			
		new_ax_ind_factors = [a * online_bool for a, online_bool in
													   zip(action['ax_ind_factor_set'], new_online_bools)]
		new_yaw_angles = action['yaw_angle_set']
		
		self.wind_farm.floris.farm.flow_field.mean_wind_speed = self.mean_wind_speed
		self.wind_farm.reinitialize_flow_field(wind_speed=new_wind_speed,
											   wind_direction=new_wind_dir,
											   layout_array=new_layout,
											   sim_time=self.episode_time_step)
		self.wind_farm.calculate_wake(yaw_angles=new_yaw_angles,
									  axial_induction=new_ax_ind_factors,
									  sim_time=self.episode_time_step)

		reward = self.wind_farm.get_farm_power()

		# Set `done` flag after EPISODE_LEN steps.
		self.episode_time_step += 1
		done = self.episode_time_step >= EPISODE_LEN

		# Update observation
		self.current_observation = {'layout':
										[[coords.x1 for coords in self.wind_farm.floris.farm.turbine_map.coords],
										 [coords.x2 for coords in self.wind_farm.floris.farm.turbine_map.coords]],
									'ax_ind_factors': [turbine.aI for turbine in self.wind_farm.floris.farm.turbines],
									'yaw_angles': [turbine.yaw_angle for turbine in self.wind_farm.floris.farm.turbines],
									'online_bool': new_online_bools,
		                            'wind_speed': new_wind_speed,
		                            'wind_direction': new_wind_dir}

		return self.current_observation, reward, done

	def seed(self, seed=None):
		random.seed(seed)