import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from floridyn import tools as wfct
from floridyn.tools.optimization.scipy.yaw import (
	YawOptimization, AiOptimization
)
from scipy.interpolate import NearestNDInterpolator
from time import perf_counter as timerpc
import pickle

DAYS_PER_YEAR = 365
HOURS_PER_DAY = 24
WATT_2_GIGAWATT = 1 / 1e9

WIND_CHANGE_PROBABILITY = 0.1
DT = 1.0  # discrete-time step for wind farm control
EPISODE_LEN = int(10 * 60 // DT)  # 10 minute episode length
WIND_SPEED_RANGE = (8, 16)
WIND_DIR_RANGE = (250, 290)
WIND_SPEED_VAR = 0.5
WIND_DIR_VAR = 5
AX_IND_FACTORS = np.array([0.11, 0.22, 0.33])
EPS = 0.0
YAW_ANGLES = np.array([-15, -10, -5, 0, 5, 10, 15])

"""
This example demonstrates how to perform a yaw optimization for multiple wind directions and multiple wind speeds.

First, we initialize our Floris Interface, and then generate a 3 turbine wind farm. Next, we create the yaw optimization object `yaw_opt` and perform the optimization using the SerialRefine method. Finally, we plot the results.
"""
FARM_LAYOUT = '9turb'
# Reinitialize as a 3-turbine farm with range of WDs and 1 WS
floris_dir = f"./{FARM_LAYOUT}_floris_input.json"


def load_windrose():
	fn = "inputs/wind_rose.csv"
	df = pd.read_csv(fn)
	df = df[(df["ws"] < 22)].reset_index(drop=True)  # Reduce size
	df["freq_val"] = df["freq_val"] / df["freq_val"].sum()  # Normalize wind rose frequencies
	
	return df


def calculate_aep(fi, df_windrose, column_name="farm_power"):
	from scipy.interpolate import NearestNDInterpolator
	
	# Define columns
	nturbs = len(fi.layout_x)
	yaw_cols = ["yaw_{:03d}".format(ti) for ti in range(nturbs)]
	ai_cols = ["ai_{:03d}".format(ti) for ti in range(nturbs)]
	
	if not "yaw_000" in df_windrose.columns:
		df_windrose[yaw_cols] = 0.0  # Add zeros
		
	if not "ai_000" in df_windrose.columns:
		df_windrose[ai_cols] = 0.33  # Add optimal
	
	# Derive the wind directions and speeds we need to evaluate in FLORIS
	wd_array = np.array(df_windrose["wd"].unique(), dtype=float)
	ws_array = np.array(df_windrose["ws"].unique(), dtype=float)
	yaw_angles = np.array(df_windrose[yaw_cols], dtype=float)
	ai_factors = np.array(df_windrose[ai_cols], dtype=float)
	
	# Map angles from dataframe onto floris wind direction/speed grid
	X, Y = np.meshgrid(wd_array, ws_array, indexing='ij')
	yaw_angle_interpolant = NearestNDInterpolator(df_windrose[["wd", "ws"]], yaw_angles)
	yaw_angles_floris = yaw_angle_interpolant(X, Y)
	ai_interpolant = NearestNDInterpolator(df_windrose[["wd", "ws"]], ai_factors)
	ai_floris = ai_interpolant(X, Y)
	farm_power_array = []
	
	# Calculate FLORIS for every WD and WS combination and get the farm power
	for wd_idx, wd in enumerate(wd_array):
		for ws_idx, ws in enumerate(ws_array):
			fi.reinitialize_flow_field(wind_direction=wd, wind_speed=ws)
			fi.calculate_wake(axial_induction=ai_floris[wd_idx, ws_idx],
			                  yaw_angles=yaw_angles_floris[wd_idx, ws_idx], sim_time=None)
			farm_power_array.append(fi.get_farm_power())
		
	# Now map FLORIS solutions to dataframe
	interpolant = NearestNDInterpolator(
		np.vstack([X.flatten(), Y.flatten()]).T,
		farm_power_array
	)
	df_windrose[column_name] = interpolant(df_windrose[["wd", "ws"]])  # Save to dataframe
	df_windrose[column_name] = df_windrose[column_name].fillna(0.0)  # Replace NaNs with 0.0
	
	# Calculate AEP in GWh
	aep = np.dot(df_windrose["freq_val"], df_windrose[column_name]) * DAYS_PER_YEAR * HOURS_PER_DAY * WATT_2_GIGAWATT
	
	return aep

class Simulator:
	def __init__(self, df_opt, floris_input_file, from_lut,
	             wind_speed_range=WIND_SPEED_RANGE, wind_speed_var=WIND_SPEED_VAR,
	            wind_dir_range=WIND_DIR_RANGE, wind_dir_var=WIND_DIR_VAR,
	             turbine_layout_std=1.0, offline_probability=0.001, wind_change_probability=0.1):
		self.wind_speed_var = wind_speed_var
		self.wind_dir_var = wind_dir_var
		self.wind_speed_range = np.arange(wind_speed_range[0], wind_speed_range[1], self.wind_speed_var)
		self.wind_dir_range = np.arange(wind_dir_range[0], wind_dir_range[1], self.wind_dir_var)
		
		self.from_lut = from_lut # if true, use optimized values in look-up table, otherwise use naive values
		
		self.offline_probability = offline_probability
		self.wind_change_probability = wind_change_probability
		
		self.floris_input_file = floris_input_file
		self.wind_farm = wfct.floris_interface.FlorisInterface(self.floris_input_file)
		self.n_turbines = len(self.wind_farm.floris.farm.turbines)
		self.yaw_cols = ["yaw_{:03d}".format(t) for t in range(self.n_turbines)]
		self.ai_cols = ["ai_{:03d}".format(t) for t in range(self.n_turbines)]
		
		self.agents = list(range(self.n_turbines))
		self._skip_env_checking = False
		self._agent_ids = set(self.agents)
		
		self.mean_layout = [
			(turbine_coords.x1, turbine_coords.x2)
			for turbine_coords in self.wind_farm.floris.farm.turbine_map.coords
		]
		
		self.df_opt = df_opt
		
		self.var_layout = turbine_layout_std ** 2 * np.eye(2)  # for x and y coordinate
		
		self.mean_wind_speed = None
		self.mean_wind_dir = None
		
		self.wind_speed_ts = None
		self.wind_dir_ts = None
		
		self.x_bounds = (min(coord.x1 for coord in self.wind_farm.floris.farm.turbine_map.coords),
		                 max(coord.x1 for coord in self.wind_farm.floris.farm.turbine_map.coords))
		
		self.y_bounds = (min(coord.x2 for coord in self.wind_farm.floris.farm.turbine_map.coords),
		                 max(coord.x2 for coord in self.wind_farm.floris.farm.turbine_map.coords))
	
	def _new_layout(self):
		new_layout = np.vstack(
			[
				np.random.multivariate_normal(self.mean_layout[t_idx], self.var_layout)
				for t_idx in range(self.n_turbines)
			]
		).T
		
		new_layout[0, :] = np.clip(
			new_layout[0, :],
			self.x_bounds[0], self.x_bounds[1]
		)
		new_layout[1, :] = np.clip(
			new_layout[1, :],
			self.y_bounds[0], self.y_bounds[1]
		)
		
		return new_layout
	
	def _new_online_bools(self):
		return [
			np.random.choice(
				[0, 1], p=[self.offline_probability, 1 - self.offline_probability]
			)
			for _ in range(self.n_turbines)
		]
	
	def reset(self, wind_speed_ts=None, wind_dir_ts=None):
		self.wind_speed_ts = wind_speed_ts
		self.wind_speed_dir = wind_dir_ts
		
		if self.wind_speed_ts is None:
			self.mean_wind_speed = np.random.choice(self.wind_speed_range)
			wind_speed = self.mean_wind_speed
		else:
			self.mean_wind_speed = wind_speed_ts[0]
			wind_speed = wind_speed_ts[0]
			
		if self.wind_dir_ts is None:
			self.mean_wind_dir = np.random.choice(self.wind_dir_range)
			wind_dir = self.mean_wind_dir
		else:
			self.mean_wind_dir = wind_dir_ts[0]
			wind_dir = wind_dir_ts[0]
			
		
		# init_action_dict = self.action_space_sample()
		
		new_layout = self._new_layout()
		
		init_online_bools = self._new_online_bools()
		
		# initialize at steady-state
		self.wind_farm = wfct.floris_interface.FlorisInterface(self.floris_input_file)
		self.wind_farm.floris.farm.flow_field.mean_wind_speed = self.mean_wind_speed
		self.wind_farm.reinitialize_flow_field(
			wind_speed=wind_speed,
			wind_direction=wind_dir,
			layout_array=new_layout,
		)
		
		if self.from_lut:
			yaw_angle_interpolant = NearestNDInterpolator(self.df_opt[["wd", "ws"]], self.df_opt[self.yaw_cols])
			ai_interpolant = NearestNDInterpolator(self.df_opt[["wd", "ws"]], self.df_opt[self.ai_cols])
			X, Y = np.meshgrid([wind_dir], [wind_speed], indexing='ij')
			yaw_angles = yaw_angle_interpolant(X, Y)[0][0]
			set_ax_ind_factors = ai_interpolant(X, Y)[0][0]
		else:
			yaw_angles = [0.0 for t in range(self.n_turbines)]
			set_ax_ind_factors = [0.33 for t in range(self.n_turbines)]
		
		(
			init_yaw_angles,
			set_init_ax_ind_factors,
			effective_init_ax_ind_factors,
		) = self.get_action_values(yaw_angles, set_ax_ind_factors, init_online_bools)
		self.current_ax_ind_factors = set_init_ax_ind_factors
		self.current_yaw_angles = init_yaw_angles
		
		self.wind_farm.calculate_wake(
			yaw_angles=init_yaw_angles, axial_induction=effective_init_ax_ind_factors
		)
		
		self.episode_time_step = 0
		
		obs = self._obs(init_online_bools)
		
		return obs
	
	def _new_wind_speed(self):
		if self.wind_speed_ts is None:
			self.mean_wind_speed = self.mean_wind_speed + np.random.choice(
				[-self.wind_speed_var, 0, self.wind_speed_var],
				p=[
					self.wind_change_probability / 2,
					1 - self.wind_change_probability,
					self.wind_change_probability / 2,
				],
			)
			self.mean_wind_speed = np.clip(
				self.mean_wind_speed, WIND_SPEED_RANGE[0], WIND_SPEED_RANGE[1]
			)
			return self.mean_wind_speed + np.random.normal(
			scale=self.wind_speed_var
			)
		else:
			return self.wind_speed_ts[self.episode_time_step]
	
	def _new_wind_dir(self):
		if self.wind_dir_ts is None:
			self.mean_wind_dir = self.mean_wind_dir + np.random.choice(
				[-self.wind_dir_var, 0, self.wind_dir_var],
				p=[
					self.wind_change_probability / 2,
					1 - self.wind_change_probability,
					self.wind_change_probability / 2,
				],
			)
			self.mean_wind_dir = np.clip(
				self.mean_wind_dir, WIND_DIR_RANGE[0], WIND_DIR_RANGE[1]
			)
			return self.mean_wind_dir + np.random.normal(scale=self.wind_dir_var)
		else:
			return self.wind_dir_ts[self.episode_time_step]
		
	def step(self):
		"""
		Given the yaw-angle and axial induction factor setting for each wind turbine (action) and the freestream wind speed (disturbance).
		Take a single step (one time-step) in the current episode
		Set the stochastic turbine (x, y) coordinates and online/offline binary variables.
		Get the effective wind speed at each wind turbine.
		Get the power output of each wind turbine and compute the overall rewards.

		"""
		
		new_wind_speed = self._new_wind_speed()
		new_wind_dir = self._new_wind_dir()
		
		# Make list of turbine x, y coordinates samples from Gaussian distributions
		new_layout = self._new_layout()
		
		# Make list of turbine online/offline booleans, offline with some small probability p,
		# if a turbine is offline, set its axial induction factor to 0
		new_online_bools = self._new_online_bools()
		
		if self.from_lut:
			yaw_angle_interpolant = NearestNDInterpolator(self.df_opt[["wd", "ws"]], self.df_opt[self.yaw_cols])
			ai_interpolant = NearestNDInterpolator(self.df_opt[["wd", "ws"]], self.df_opt[self.ai_cols])
			X, Y = np.meshgrid([new_wind_dir], [new_wind_speed], indexing='ij')
			yaw_angles = yaw_angle_interpolant(X, Y)[0][0]
			set_ax_ind_factors = ai_interpolant(X, Y)[0][0]
		else:
			yaw_angles = [0.0 for t in range(self.n_turbines)]
			set_ax_ind_factors = [0.33 for t in range(self.n_turbines)]
		
		
		(
			new_yaw_angles,
			set_new_ax_ind_factors,
			effective_new_ax_ind_factors,
		) = self.get_action_values(yaw_angles, set_ax_ind_factors, new_online_bools)
		self.current_ax_ind_factors = set_new_ax_ind_factors
		self.current_yaw_angles = new_yaw_angles
		
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
		rewards = {k: self.wind_farm.get_turbine_power()[k] for k in self._agent_ids}
		
		# Set `done` flag after EPISODE_LEN steps.
		self.episode_time_step += 1
		dones = {"__all__": self.episode_time_step >= EPISODE_LEN}
		
		# Update observation
		obs = self._obs(online_bools=new_online_bools)
		
		return obs, rewards, dones, {}
	
	def _obs(self, online_bools):
		return {k: self._agent_obs(k, online_bools) for k in self._agent_ids}
	
	def _agent_obs(self, agent_idx, online_bools):
		online_bools = np.array(online_bools)
		
		
		return {
			"obs": {
				"layout_x":
					[
						coords.x1
						for coords in self.wind_farm.floris.farm.turbine_map.coords
					]
				,
				"layout_y":
					[
						coords.x2
						for coords in self.wind_farm.floris.farm.turbine_map.coords
					]
				,
				"ax_ind_factors":
					[
						self.current_ax_ind_factors
					]
				,
				"yaw_angles":
					[
						self.current_yaw_angles
					]
				,
				"online_bool": online_bools
				# "turbine_idx": agent_idx,
			}
		}
	
	def get_action_values(self, yaw_angles, set_ax_ind_factors, online_bools):

		effective_ax_ind_factors = np.array(
			[
				a if online_bool else EPS
				for a, online_bool in zip(set_ax_ind_factors, online_bools)
			]
		)
		return yaw_angles, set_ax_ind_factors, effective_ax_ind_factors

if __name__ == "__main__":
	# Load a dataframe containing the wind rose information
	df_windrose = load_windrose()
	
	# Load FLORIS
	fi = wfct.floris_interface.FlorisInterface(floris_dir)
	fi.reinitialize_flow_field(wind_speed=8.0)
	nturbs = len(fi.layout_x)
	
	# First, get baseline AEP, without wake steering
	# start_time = timerpc()
	# print(" ")
	# print("===========================================================")
	# print("Calculating baseline annual energy production (AEP)...")
	# aep_bl = calculate_aep(fi, df_windrose, "farm_power_baseline")
	# t = timerpc() - start_time
	# print("Baseline AEP: {:.3f} GWh. Time spent: {:.1f} s.".format(aep_bl, t))
	# print("===========================================================")
	# print(" ")
	
	OPTIMIZE = False
	
	if OPTIMIZE:
		# Now optimize the yaw angles using the Serial Refine method
		print("Now starting yaw optimization for the entire wind rose...")
		start_time = timerpc()
		
		df_opt = {'wd': [], 'ws': [], 'yaw_angles_opt': [], 'ai_opt': []}
		
		for wd in np.arange(WIND_DIR_RANGE[0], WIND_DIR_RANGE[1], WIND_DIR_VAR):
			for ws in np.arange(WIND_SPEED_RANGE[0], WIND_SPEED_RANGE[1], WIND_SPEED_VAR):
				fi.reinitialize_flow_field(
					wind_direction=wd,
					wind_speed=ws
				)
				yaw_opt = YawOptimization(
					fi=fi,
					minimum_yaw_angle=0.0,  # Allowable yaw angles lower bound
					maximum_yaw_angle=20.0,  # Allowable yaw angles upper bound
					include_unc=False
				)
		
				yaw_angles_opt = yaw_opt.optimize()
				
				# ai_opt
				ai_opt = AiOptimization(
					fi=fi,
					yaw_angles=yaw_angles_opt,
					minimum_ai_factor=0,
					maximum_ai_factor=0.33,
					include_unc=False
				)
				
				ai_set_opt = ai_opt.optimize()
				
				end_time = timerpc()
				t_tot = end_time - start_time
		
				print("Optimization for {:2f} m/s, {:d} deg finished in {:.2f} seconds.".format(ws, wd, t_tot))
				print(" ")
				print(yaw_angles_opt, ai_set_opt)
				print(" ")
				
				df_opt['wd'].append(wd)
				df_opt['ws'].append(ws)
				df_opt['yaw_angles_opt'].append(yaw_angles_opt)
				df_opt['ai_opt'].append(ai_set_opt)
		
		df_opt = pd.DataFrame(df_opt)
		df_opt['freq_val'] = np.zeros_like(df_opt.index)
		
		# df_windrose.loc[1.5] = {'ws': np.nan, 'wd': np.nan, 'freq_val': np.nan}
		# for wd in df_opt['wd'].unique():
		# 	for ws in df_opt['ws'].unique():
		# 		if ws not in df_windrose['ws']:
		# 			new_idx = df_windrose['ws'] == ws -
		
		# df_windrose = df_windrose.sort_index().reset_index(drop=True)
		for row_idx, row in df_opt.iterrows():
			if row['ws'] in df_windrose['ws']:
				df_opt.loc[row_idx, 'freq_val'] = \
					df_windrose.loc[(df_windrose['ws'] == row['ws']) & (df_windrose['wd'] == row['wd']), 'freq_val'].values[0]
			else:
				idx = df_windrose['wd'] == row['wd']
				df_opt.loc[row_idx, 'freq_val'] = \
					np.interp(row['ws'], df_windrose.loc[idx, 'ws'], df_windrose.loc[idx, 'freq_val'])
		# Now define how the optimal yaw angles for 8 m/s are applied over the other wind speeds
		# yaw_angles_wind_rose = np.zeros((len(WIND_DIR_RANGE), len(WIND_SPEED_RANGE), nturbs))
		# for wd_idx, wd in enumerate(WIND_DIR_RANGE):
		# 	for ws_idx, ws in enumerate(WIND_SPEED_RANGE):
		# 		# Interpolate the optimal yaw angles for this wind direction and wind_speed from df_opt
		# 		id_opt = (df_opt["wind_direction"] == wd) & (df_opt["wind_speed"] == ws)
		# 		yaw_opt_full = np.array(df_opt.loc[id_opt, "yaw_angles_opt"])[0]
		# 		yaw_angles_wind_rose[wd_idx, ws_idx, :] = yaw_opt_full
				
				# Now decide what to do for different wind speeds
				# if (wind_speed < 4.0) | (wind_speed > 14.0):
				# 	yaw_opt = np.zeros(nturbs)  # do nothing for very low/high speeds
				# elif wind_speed < 6.0:
				# 	yaw_opt = yaw_opt_full * (6.0 - wind_speed) / 2.0  # Linear ramp up
				# elif wind_speed > 12.0:
				# 	yaw_opt = yaw_opt_full * (14.0 - wind_speed) / 2.0  # Linear ramp down
				# else:
				# 	yaw_opt = yaw_opt_full  # Apply full offsets between 6.0 and 12.0 m/s
				
				# Save to collective array
				# yaw_angles_wind_rose[ii, :] = yaw_opt
		
		# Add optimal and interpolated angles to the wind rose dataframe
		yaw_cols = ["yaw_{:03d}".format(ti) for ti in range(nturbs)]
		df_opt[yaw_cols] = np.vstack(df_opt['yaw_angles_opt'].values)
		ai_cols = ["ai_{:03d}".format(ti) for ti in range(nturbs)]
		df_opt[ai_cols] = np.vstack(df_opt['ai_opt'].values)
		# df_windrose[yaw_cols] = yaw_angles_wind_rose
		
		df_opt.to_csv('./df_opt.csv')
	else:
		df_opt = pd.read_csv('./df_opt.csv')
	
	# simulate for ten minutes with optimized values
	simulator_opt = Simulator(df_opt=df_opt, floris_input_file=floris_dir, from_lut=True)
	init_wind_speed = 12
	init_wind_dir = 270
	wind_speed_ts = [init_wind_speed]
	wind_dir_ts = [init_wind_dir]
	simulator_opt.mean_wind_speed = init_wind_speed
	simulator_opt.mean_wind_dir = init_wind_dir
	for k in range(1, EPISODE_LEN):
		wind_speed_ts.append(simulator_opt._new_wind_speed())
		wind_dir_ts.append(simulator_opt._new_wind_dir())
	
	simulator_opt.reset(wind_speed_ts=wind_speed_ts, wind_dir_ts=wind_dir_ts)
	episode_rewards_opt = []
	for k in range(EPISODE_LEN):
		obs, rewards, dones, _ = simulator_opt.step()
		episode_rewards_opt.append(list(rewards.values()))
	
	episode_rewards_opt_farm = np.sum(episode_rewards_opt, axis=1)
	mean_episode_reward_opt = np.mean(episode_rewards_opt_farm)
	
	# simulate for ten minutes with naive values
	simulator_baseline = Simulator(df_opt=df_opt, floris_input_file=floris_dir, from_lut=False)
	simulator_baseline.reset(wind_speed_ts=wind_speed_ts, wind_dir_ts=wind_dir_ts)
	episode_rewards_baseline = []
	for k in range(EPISODE_LEN):
		obs, rewards, dones, _ = simulator_baseline.step()
		episode_rewards_baseline.append(list(rewards.values()))
	
	episode_rewards_baseline_farm = np.sum(episode_rewards_baseline, axis=1)
	mean_episode_reward_baseline = np.mean(episode_rewards_baseline_farm)
	
	# Now calculate helpful variables and then plot wind rose information
	df = df_windrose.copy()
	df["farm_power_relative"] = (
		df["farm_power_opt"] / df["farm_power_baseline"]
	)
	df["farm_energy_baseline"] = df["freq_val"] * df["farm_power_baseline"]
	df["farm_energy_opt"] = df["freq_val"] * df["farm_power_opt"]
	df["energy_uplift"] = df["farm_energy_opt"] - df["farm_energy_baseline"]
	df["rel_energy_uplift"] = df["energy_uplift"] / df["energy_uplift"].sum()

	# Now get AEP with optimized yaw angles
	# start_time = timerpc()
	# print("==================================================================")
	# print("Calculating annual energy production (AEP) with wake steering...")
	# aep_opt = calculate_aep(fi, df_opt, "farm_power_opt")
	# aep_uplift = 100.0 * (aep_opt / aep_bl - 1)
	# t = timerpc() - start_time
	# print("Optimal AEP: {:.3f} GWh. Time spent: {:.1f} s.".format(aep_opt, t))
	# print("Relative AEP uplift by wake steering: {:.3f} %.".format(aep_uplift))
	# print("==================================================================")
	# print(" ")
	
	# Plot power and AEP uplift across wind direction
	# fig, ax = plt.subplots(nrows=3, sharex=True)
	#
	# df_8ms = df[df["ws"] == 8.0].reset_index(drop=True)
	# pow_uplift = 100 * (
	# 	df_8ms["farm_power_opt"] / df_8ms["farm_power_baseline"] - 1
	# )
	# ax[0].bar(
	# 	x=df_8ms["wd"],
	# 	height=pow_uplift,
	# 	color="darkgray",
	# 	edgecolor="black",
	# 	width=4.5,
	# )
	# ax[0].set_ylabel("Power uplift \n at 8 m/s (%)")
	# ax[0].grid(True)
	#
	# dist = df.groupby("wd").sum().reset_index()
	# ax[1].bar(
	# 	x=dist["wd"],
	# 	height=100 * dist["rel_energy_uplift"],
	# 	color="darkgray",
	# 	edgecolor="black",
	# 	width=4.5,
	# )
	# ax[1].set_ylabel("Contribution to \n AEP uplift (%)")
	# ax[1].grid(True)
	#
	# ax[2].bar(
	# 	x=dist["wd"],
	# 	height=dist["freq_val"],
	# 	color="darkgray",
	# 	edgecolor="black",
	# 	width=4.5,
	# )
	# ax[2].set_xlabel("Wind direction (deg)")
	# ax[2].set_ylabel("Frequency of \n occurrence (-)")
	# ax[2].grid(True)
	# plt.tight_layout()
	#
	# # Plot power and AEP uplift across wind direction
	# fig, ax = plt.subplots(nrows=3, sharex=True)
	#
	# df_avg = df.groupby("ws").mean().reset_index(drop=False)
	# mean_power_uplift = 100.0 * (df_avg["farm_power_relative"] - 1.0)
	# ax[0].bar(
	# 	x=df_avg["ws"],
	# 	height=mean_power_uplift,
	# 	color="darkgray",
	# 	edgecolor="black",
	# 	width=0.95,
	# )
	# ax[0].set_ylabel("Mean power \n uplift (%)")
	# ax[0].grid(True)
	#
	# dist = df.groupby("ws").sum().reset_index()
	# ax[1].bar(
	# 	x=dist["ws"],
	# 	height=100 * dist["rel_energy_uplift"],
	# 	color="darkgray",
	# 	edgecolor="black",
	# 	width=0.95,
	# )
	# ax[1].set_ylabel("Contribution to \n AEP uplift (%)")
	# ax[1].grid(True)
	#
	# ax[2].bar(
	# 	x=dist["ws"],
	# 	height=dist["freq_val"],
	# 	color="darkgray",
	# 	edgecolor="black",
	# 	width=0.95,
	# )
	# ax[2].set_xlabel("Wind speed (m/s)")
	# ax[2].set_ylabel("Frequency of \n occurrence (-)")
	# ax[2].grid(True)
	# plt.tight_layout()
	#
	# # Now plot yaw angle distributions over wind direction up to first three turbines
	# for ti in range(np.min([nturbs, 3])):
	# 	fig, ax = plt.subplots(figsize=(6, 3.5))
	# 	ax.plot(
	# 		df_opt["wind_direction"],
	# 		yaw_angles_opt[:, ti],
	# 		"-o",
	# 		color="maroon",
	# 		markersize=3,
	# 		label="For wind speeds between 6 and 12 m/s",
	# 	)
	# 	ax.plot(
	# 		df_opt["wind_direction"],
	# 		0.5 * yaw_angles_opt[:, ti],
	# 		"-v",
	# 		color="dodgerblue",
	# 		markersize=3,
	# 		label="For wind speeds of 5 and 13 m/s",
	# 	)
	# 	ax.plot(
	# 		df_opt["wind_direction"],
	# 		0.0 * yaw_angles_opt[:, ti],
	# 		"-o",
	# 		color="grey",
	# 		markersize=3,
	# 		label="For wind speeds below 4 and above 14 m/s",
	# 	)
	# 	ax.set_ylabel("Assigned yaw offsets (deg)")
	# 	ax.set_xlabel("Wind direction (deg)")
	# 	ax.set_title("Turbine {:d}".format(ti))
	# 	ax.grid(True)
	# 	ax.legend()
	# 	plt.tight_layout()
	#
	# plt.show()