class TurbineEnv(Env):
	# TODO agent for a single wind turbine, reward depends on power outputs of downstream wind turbines (wind-direction dependent)
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
		
		print(os.getcwd())
		self.wind_farm = wfct.floris_interface.FlorisInterface(self.floris_input_file)
		self.x_bounds = (min(self.wind_farm.layout_x), max(self.wind_farm.layout_x))
		self.y_bounds = (min(self.wind_farm.layout_y), max(self.wind_farm.layout_y))
		
		self.x_nom = self.wind_farm.layout_x
		self.y_nom = self.wind_farm.layout_y
		
		self.n_turbines = len(self.wind_farm.floris.farm.turbines)
		
		# each turbine can take take two actions: yaw angle and axial induction factor
		self.turbine_action_space = Dict({'ai_factor': Discrete(len(AX_IND_FACTORS)),
		                                  'yaw_angle': Discrete(len(
			                                  YAW_CHANGES))})  # MultiDiscrete([len(AX_IND_FACTORS), len(YAW_CHANGES)]) # Discrete(len(AX_IND_FACTORS) * len(YAW_CHANGES))
		# format of valid actions
		self._turbine_ids = set(list(range(self.n_turbines)))
		self.action_space = Dict({k: self.turbine_action_space for k in self._turbine_ids})
		
		self.state = None
		
		self._skip_env_checking = False
		
		# agent (single FOWF controller) can see all turbine locations, axial induction factors, yaw angles and online status (TODO could merge with axial induction factor)
		self.turbine_observation_space = Dict({
			'loc_mag': Box(low=0, high=PTFM_RNG, shape=()),
			'loc_dir': Box(low=0, high=2 * np.pi, shape=()),
			'ai_factor': Discrete(len(AX_IND_FACTORS)),
			'yaw_angle': Discrete(len(YAW_CHANGES)),
			'online_bool': Discrete(2)
		})
		self.observation_space = Dict(
			{
				"wind_speed": Box(low=0, high=np.infty, shape=()),
				"wind_dir": Box(low=180, high=360, shape=()),
				"turbines": Dict({k: self.turbine_observation_space for k in self._turbine_ids})
			}
		)
		
		self.mean_layout = [
			(turbine_coords.x1, turbine_coords.x2)
			for turbine_coords in self.wind_farm.floris.farm.turbine_map.coords
		]
		
		turbine_layout_std = (
			config["turbine_layout_std"] if "turbine_layout_std" in config else 1.0
		)
		self.var_layout = turbine_layout_std ** 2 * np.eye(2)  # for x and y coordinate
		
		# set wind speed/dir change probabilities and variability parameters
		self.wind_change_probability = 0.1
		self.wind_speed_var = 0.5
		self.wind_dir_var = 5.0
		
		self.mean_wind_speed = None
		self.mean_wind_dir = None
	
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
				new_dir[i] = 2 * np.pi + d
			if np.isnan(d):  # if mag == 0
				new_dir[i] = 0
		
		if any(new_dir[k] > 2 * np.pi for k in self._turbine_ids):
			print('oh no')
		
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
		return 40 * 1e6
	
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
		
		# self.current_ax_ind_factors = set_init_ax_ind_factors
		# self.current_yaw_angles = init_yaw_angles
		# self.current_loc_mag = init_mag
		# self.current_loc_dir = init_dir
		
		self.wind_farm.calculate_wake(
			yaw_angles=init_yaw_angles, axial_induction=effective_init_ax_ind_factors
		)
		
		self.episode_time_step = 0
		
		obs = self._obs(self.mean_wind_speed, self.mean_wind_dir, init_mag, init_dir, set_init_ax_ind_factors,
		                init_yaw_angles, init_online_bools)
		info = {}
		return obs, info
	
	def get_action_values(self, action_dict, online_bools):
		import pdb
		# pdb.set_trace()
		ax_ind_factor_idx = np.array(
			[int(action_dict[k]['ai_factor']) for k in self._turbine_ids]
		)
		yaw_angle_idx = np.array(
			[int(action_dict[k]['yaw_angle']) for k in self._turbine_ids]
		)
		
		yaw_angles = YAW_CHANGES[yaw_angle_idx]
		set_ax_ind_factors = AX_IND_FACTORS[ax_ind_factor_idx]
		
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
		rewards = {k: self.wind_farm.get_turbine_power()[k] for k in self._turbine_ids}
		
		# Set `done` flag after EPISODE_LEN steps.
		self.episode_time_step += 1
		dones = {"__all__": self.episode_time_step >= EPISODE_LEN}
		dones = dones["__all__"]
		reward = 0
		for idx, k in enumerate(rewards):
			reward += rewards[k] * new_online_bools[idx]
		
		# Update observation
		obs = self._obs(new_wind_speed, new_wind_dir, new_mag, new_dir, set_new_ax_ind_factors, new_yaw_angles,
		                online_bools=new_online_bools)
		
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
		
		return {
			"wind_speed": wind_speed,
			"wind_dir": wind_dir,
			"turbines": {k:
				{
					'loc_mag': loc_mag[k],
					'loc_dir': loc_dir[k],
					'ai_factor': np.where(AX_IND_FACTORS == set_ax_ind_factors[k])[0][0],
					'yaw_angle': np.where(YAW_CHANGES == yaw_angles[k])[0][0],
					'online_bool': online_bools[k]
				} for k in self._turbine_ids}
		}
	
	def seed(self, seed=None):
		random.seed(seed)

