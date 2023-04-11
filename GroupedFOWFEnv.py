class GroupedFOWFEnv(MultiAgentEnv):
	def __init__(self, env_config):
		super().__init__()
		self.env = FOWFEnv(env_config)
		# TODO hierarchical policy, can I have same turbine in multiple groups?, how to define tuple obs and act spaces?
		# grouping = {
		#     "r1c1": [1, 2, 4, 5], # bottom-left 4 turbines
		#     "r2c1": [2, 3, 4, 5], # top-left 4 turbines
		#     "r1c2": [4, 5, 7, 8], # bottom-right 4 turbines
		#     "r2c2": [5, 6, 8, 9] # top-right 4 turbines
		# }
		
		# grouping = {
		#     "farm_1": list(range(env.n_turbines)),
		# }
		# grouping = {
		#     "col_1": [0, 1, 2],
		#     "col_2": [3, 4, 5],
		#     "col_3": [6, 7, 8]
		# }
		
		tuple_obs_space = Tuple([self.env.agent_observation_space] * self.env.n_turbines)
		
		tuple_act_space = Tuple([self.env.agent_action_space] * self.env.n_turbines)
		
		# self.env = self.env.with_agent_groups(
		#     groups=grouping,
		#     obs_space=tuple_obs_space,
		#     act_space=tuple_act_space,
		# )
		# self.env = env.with_agent_groups(
		#     groups=grouping
		# )
		
		self._agent_ids = self.env._agent_ids
		self._skip_env_checking = False
		
		self.observation_space = self.env.observation_space
		self.action_space = self.env.action_space
		
		# self.observation_space = tuple_obs_space
		# self.action_space = tuple_act_space
		
		self._spaces_in_preferred_format = True
	
	def reset(self):
		# return self.env.reset()
		# Tuple of 9 elements
		#  each element is a Dict with key 'obs'
		#  which is a Dict with keys ['ax_ind_factors', 'layout_x', 'layout_y', 'online_bool', 'turbine_idx', 'yaw_angles']
		return self.env.reset()
	
	def step(self, actions):
		return self.env.step(actions)
	
	def action_space_sample(self, agent_ids: list = None) -> MultiAgentDict:
		# return {k: self.env.action_space.sample() for k in self._agent_ids}
		return self.action_space.sample()
	
	def observation_space_sample(self, agent_ids: list = None) -> MultiEnvDict:
		# sample = self.env.observation_space.sample()
		# return {
		#     k: {kk: {"obs": sample[kk]} for kk in range(sample.__len__())}
		#     for k in self._agent_ids
		# }
		return self.observation_space.sample()

