from gym.spaces import Box
import numpy as np
from constants import N_PREVIOUS_TIME_STEPS, N_PREVIEW_TIME_STEPS, YAW_ACTUATION, AI_FACTOR_ACTUATION



class MultiTurbineAgent:
	def __init__(self, n_turbines):
		self.n_turbines = n_turbines
		self._turbine_ids = set(list(range(self.n_turbines)))
		
		self.action_space = None
		self.reward = None
		self.n_observations = ((N_PREVIEW_TIME_STEPS + 1) +  # power_ref_preview
		                       self.n_turbines * (
			                       (N_PREVIOUS_TIME_STEPS + 1)  # online-bool
			                       # + 1 # power
			                       + (N_PREVIOUS_TIME_STEPS + 1)  # wind_speed
			                       + (N_PREVIOUS_TIME_STEPS + 1)  # wind_dir
			                       + (N_PREVIOUS_TIME_STEPS + 1)  # yaw_angle
			                       # + 1 # yaw_travel
			                       + (N_PREVIOUS_TIME_STEPS + 1)  # ai_factor
			                       # + 1 # rotor_thrust
		                       ) + 1) # agent id)
		self.observation_space = Box(low=0, high=1, shape=(self.n_observations,), dtype=np.float16)
		# isinstance(self.observation_space, Box)
		
		self.n_actions = (int(YAW_ACTUATION) + int(AI_FACTOR_ACTUATION))
		
		self.action_space = Box(low=0, high=1, shape=(self.n_actions,), dtype=np.float16)
	
	def step(self, action):
		pass
	
	def reset(self):
		pass
	
	def _obs(self):
		pass
	
class Agent:
	def __init__(self, n_turbines):
		self.n_turbines = n_turbines
		self._turbine_ids = set(list(range(self.n_turbines)))
		
		self.action_space = None
		self.reward = None
		
		# turbine_observation_space = {
		# 	'online_bool': Tuple((Discrete(2, start=0) for _ in range(N_PREVIOUS_TIME_STEPS + 1))),
		# 	'power': Box(low=0, high=np.infty, shape=()),
		# 	'wind_speed': Tuple((Box(low=0, high=np.infty, shape=()) for _ in range(N_PREVIOUS_TIME_STEPS + 1))),
		# 	'wind_dir': Tuple((Box(low=180, high=360, shape=(), dtype=int) for _ in range(N_PREVIOUS_TIME_STEPS + 1))),
		# 	'yaw_angle': Tuple((Box(low=YAW_LIMITS[0], high=YAW_LIMITS[1], shape=()) for _ in range(N_PREVIOUS_TIME_STEPS + 1))),
		# 	'yaw_travel': Box(low=0, high=np.infty, shape=(), dtype=int),
		# 	'ai_factor': Tuple((Box(low=AI_FACTOR_LIMITS[0], high=AI_FACTOR_LIMITS[1], shape=()) for _ in range(N_PREVIOUS_TIME_STEPS + 1))),
		# 	'rotor_thrust': Box(low=0, high=np.infty, shape=())
		# 	}
		#
		# turbine_observation_space = Dict(turbine_observation_space)
		# self.observation_space = Dict(
		# 	{
		# 		"power_ref_preview": Tuple((Box(low=0, high=np.infty, shape=())) for _ in range(N_PREVIEW_TIME_STEPS + 1)),
		# 		"turbines": Dict({str(k): turbine_observation_space for k in self._turbine_ids})
		# 	}
		# )
		self.n_observations = ((N_PREVIEW_TIME_STEPS + 1) + # power_ref_preview
			self.n_turbines * (
				(N_PREVIOUS_TIME_STEPS + 1) # online-bool
				# + 1 # power
				+ (N_PREVIOUS_TIME_STEPS + 1) # wind_speed
				+ (N_PREVIOUS_TIME_STEPS + 1) # wind_dir
				+ (N_PREVIOUS_TIME_STEPS + 1) # yaw_angle
				# + 1 # yaw_travel
				+ (N_PREVIOUS_TIME_STEPS + 1) # ai_factor
				# + 1 # rotor_thrust
			))
		self.observation_space = Box(low=0, high=1, shape=(self.n_observations, ), dtype=np.float16)
		
		self.n_actions = self.n_turbines
		self.action_space = Box(low=0, high=1, shape=(self.n_actions,), dtype=np.float16)
	
	def step(self, action):
		pass
	
	def reset(self):
		pass
	
	def _obs(self):
		pass


class YawAgent(Agent):
	def __init__(self, n_turbines):
		super().__init__(n_turbines, 'yaw_angle')
		
		self.action_space = Box(low=0, high=1, shape=(n_turbines, ), dtype=np.float16)
		
		# format of valid actions
		# self.action_space = Dict({str(k): turbine_action_space for k in self._turbine_ids})
		self.n_actions = self.n_turbines
		

class AxIndFactorAgent(Agent):
	def __init__(self, n_turbines):
		super().__init__(n_turbines, 'ai_factor')
		
		self.action_space = Box(low=0, high=1, shape=(n_turbines, ), dtype=np.float16)
		
		# format of valid actions
		self.n_turbines = n_turbines
		self._turbine_ids = set(list(range(self.n_turbines)))
		
		self.n_actions = self.n_turbines

# if PTFM_ACTUATION:
# 	turbine_action_space = {**turbine_action_space, **{'loc_mag': Box(low=0, high=PTFM_RNG, shape=()),
# 	                                                   'loc_dir': Box(low=0, high=2 * np.pi, shape=())}}
# if PTFM_ACTUATION:
# 	turbine_observation_space = {**turbine_observation_space, **{
# 		'loc_mag': Box(low=0, high=PTFM_RNG, shape=()),
# 		'loc_dir': Box(low=0, high=2 * np.pi, shape=())}}