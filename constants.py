import os

YAW_CHANGES = [-1, 0, 1]
YAW_LIMITS = [-30, 30]
AI_FACTOR_LIMITS = [0, 1/3]
IS_DYNAMIC = False
POWER_REF_PREVIEW = False

# number of historic time-steps to consider for changes in wind speed, wind dir, yaw angle, ax-ind factor, online-bool
N_PREVIOUS_TIME_STEPS = 0
	# {'yaw_angle': 600, 'ai_factor': 60} if IS_DYNAMIC else {'yaw_angle': 0, 'ai_factor': 0} # TODO will this capture enough information?
SAMPLING_TIME = {'yaw_angle': 60, 'ai_factor': 1} # interval of DT seconds at which each agent takes a step
N_PREVIEW_TIME_STEPS = 0
	# {'yaw_angle': 600, 'ai_factor': 60} if POWER_REF_PREVIEW else {'yaw_angle': 0, 'ai_factor': 0}

DT = 1.0  # discrete-time step for wind farm control
EPISODE_LEN = int(24 * 60 * 60 // DT)  # 1 day episode length
# EPISODE_LEN = int(10 * 60 // DT)  # 10 minute episode length
WIND_SPEED_RANGE = (8, 16)
WIND_DIR_RANGE = (250, 290)
# RHO = 1

# AX_IND_FACTORS = np.linspace(0.2, 0.4, 11, endpoint=True)
EPS = 0.0 # substitute for zero axial-ind factor
# YAW_CHANGES = np.array([-1, 0, 1]) # np.array([-20, -15, -10, -5, 0, 5, 10, 15, 20])
YAW_RATE = 0.5 # degrees per second
DELTA_YAW = DT * YAW_RATE
PTFM_RNG = 200  # 200 of platform relocation range

YAW_ACTUATION = True
AI_FACTOR_ACTUATION = True
PTFM_ACTUATION = False

MAX_YAW_TRAVEL_THR = 100 # 100 degrees
MAX_YAW_TRAVEL_TIME = 600 # 10 minutes

TIME_VARYING = {'power_ref': False, 'wind_speed_mean': False, 'wind_dir_mean': False, 'online': False,
                'wind_speed_turbulence': False, 'wind_dir_turbulence': False}

WORKING_DIR = os.getcwd()
ENV_CONFIG = {  # EnvSpec("wf_env", max_episode_steps=int(24*3600//DT), kwargs={
		"floris_input_file": os.path.join(WORKING_DIR, "9turb_floris_input.json"),
		"turbine_layout_std": 1,
		"offline_probability": 0.1 if TIME_VARYING['online'] else 0, # probability of any given turbine going offline at each time-step
		"wind_speed_change_probability": 0.1 if TIME_VARYING['wind_speed_mean'] else 0, # probability of wind speed/direction changing (1/2 for increase, 1/2 for decrease)
		"wind_dir_change_probability": 0.1 if TIME_VARYING['wind_dir_mean'] else 0, # probability of wind speed/direction changing (1/2 for increase, 1/2 for decrease)
		"wind_speed_var": 0.5, # step change in m/s of wind speed
		"wind_dir_var": 5, # step change in degrees of wind direction,
		"wind_speed_turb_std": 0.5 if TIME_VARYING['wind_speed_turbulence'] else 0,  # 0.5, # standard deviation of normal turbulence of wind speed, set to 0 for no turbulence
		"wind_dir_turb_std": 5 if TIME_VARYING['wind_dir_turbulence'] else 0,  # 5, # standard deviation of normal turbulence  of wind direction, set to 0 for no turbulence
		"max_yaw_travel_thr": MAX_YAW_TRAVEL_THR,
		"max_yaw_travel_time": MAX_YAW_TRAVEL_TIME,
		"max_episode_time_step": int((24 * 3600) // DT) -N_PREVIEW_TIME_STEPS, # ensure there is enough power reference preview steps left before the full 24 hour mark
		"sampling_time": SAMPLING_TIME
}

# TODO this should be a small value for power in the case of yaw, where we just want to coarsely follow the power reference, and large value for ax ind factor, where we want to follow it closesly
ALPHA = {'yaw_angle': {'power': 1e-3, 'rotor_thrust': 1e-2, 'yaw_travel': 1e-2},
         'ai_factor': {'power': 1e-1, 'rotor_thrust': 1e-2, 'yaw_travel': 0} # QUESTION MISHA does thrust depend on ai factor significantly ?
         }
WEIGHTING = {'yaw_angle': {'power': 1, 'rotor_thrust': 0, 'yaw_travel': 0*0.5},
         'ai_factor': {'power': 1, 'rotor_thrust': 0, 'yaw_travel': 0} # QUESTION MISHA does thrust depend on ai factor significantly ?
         }

TURBINE_ALPHA = {'power': 1e-3, 'rotor_thrust': 1e-2, 'yaw_travel': 1e-2}
TURBINE_WEIGHTING = {'power': 1, 'rotor_thrust': 0, 'yaw_travel': 0*0.5}

# ACTION_MAPPING = {'yaw_angle': lambda k: {0: -1, 0.5: 0, 1: 1}[k], 'ai_factor': lambda k: k * 1/3}
# ACTION_MAPPING = {'yaw_angle': lambda k: k, 'ai_factor': lambda k: (k + 1) * 1/6}
ACTION_RANGE = {'yaw_angle': YAW_LIMITS, 'ai_factor': AI_FACTOR_LIMITS}

n_turbines = 9
OBSERVATION_RANGE = {'wind_speed': WIND_SPEED_RANGE, 'wind_dir': WIND_DIR_RANGE, 'ai_factor': AI_FACTOR_LIMITS,
                      'turbine_power': (0, 5e6),'rotor_thrust': (0, 5e6), 'yaw_angle': YAW_LIMITS,
                     'yaw_travel': (0, ENV_CONFIG['max_yaw_travel_thr']), 'farm_power': (0, 5e6 * n_turbines)}