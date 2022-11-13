from FOWFEnv import FOWFEnv, RLALgo
from ray.rllib.utils.pre_checks.env import (
    check_env,
    check_gym_environments,
    check_multiagent_environments,
    check_base_env,
)

if __name__ == '__main__':
	fowf_env = FOWFEnv(
		floris_input_file="./9turb_floris_input.json",
		turbine_layout_std=1.,
		offline_probability=0.001
	)
	
	init_action = {
		'yaw_angle_set': [0] * fowf_env.n_turbines,
		'ax_ind_factor_set': [0.33] * fowf_env.n_turbines
	}
	
	init_disturbance = {
		'wind_speed': 8,
		'wind_dir': 270
	}
	
	# fowf_env.reset(options-{'init_action': init_action, 'init_disturbance': init_disturbance)
	check_env(FOWFEnv)
	
	
	rl_algo = RLALgo(FOWFEnv)
	rl_algo.configure_rl_algo()
	rl_algo.train(100)
	rl_algo.evaluate()