from FOWFEnv import FOWFEnv

if __name__ == "__main__":
    fowf_env = FOWFEnv(
        floris_input_file="./9turb_floris_input.json",
        turbine_layout_std=1.0,
        offline_probability=0.001,
    )

    init_action = {
        "yaw_angle_set": [0] * fowf_env.n_turbines,
        "ax_ind_factor_set": [0.33] * fowf_env.n_turbines,
    }

    init_disturbance = {"wind_speed": 8, "wind_dir": 270}

    fowf_env.reset(init_action, init_disturbance)
