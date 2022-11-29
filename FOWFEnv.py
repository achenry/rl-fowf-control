from ray.rllib.algorithms.qmix import QMixConfig
from ray.rllib.env.env_context import EnvContext
from ray import tune
import numpy as np
from floridyn import tools as wfct
import os
import random
from gym.spaces import Dict, MultiDiscrete, MultiBinary, Box, Discrete, Tuple
from ray.rllib.env.multi_agent_env import (
    MultiAgentEnv,
    ENV_STATE,
    MultiAgentDict,
    MultiEnvDict,
)

DT = 1.0  # discrete-time step for wind farm control
EPISODE_LEN = int(10 * 60 // DT)  # 10 minute episode length
WIND_SPEED_RANGE = (8, 16)
WIND_DIR_RANGE = (250, 290)

AX_IND_FACTORS = np.array([0.11, 0.22, 0.33])
EPS = 0.0
YAW_ANGLES = np.array([-15, -10, -5, 0, 5, 10, 15])


class FOWFEnv(MultiAgentEnv):
    def __init__(self, config: EnvContext):
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
        self.n_turbines = len(self.wind_farm.floris.farm.turbines)

        # each agent can take take two actions: yaw angle and axial induction factor
        self.agent_action_space = Discrete(len(AX_IND_FACTORS) * len(YAW_ANGLES))
        self.state = None
        self.agents = list(range(self.n_turbines))
        self._skip_env_checking = False
        self._agent_ids = set(self.agents)

        self.action_space = Dict({k: self.agent_action_space for k in self._agent_ids})

        self.agent_observation_space = Dict(
            {
                "obs": Dict(
                    {
                        "layout_x": Box(
                            low=min(
                                coord.x1
                                for coord in self.wind_farm.floris.farm.turbine_map.coords
                            ),
                            high=max(
                                coord.x1
                                for coord in self.wind_farm.floris.farm.turbine_map.coords
                            ),
                            shape=(self.n_turbines,),
                            dtype=np.float16,
                        ),
                        "layout_y": Box(
                            low=min(
                                coord.x2
                                for coord in self.wind_farm.floris.farm.turbine_map.coords
                            ),
                            high=max(
                                coord.x2
                                for coord in self.wind_farm.floris.farm.turbine_map.coords
                            ),
                            shape=(self.n_turbines,),
                            dtype=np.float16,
                        ),
                        "ax_ind_factors": MultiDiscrete(
                            [len(AX_IND_FACTORS)] * self.n_turbines
                        ),
                        "yaw_angles": MultiDiscrete(
                            [len(YAW_ANGLES)] * self.n_turbines
                        ),
                        "online_bool": MultiBinary(self.n_turbines),
                        "turbine_idx": Discrete(self.n_turbines),
                    }
                )
            }
        )

        self.observation_space = Dict(
            {k: self.agent_observation_space for k in self._agent_ids}
        )

        # self.ax_ind_factor_indices = np.arange(0, self.n_turbines)
        # self.yaw_angle_indices = np.arange(self.n_turbines, 2 * self.n_turbines)

        self.mean_layout = [
            (turbine_coords.x1, turbine_coords.x2)
            for turbine_coords in self.wind_farm.floris.farm.turbine_map.coords
        ]

        turbine_layout_std = (
            config["turbine_layout_std"] if "turbine_layout_std" in config else 1.0
        )
        self.var_layout = turbine_layout_std**2 * np.eye(2)  # for x and y coordinate

        # set wind speed/dir change probabilities and variability parameters
        self.wind_change_probability = 0.1
        self.wind_speed_var = 0.5
        self.wind_dir_var = 5.0

        self.mean_wind_speed = None
        self.mean_wind_dir = None

    def reset(self):
        self.mean_wind_speed = np.random.choice(
            np.arange(WIND_SPEED_RANGE[0], WIND_SPEED_RANGE[1], self.wind_speed_var)
        )
        self.mean_wind_dir = np.random.choice(
            np.arange(WIND_DIR_RANGE[0], WIND_DIR_RANGE[1], self.wind_dir_var)
        )

        init_action_dict = self.action_space_sample()

        new_layout = np.vstack(
            [
                np.random.multivariate_normal(self.mean_layout[t_idx], self.var_layout)
                for t_idx in range(self.n_turbines)
            ]
        ).T

        new_layout[0, :] = np.clip(
            new_layout[0, :],
            self.agent_observation_space["obs"]["layout_x"].low,
            self.agent_observation_space["obs"]["layout_x"].high,
        )
        new_layout[1, :] = np.clip(
            new_layout[1, :],
            self.agent_observation_space["obs"]["layout_y"].low,
            self.agent_observation_space["obs"]["layout_y"].high,
        )

        init_online_bools = [
            np.random.choice(
                [0, 1], p=[self.offline_probability, 1 - self.offline_probability]
            )
            for _ in range(self.n_turbines)
        ]

        # initialize at steady-state
        self.wind_farm = wfct.floris_interface.FlorisInterface(self.floris_input_file)
        self.wind_farm.floris.farm.flow_field.mean_wind_speed = self.mean_wind_speed
        self.wind_farm.reinitialize_flow_field(
            wind_speed=self.mean_wind_speed,
            wind_direction=self.mean_wind_dir,
            layout_array=new_layout,
        )

        (
            init_yaw_angles,
            set_init_ax_ind_factors,
            effective_init_ax_ind_factors,
        ) = self.get_action_values(init_action_dict, init_online_bools)
        self.current_ax_ind_factors = set_init_ax_ind_factors
        self.current_yaw_angles = init_yaw_angles

        self.wind_farm.calculate_wake(
            yaw_angles=init_yaw_angles, axial_induction=effective_init_ax_ind_factors
        )

        self.episode_time_step = 0

        obs = self._obs(init_online_bools)

        return obs

    def get_action_values(self, action_dict, online_bools):
        ax_ind_factor_idx = np.array(
            [int(action_dict[k] // len(YAW_ANGLES)) for k in self._agent_ids]
        )
        yaw_angle_idx = np.array(
            [int(action_dict[k] % len(YAW_ANGLES)) for k in self._agent_ids]
        )
        yaw_angles = YAW_ANGLES[yaw_angle_idx]
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

        new_wind_speed = self.mean_wind_speed + np.random.normal(
            scale=self.wind_speed_var
        )
        new_wind_dir = self.mean_wind_dir + np.random.normal(scale=self.wind_dir_var)

        # Make list of turbine x, y coordinates samples from Gaussian distributions
        new_layout = np.vstack(
            [
                np.random.multivariate_normal(self.mean_layout[t_idx], self.var_layout)
                for t_idx in range(self.n_turbines)
            ]
        ).T
        new_layout[0, :] = np.clip(
            new_layout[0, :],
            self.agent_observation_space["obs"]["layout_x"].low,
            self.agent_observation_space["obs"]["layout_x"].high,
        )
        new_layout[1, :] = np.clip(
            new_layout[1, :],
            self.agent_observation_space["obs"]["layout_y"].low,
            self.agent_observation_space["obs"]["layout_y"].high,
        )

        # Make list of turbine online/offline booleans, offline with some small probability p,
        # if a turbine is offline, set its axial induction factor to 0
        new_online_bools = [
            np.random.choice(
                [0, 1], p=[self.offline_probability, 1 - self.offline_probability]
            )
            for _ in range(self.n_turbines)
        ]

        (
            new_yaw_angles,
            set_new_ax_ind_factors,
            effective_new_ax_ind_factors,
        ) = self.get_action_values(action_dict, new_online_bools)
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

        try:
            assert all(
                turbine.ai_set in AX_IND_FACTORS or turbine.ai_set == 0
                for turbine in self.wind_farm.floris.farm.turbines
            ) and all(
                turbine.yaw_angle in YAW_ANGLES
                for turbine in self.wind_farm.floris.farm.turbines
            )
        except Exception as e:
            print(e)

        return {
            "obs": {
                "layout_x": np.array(
                    [
                        coords.x1
                        for coords in self.wind_farm.floris.farm.turbine_map.coords
                    ]
                ),
                "layout_y": np.array(
                    [
                        coords.x2
                        for coords in self.wind_farm.floris.farm.turbine_map.coords
                    ]
                ),
                "ax_ind_factors": np.array(
                    [
                        np.where(AX_IND_FACTORS == ai)[0][0]
                        for ai in self.current_ax_ind_factors
                    ]
                ),
                "yaw_angles": np.array(
                    [
                        np.where(YAW_ANGLES == gamma)[0][0]
                        for gamma in self.current_yaw_angles
                    ]
                ),
                "online_bool": online_bools,
                "turbine_idx": agent_idx,
            }
        }

    def seed(self, seed=None):
        random.seed(seed)


class FOWFEnvWithGroupedAgents(MultiAgentEnv):
    def __init__(self, env_config):
        super().__init__()
        env = FOWFEnv(env_config)

        grouping = {
            "farm_1": list(range(env.n_turbines)),
        }

        tuple_obs_space = Tuple([env.agent_observation_space] * env.n_turbines)

        tuple_act_space = Tuple([env.agent_action_space] * env.n_turbines)

        self.env = env.with_agent_groups(
            groups=grouping,
            obs_space=tuple_obs_space,
            act_space=tuple_act_space,
        )
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self._agent_ids = {"farm_1"}
        self._skip_env_checking = False

    def reset(self):
        return self.env.reset()

    def step(self, actions):
        return self.env.step(actions)

    def action_space_sample(self, agent_ids: list = None) -> MultiAgentDict:
        return {k: self.action_space.sample() for k in self._agent_ids}

    def observation_space_sample(self, agent_ids: list = None) -> MultiEnvDict:
        sample = self.observation_space.sample()
        return {
            k: {kk: {"obs": sample[kk]} for kk in range(sample.__len__())}
            for k in self._agent_ids
        }
