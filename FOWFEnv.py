import gym
from gym.spaces import MultiBinary
from ray.rllib.algorithms.qmix import QMixConfig
import numpy as np
from floridyn import tools as wfct
from gym import spaces

DT = 1.0  # discrete-time step for wind farm control
EPISODE_LEN = int(10 * 60 // DT)  # 10 minute episode length


def step_wind_field(wind_speed_mean, wind_dir_mean, wind_speed_TI, wind_dir_TI):
    ws = np.random.normal(
        loc=wind_speed_mean, scale=(wind_speed_TI / 100) * wind_speed_mean
    )[
        0
    ]  # np.random.uniform(low=8, high=8.3)
    wd = np.random.normal(loc=wind_dir_mean, scale=(wind_dir_TI / 100) * wind_dir_mean)[
        0
    ]
    return ws, wd


class FOWFEnv(gym.Env):
    def __init__(
        self,
        floris_input_file="./9turb_floris_input.json",
        turbine_layout_std=1.0,
        offline_probability=0.001,
    ):
        # TODO use gym's Discrete object for action space

        self.episode_time_step = None
        self.offline_probability = offline_probability

        self.floris_input_file = floris_input_file
        self.wind_farm = wfct.floris_interface.FlorisInterface(self.floris_input_file)
        self.n_turbines = len(self.wind_farm.floris.farm.turbines)

        self.mean_layout = [
            (turbine_coords.x1, turbine_coords.x2)
            for turbine_coords in self.wind_farm.floris.farm.turbine_map.coords
        ]
        self.var_layout = turbine_layout_std**2 * np.eye(2)  # for x and y coordinate

        self.action_space = {
            "ax_ind_factor_set": spaces.Discrete(3),
            "yaw_angle_set": spaces.Discrete(7),
        }

        self._action_space = {
            "ax_ind_factor_set": np.array([0.11, 0.22, 0.33] * self.n_turbines),
            "yaw_angle_set": np.array([-15, -10, -5, 0, 5, 10, 15] * self.n_turbines),
        }

        self.state_space = {
            "layout": spaces.Box(0, 2, shape=(2,), dtype=int),
            "ax_ind_factor_set": spaces.Discrete(3),
            "yaw_angle_set": spaces.Discrete(7),
            "online_bool": spaces.MultiBinary([3, 3]),
        }

        self._state_space = {
            "layout": [],
            "ax_ind_factors_actual": [],
            "yaw_angles_actual": [],
            "online_bool": [],
        }

        self.observation_space = self.state_space.copy()
        self._observation_space = self._state_space.copy()

        self.current_observation = {
            "layout": [],
            "ax_ind_factors": [],
            "yaw_angles": [],
            "online_bool": [],
        }

    def _get_obs(self):
        return self.current_observation

    def _get_info(self):
        return {"Total Power": self.wind_farm.get_farm_power()}

    def reset(self, init_action, init_disturbance):

        new_layout = np.vstack(
            [
                np.random.multivariate_normal(self.mean_layout[t_idx], self.var_layout)
                for t_idx in range(self.n_turbines)
            ]
        ).T

        new_online_bools = [
            np.random.choice(
                [0, 1], p=[self.offline_probability, 1 - self.offline_probability]
            )
            for t_idx in range(self.n_turbines)
        ]
        yaw_act = self._action_space["yaw_angle_set"][init_action["yaw_angle_set"]]
        ax_act = self._action_space["ax_ind_factor_set"][
            init_action["ax_ind_factor_set"]
        ]

        # initialize at steady-state
        self.wind_farm = wfct.floris_interface.FlorisInterface(self.floris_input_file)
        self.wind_farm.floris.farm.flow_field.mean_wind_speed = init_disturbance[
            "wind_speed"
        ]
        self.wind_farm.reinitialize_flow_field(
            wind_speed=init_disturbance["wind_speed"],
            wind_direction=init_disturbance["wind_dir"],
            layout_array=new_layout,
        )
        self.wind_farm.calculate_wake(
            yaw_angles=[
                a * online_bool for a, online_bool in zip(yaw_act, new_online_bools)
            ],
            axial_induction=[
                a * online_bool for a, online_bool in zip(ax_act, new_online_bools)
            ],
        )

        self.episode_time_step = 0
        xs = [t.x1 for t in self.wind_farm.floris.farm.turbine_map.coords]
        ys = [t.x2 for t in self.wind_farm.floris.farm.turbine_map.coords]
        self.current_observation = {
            "layout": np.array([xs, ys]),
            "ax_ind_factors": np.array(
                [turbine.aI for turbine in self.wind_farm.floris.farm.turbines]
            ),
            "yaw_angles": np.array(
                [turbine.yaw_angle for turbine in self.wind_farm.floris.farm.turbines]
            ),
            "online_bool": np.array(new_online_bools),
        }

        return self.current_observation

    def step(self, action, disturbance):
        """
        Given the yaw-angle and axial induction factor setting for each wind turbine (action) and the freestream wind speed (disturbance).
        Take a single step (one time-step) in the current episode
        Set the stochastic turbine (x, y) coordinates and online/offline binary variables.
        Get the effective wind speed at each wind turbine.
        Get the power output of each wind turbine and compute the overall rewards.

        """
        yaw_act = self._action_space["yaw_angle_set"][action["yaw_angle_set"]]
        ax_act = self._action_space["ax_ind_factor_set"][action["ax_ind_factor_set"]]

        # Make list of turbine x, y coordinates samples from Gaussian
        # distributions
        new_layout = np.vstack(
            [
                np.random.multivariate_normal(self.mean_layout[t_idx], self.var_layout)
                for t_idx in range(self.n_turbines)
            ]
        ).T

        # Make list of turbine online/offline booleans, offline with some small probability p,
        # if a turbine is offline, set its axial induction factor to 0
        new_online_bools = np.array(
            [
                np.random.choice(
                    [0, 1], p=[self.offline_probability, 1 - self.offline_probability]
                )
                for t_idx in range(self.n_turbines)
            ]
        )

        self.wind_farm.floris.farm.flow_field.mean_wind_speed = disturbance[
            "wind_speed"
        ]
        self.wind_farm.reinitialize_flow_field(
            wind_speed=disturbance["wind_speed"],
            wind_direction=disturbance["wind_dir"],
            layout_array=new_layout,
            sim_time=self.episode_time_step,
        )
        self.wind_farm.calculate_wake(
            yaw_angles=yaw_act,
            axial_induction=[
                a * online_bool for a, online_bool in zip(ax_act, new_online_bools)
            ],
            sim_time=self.episode_time_step,
        )

        reward = self.wind_farm.get_farm_power()

        # Set `done` flag after EPISODE_LEN steps.
        self.episode_time_step += 1
        done = self.episode_time_step >= EPISODE_LEN
        axial_induction = [
            a * online_bool for a, online_bool in zip(ax_act, new_online_bools)
        ]
        # Update observation
        self.current_observation = {
            "layout": new_layout,
            "ax_ind_factors": ax_act,
            "yaw_angles": yaw_act,
            "online_bool": new_online_bools,
        }

        return self._get_obs(), reward, done, False, self._get_info()


if __name__ == "__main__":
    fowf_env = FOWFEnv(
        floris_input_file="./9turb_floris_input.json",
        turbine_layout_std=1.0,
        offline_probability=0.001,
    )

    init_action = {
        "yaw_angle_set": [0 + (i * 7) for i in range(fowf_env.n_turbines)],
        "ax_ind_factor_set": [2 + (i * 3) for i in range(fowf_env.n_turbines)],
    }

    init_disturbance = {"wind_speed": 8, "wind_dir": 270}

    fowf_env.reset(init_action, init_disturbance)
