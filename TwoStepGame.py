"""The two-step game from QMIX: https://arxiv.org/pdf/1803.11485.pdf
Configurations you can try:
    - normal policy gradients (PG)
    - MADDPG
    - QMIX
See also: centralized_critic.py for centralized critic PPO on this game.
"""

from gym.spaces import Dict, Tuple, MultiDiscrete
import os

import ray
from ray import air, tune
from ray.tune import register_env
from ray.rllib.env.multi_agent_env import ENV_STATE
from ray.rllib.examples.env.two_step_game import TwoStepGame
from ray.rllib.algorithms.qmix import QMixConfig
# from ray.rllib.policy.policy import PolicySpec
from ray.rllib.utils.test_utils import check_learning_achieved
# from ray.tune.registry import get_trainable_cls
import multiprocessing as mp

DEBUG = True

RL_ALG = "QMIX" # The RLlib-registered algorithm to use.
STOP_REWARD = 7 #8.0 # Reward at which we stop training.
STOP_ITERS = 200 # Number of iterations to train.
STOP_TIMESTEPS = 70000 # Number of timesteps to train.
FRAMEWORK = "torch" # The DL framework specifier.
MIXER = "qmix" # The mixer model to use.

if __name__ == "__main__":
    
    ray.init(num_cpus=mp.cpu_count() or None, local_mode=DEBUG)

    grouping = {
        "group_1": [0, 1],
    }
    obs_space = Tuple(
        [
            Dict(
                {
                    "obs": MultiDiscrete([2, 2, 2, 3]),
                    ENV_STATE: MultiDiscrete([2, 2, 2]),
                }
            ),
            Dict(
                {
                    "obs": MultiDiscrete([2, 2, 2, 3]),
                    ENV_STATE: MultiDiscrete([2, 2, 2]),
                }
            ),
        ]
    )
    act_space = Tuple(
        [
            TwoStepGame.action_space,
            TwoStepGame.action_space,
        ]
    )
    
    register_env(
        "grouped_twostep",
        lambda config: TwoStepGame(config).with_agent_groups(
            grouping, obs_space=obs_space, act_space=act_space
        ),
    )

    config = (
        QMixConfig()
        # .environment(env=TwoStepGame)
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        
    )

    
    config = (
        QMixConfig()
            .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")))
            .framework(FRAMEWORK)
            .training(mixer=MIXER, train_batch_size=32)
            .rollouts(num_rollout_workers=0, rollout_fragment_length=4)
            .exploration(
                exploration_config={
                    "final_epsilon": 0.0,
                }
            )
            .environment(
                env="grouped_twostep",
                env_config={
                    "separate_state_space": True,
                    "one_hot_state_encoding": True,
                },
            )
    )

    stop = {
        "episode_reward_mean": STOP_REWARD,
        "timesteps_total": STOP_TIMESTEPS,
        "training_iteration": STOP_ITERS,
    }
    results = tune.run('QMIX', config=config.to_dict())
    
    # results = tune.Tuner(
    #     RL_ALG,
    #     run_config=air.RunConfig(stop=stop, verbose=2),
    #     param_space=config,
    # ).fit()

    if DEBUG:
        check_learning_achieved(results, STOP_REWARD)

    ray.shutdown()