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
from ray.rllib.algorithms.qmix import QMixConfig, QMix
# from ray.rllib.policy.policy import PolicySpec
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.tune.registry import get_trainable_cls, ENV_CREATOR, _global_registry
import multiprocessing as mp
from ray.air.config import CheckpointConfig

DEBUG = True

RL_ALG = "QMIX" # The RLlib-registered algorithm to use.
STOP_REWARD = 7 #8.0 # Reward at which we stop training.
STOP_ITERS = 200 if not DEBUG else 2 # Number of iterations to train.
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
    checkpoint = CheckpointConfig(checkpoint_at_end=True, num_to_keep=1)
    TRAIN = False
    if TRAIN:
        results = tune.Tuner(
            "QMIX",
            run_config=air.RunConfig(stop=stop, checkpoint_config=checkpoint),
            param_space=config.to_dict()
        ).fit()
    
        # list of lists: one list per checkpoint; each checkpoint list contains
        # 1st the path, 2nd the metric value
        # if there are multiple trials, select a specific trial or automatically
        # choose the best one according to a given metric
        last_checkpoint = results._experiment_analysis.get_last_checkpoint(
            metric="episode_reward_mean", mode="max"
        )
        checkpoint_path = last_checkpoint._local_path

    ## TESTING
    TEST = True
    N_TESTS = 1
    if TEST:
        def run_test(agent, env_class, env_config, test_idx):
            print(f'Running test {test_idx}')
            # run until episode ends
            episode_results = {'obs': [], 'action': [], 'reward': []}
            env = env_class(env_config)
            episode_reward = 0
            done = False
            obs = env.reset()
            k = 0
            while not done:
                episode_results['obs'].append(obs)
                # compute action for farm_1 group
                action = agent.compute_single_action(obs['group_1'])
                episode_results['action'].append(action)
                obs, reward, done, info = env.step(action)
                episode_results['reward'].append(reward)
                episode_reward += reward
                print(test_idx, k, episode_reward)
                k += 1
            return episode_results
    
    
        # load and restore a trained agent from a checkpoint
        checkpoint_path = '/Users/aoifework/ray_results/QMIX/QMIX_grouped_twostep_4abd4_00000_0_2022-11-29_16-12-27/checkpoint_000002/'
        env_cls = _global_registry.get(ENV_CREATOR, "grouped_twostep")
        env_config = {}
        agent = QMix(config=config, env="grouped_twostep")
        agent.restore(checkpoint_path)
    
        if DEBUG:
            all_episode_results = []
            for test_idx in range(N_TESTS):
                all_episode_results.append(run_test(agent, env_cls, env_config, test_idx))
        else:
            pool = mp.Pool(mp.cpu_count())
            all_episode_results = pool.starmap(run_test,
                                               [(agent, env_cls, env_config, test_idx) for test_idx in
                                                range(N_TESTS)])
            pool.close()

    ray.shutdown()