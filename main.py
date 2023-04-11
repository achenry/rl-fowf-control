from WFEnv import FOWFEnvWithGroupedAgents, FOWFEnv
import os

import ray
from ray import tune, air
from ray.tune import register_env
from ray.rllib.algorithms.qmix import QMixConfig, QMix
import multiprocessing as mp
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.air.config import CheckpointConfig
from ray.rllib.algorithms.ppo import PPOConfig

import gymnasium as gym

DEBUG = True
TRAIN = True
TEST = True
N_TESTS = 1

RL_ALG = "QMIX"  # The RLlib-registered algorithm to use.
STOP_REWARD = 5 * 10**10  # 8.0 # Reward at which we stop training.
STOP_ITERS = 200 if not DEBUG else 2  # Number of iterations to train.
STOP_TIMESTEPS = STOP_ITERS * 600  # Number of timesteps to train.
FRAMEWORK = "torch"  # The DL framework specifier.
MIXER = "qmix"  # The mixer model to use.
WORKING_DIR =  "/Users/aoifework/Documents/Research/rl_fowf_control/rl-fowf-control"

if __name__ == "__main__":

    env_config = {
        "floris_input_file": os.path.join(WORKING_DIR, "9turb_floris_input.json"),
        "turbine_layout_std": 1,
        "offline_probability": 0.001,
    }
    
    # gym.register(
    register_env(
        "grouped_fowf_env",
        lambda config: FOWFEnvWithGroupedAgents(config),
    )

    register_env(
        "fowf_env",
        lambda config: FOWFEnv(config),
    )

    config = (  # 1. Configure the algorithm,
        PPOConfig()
            .environment("fowf_env")
            .rollouts(num_rollout_workers=2)
            .framework("tf2")
            .training(model={"fcnet_hiddens": [64, 64]})
            .evaluation(evaluation_num_workers=1)
    )

    algo = config.build()  # 2. build the algorithm,

    for _ in range(5):
        print(algo.train())  # 3. train it,

    algo.evaluate()  # 4. and evaluate it.
    
    # DOCS: https://docs.ray.io/en/latest/rllib/rllib-training.html

    
    # ray.init(num_cpus=1, local_mode=DEBUG)

    # Sample batches encode one or more fragments of a trajectory.
    # RLlib collects batches of size rollout_fragment_length from rollout workers,
    # and concatenates one or more of these batches into a batch of size train_batch_size that is the input to SGD.
    # In multi-agent mode, sample batches are collected separately for each individual policy. These batches are wrapped up together in a MultiAgentBatch, serving as a containe
    # Rollout workers query the policy to determine agent actions
    # In multi-agent, there may be multiple policies, each controlling one or more agents
    # TODO could optimize this config if we knew what the params meant
    # config = (
    #     QMixConfig()
    #     .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")))
    #     .framework(FRAMEWORK)
    #     .training(mixer=MIXER, train_batch_size=600, replay_buffer_config={'replay_mode': True})
    #     .multi_agent(policy_mapping_fn=lambda agent_id, episode, kwargs: "default_policy")
    #     # .rollouts(num_rollout_workers=0, rollout_fragment_length=4)
    #     # .exploration(
    #     #     exploration_config={
    #     #         "final_epsilon": 0.0,
    #     #     }
    #     # )
    #     .environment(
    #         env="grouped_fowf_env",
    #         env_config=env_config,
    #     )
    # )

    # stop when trial has reached STOP_ITERS iterations
    # keys in the return result of tune.train()
    # stop = {
    #     "episode_reward_mean": STOP_REWARD,
    #     "timesteps_total": STOP_TIMESTEPS,
    #     "training_iteration": STOP_ITERS,
    # }
    # checkpoint = CheckpointConfig(checkpoint_at_end=True, num_to_keep=1)
    #
    # ray.init(num_cpus=mp.cpu_count() or None, local_mode=DEBUG)
    
    # .train() - Runs one logical iteration of training.
    # .iteration - Current training iteration, value automatically incremented every time train() is called
    # The simulation iterations of action -> reward -> next state -> train -> repeat, until the end state, is called an episode, or in RLlib, a rollout.
    # if TRAIN:
    #     results = tune.Tuner(
    #         "QMIX",
    #         run_config=air.RunConfig(stop=stop, checkpoint_config=checkpoint),
    #         param_space=config.to_dict()
    #     ).fit()
    #
    #     # list of lists: one list per checkpoint; each checkpoint list contains
    #     # 1st the path, 2nd the metric value
    #     # if there are multiple trials, select a specific trial or automatically
    #     # choose the best one according to a given metric
    #     last_checkpoint = results._experiment_analysis.get_last_checkpoint(
    #         metric="episode_reward_mean", mode="max"
    #     )
    #     checkpoint_path = last_checkpoint._local_path
    #
    # ## TESTING
    # # in parallel, run multiple instances of FOWFEnv, each reset with different initial wind speeds and directions
    # if TEST:
    #     def run_test(agent, env_class, env_config, test_idx):
    #         print(f'Running test {test_idx}')
    #         # run until episode ends
    #         episode_results = {'obs': [], 'action': [], 'reward': []}
    #         env = env_class(env_config)
    #         episode_reward = 0
    #         done = False
    #         # TODO change this to not have farm_1 key OR change env.observation_space to have farm_1 key
    #         obs = env.reset()
    #         k = 0
    #         while not done:
    #             episode_results['obs'].append(obs)
    #             # compute action for farm_1 group
    #             action = agent.compute_action(obs)
    #             episode_results['action'].append(action)
    #             obs, reward, done, info = env.step(action)
    #             episode_results['reward'].append(reward)
    #             episode_reward += reward
    #             print(test_idx, k, episode_reward)
    #             k += 1
    #         return episode_results
    #
    #     # load and restore a trained agent from a checkpoint
    #     checkpoint_path = '/Users/aoifework/ray_results/QMIX/QMIX_grouped_fowf_env_97ac5_00000_0_2022-11-29_11-49-45/checkpoint_000001/'
    #     fowf_env = FOWFEnvWithGroupedAgents(env_config)
    #     agent = QMix(config=config, env=FOWFEnvWithGroupedAgents)
    #     agent.restore(checkpoint_path)
        
        # if DEBUG:
        #     all_episode_results = []
        #     for test_idx in range(N_TESTS):
        #         all_episode_results.append(run_test(agent, FOWFEnvWithGroupedAgents, env_config, test_idx))
        # else:
        #     pool = mp.Pool(mp.cpu_count())
        #     all_episode_results = pool.starmap(run_test, [(agent, FOWFEnvWithGroupedAgents, env_config, test_idx) for test_idx in range(N_TESTS)])
        #     pool.close()
        
    # results = tune.run(
    #     "QMIX",
    #     config=config.to_dict(),
    #     stop=stop,
    #     keep_checkpoints_num=1,
    #     checkpoint_at_end=True,
    # )
    #
    # best_trial = results.get_best_trial(mode='max', metric='episode_reward_mean')
	  # results.results['53306_00000']['info']['learner']['default_policy']
	  # list(results.fetch_trial_dataframes().values())[0].keys()
    # https://docs.ray.io/en/latest/rllib/package_ref/policy/policy.html
    
    import pdb

    # pdb.set_trace()
    # check_learning_achieved(results, STOP_REWARD)

    ray.shutdown()
