from FOWFEnv import FOWFEnvWithGroupedAgents
import os

import ray
from ray import tune
from ray.tune import register_env
from ray.rllib.algorithms.qmix import QMixConfig
import multiprocessing as mp
from ray.rllib.utils.test_utils import check_learning_achieved

DEBUG = False

RL_ALG = "QMIX"  # The RLlib-registered algorithm to use.
STOP_REWARD = 5 * 10**10  # 8.0 # Reward at which we stop training.
STOP_ITERS = 200  # Number of iterations to train.
STOP_TIMESTEPS = STOP_ITERS * 600  # Number of timesteps to train.
FRAMEWORK = "torch"  # The DL framework specifier.
MIXER = "qmix"  # The mixer model to use.

if __name__ == "__main__":

    # env_config = EnvContext(
    env_config = {
        "floris_input_file": "/scratch/alpine/jomy5901/rl-fowf-control/9turb_floris_input.json",
        "turbine_layout_std": 1,
        "offline_probability": 0.001,
    }

    register_env(
        "grouped_fowf_env",
        lambda config: FOWFEnvWithGroupedAgents(config),
    )

    ray.init(num_cpus=mp.cpu_count() or None, local_mode=DEBUG)
    # ray.init(num_cpus=1, local_mode=DEBUG)

    # Sample batches encode one or more fragments of a trajectory.
    # RLlib collects batches of size rollout_fragment_length from rollout workers,
    # and concatenates one or more of these batches into a batch of size train_batch_size that is the input to SGD.
    # In multi-agent mode, sample batches are collected separately for each individual policy. These batches are wrapped up together in a MultiAgentBatch, serving as a containe
    # Rollout workers query the policy to determine agent actions
    # In multi-agent, there may be multiple policies, each controlling one or more agents
    config = (
        QMixConfig()
        .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")))
        .framework(FRAMEWORK)
        .training(mixer=MIXER, train_batch_size=600)
        .rollouts(num_rollout_workers=0, rollout_fragment_length=4)
        .exploration(
            exploration_config={
                "final_epsilon": 0.0,
            }
        )
        .environment(
            env="grouped_fowf_env",
            env_config=env_config,
        )
    )

    # stop when trial has reached STOP_ITERS iterations
    # keys in the return result of tune.train()
    stop = {
        "episode_reward_mean": STOP_REWARD,
        "timesteps_total": STOP_TIMESTEPS,
        "training_iteration": STOP_ITERS,
    }

    # .train() - Runs one logical iteration of training.
    # .iteration - Current training iteration, value automatically incremented every time train() is called
    # The simulation iterations of action -> reward -> next state -> train -> repeat, until the end state, is called an episode, or in RLlib, a rollout.
    results = tune.run(
        "QMIX",
        config=config.to_dict(),
        stop=stop,
        keep_checkpoints_num=1,
        checkpoint_at_end=True,
    )
    import pdb

    # pdb.set_trace()
    check_learning_achieved(results, STOP_REWARD)

    ray.shutdown()
