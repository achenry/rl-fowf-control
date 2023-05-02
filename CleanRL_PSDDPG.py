# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ddpg/#ddpg_continuous_actionpy
# sinteractive --partition=ami100 --ntasks=20 --gres=gpu:1 --time=30:00:00
# rocm-smi
# seff [job_id]
import argparse
import os
import random
import time
from distutils.util import strtobool
import sys
from collections import namedtuple, deque, defaultdict

import gym
import numpy as np
# import pybullet_envs  # noqa
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
import pickle

from TurbineEnv import MultiTurbineEnv
from constants import ENV_CONFIG, EPISODE_LEN, DT, SAMPLING_TIME, YAW_ACTUATION, AI_FACTOR_ACTUATION, LEARNING_ENDS, \
    SAVE_DIR, N_TRAINING_EPISODES, N_TESTING_EPISODES

TORCH_NUMERICAL_TYPE = torch.float32
NUMPY_NUMERICAL_TYPE = np.float32

# Transition = namedtuple('Transition',
#                         ('state', 'action', 'next_state', 'reward'))

# Fixed yaw RoC (0.3-0.5 deg/s) - add as soft constraint in objective func
# Sampling time for yaw = 1 min, axindfactor = 1 sec


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="cleanRL",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="multi_turbine_env-v0",
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=EPISODE_LEN,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=3e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--buffer-size", type=int, default=int(1e6),
        help="the replay memory buffer size")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--tau", type=float, default=0.005,
        help="target smoothing coefficient (default: 0.005)")
    parser.add_argument("--batch-size", type=int, default=256,
        help="the batch size of sample from the replay memory")
    parser.add_argument("--exploration-noise", type=float, default=0.1,
        help="the scale of exploration noise")
    parser.add_argument("--learning-starts", type=int, default=0, #25e3,
        help="timestep to start learning")
    parser.add_argument("--learning-ends", type=int, default=LEARNING_ENDS,  # 6 hours
                        help="timestep to stop learning")
    parser.add_argument("--policy-frequency", type=int, default=2, # QUESTION what is this
        help="the frequency of training policy (delayed)")
    parser.add_argument("--noise-clip", type=float, default=0.5,
        help="noise clip parameter of the Target Policy Smoothing Regularization")
    args = parser.parse_args()
    # fmt: on
    return args


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk

# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.observation_space.shape).prod() + np.prod(env.action_space.shape), 256,
                             dtype=TORCH_NUMERICAL_TYPE)
        self.fc2 = nn.Linear(256, 256, dtype=TORCH_NUMERICAL_TYPE)
        self.fc3 = nn.Linear(256, 1, dtype=TORCH_NUMERICAL_TYPE)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.observation_space.shape).prod(), 256, dtype=TORCH_NUMERICAL_TYPE)
        self.fc2 = nn.Linear(256, 256, dtype=TORCH_NUMERICAL_TYPE)
        self.fc_mu = nn.Linear(256, np.prod(env.action_space.shape), dtype=TORCH_NUMERICAL_TYPE)
        # action rescaling
        self.register_buffer(
            "action_scale", torch.tensor((env.action_space.high - env.action_space.low) / 2.0, dtype=TORCH_NUMERICAL_TYPE)
        )
        self.register_buffer(
            "action_bias", torch.tensor((env.action_space.high + env.action_space.low) / 2.0, dtype=TORCH_NUMERICAL_TYPE)
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc_mu(x))
        return x * self.action_scale + self.action_bias


if __name__ == "__main__":
    # register the environment
    gym.register(id='multi_turbine_env-v0', entry_point=MultiTurbineEnv, kwargs=ENV_CONFIG)
    
    # parse arguments
    args = parse_args()
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    
    # load power reference signal
    power_ref_preview = [30 * 1e6] * int(24 * 3600 // DT)
    
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(log_dir=os.path.join(SAVE_DIR, f"runs/{run_name}"))
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
        
    # env setup
    # env = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed, 0, args.capture_video, run_name)])
    env = make_env(args.env_id, args.seed, 0, args.capture_video, run_name)()
    # env = make_env(args.env_id, args.seed, 0, args.capture_video, run_name)
    assert isinstance(env.action_space, gym.spaces.Box), "only continuous action space is supported"

    actor = Actor(env).to(device)
    qf1 = QNetwork(env).to(device)
    qf1_target = QNetwork(env).to(device)
    target_actor = Actor(env).to(device)
    target_actor.load_state_dict(actor.state_dict())
    qf1_target.load_state_dict(qf1.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()), lr=args.learning_rate)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.learning_rate)

    env.observation_space.dtype = NUMPY_NUMERICAL_TYPE
    isinstance(env.observation_space, gym.spaces.Box)
    rb = ReplayBuffer(
        args.buffer_size,
        env.observation_space,
        env.action_space,
        device,
        handle_timeout_termination=True,
    )
    start_time = time.time()
    
    trajectory = {'episode_length': [], 'power_tracking_error': [], 'farm_power': [], 'turbine_power': [],
                  'rotor_thrust': [], 'yaw_travel': [],
                  'ai_factors': [], 'yaw_angles': [],
                  'power_tracking_reward': [], 'rotor_thrust_reward': [], 'yaw_travel_reward': [], 'total_reward': []}
    
    # TRY NOT TO MODIFY: start the game
    global_step = 0
    for episode_idx in range(N_TRAINING_EPISODES + N_TESTING_EPISODES):
        tic = time.perf_counter()
        print(f'Running Episode {episode_idx}')
        
        for metric in trajectory:
            if metric != 'episode_length':
                trajectory[metric].append([])
        
        obs, info = env.reset(options={'power_ref_preview': power_ref_preview,
                                  'sampling_time': SAMPLING_TIME})
        previous_actions = info['init_action']
        
        for local_step in range(EPISODE_LEN):
            
            if (local_step * DT) % (3600 * 1) == 0:
                print(f'{int((local_step * DT) / 3600)} hours passed in episode {episode_idx}')
            
            
            actions = defaultdict(None)
            for agent_id in env.agent_ids:
                
                # ALGO LOGIC: put action logic here
                if global_step < args.learning_starts: # TODO why have this as nonzero?
                    actions[agent_id] = env.action_space.sample()
                else:
                    with torch.no_grad():
                        actions[agent_id] = actor(torch.Tensor(obs[agent_id]).to(device))
                        # add noise if we are training, not when testing
                        # if global_step < args.learning_ends:
                        if episode_idx < N_TRAINING_EPISODES:
                            actions[agent_id] += torch.normal(0, actor.action_scale * args.exploration_noise)
                        actions[agent_id] = actions[agent_id].cpu().numpy().clip(env.action_space.low, env.action_space.high)
                        
                    # trim actions
                    # actions[agent_id] = actions[agent_id][:env.n_agent_actions[agent_id]]
                    
                # if it is not this agent's turn to go, set to last value
                actuation_idx = 0
                if YAW_ACTUATION and local_step % SAMPLING_TIME['yaw_angle'] != 0:
                    actions[agent_id][actuation_idx] = previous_actions[agent_id][actuation_idx]
                    actuation_idx += 1
                if AI_FACTOR_ACTUATION and local_step % SAMPLING_TIME['ai_factor'] != 0:
                    actions[agent_id][actuation_idx] = previous_actions[agent_id][actuation_idx]
                
            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, rewards, dones, infos = env.step(actions)
            
            trajectory['farm_power'][episode_idx].append(env.farm_power)
            trajectory['power_tracking_error'][episode_idx].append(env.power_tracking_error)
            trajectory['turbine_power'][episode_idx].append(env.turbine_power)
            trajectory['rotor_thrust'][episode_idx].append(tuple(env.rotor_thrust))
            trajectory['yaw_travel'][episode_idx].append(tuple(env.yaw_travel))
            trajectory['ai_factors'][episode_idx].append(tuple(env.eff_ai_factors))
            trajectory['yaw_angles'][episode_idx].append(tuple(env.yaw_angles))
            trajectory['power_tracking_reward'][episode_idx].append(env.reward['power_tracking'])
            trajectory['rotor_thrust_reward'][episode_idx].append(env.reward['rotor_thrust'])
            trajectory['yaw_travel_reward'][episode_idx].append(env.reward['yaw_travel'])
            trajectory['total_reward'][episode_idx].append(rewards)
            
            # TRY NOT TO MODIFY: record rewards for plotting purposes
            # for info in infos:
            if "episode" in info.keys():
                print(f"global_step={global_step}, episodic_return={infos['episode']['r']}")
                writer.add_scalar("charts/episodic_return", infos["episode"]["r"], global_step)
                writer.add_scalar("charts/episodic_length", infos["episode"]["l"], global_step)
                break
    
            # TRY NOT TO MODIFY: save data to reply buffer; handle `terminal_observation`
            for i, agent_id in enumerate(env.agent_ids):
                
                real_next_obs = next_obs[agent_id].copy()
                if dones[agent_id]:  # if done
                    real_next_obs = torch.zeros((1, env.n_observations), device=device)
                    # infos[idx]["terminal_observation"]
                real_next_obs[-1] = next_obs[agent_id][-1] # copy value of agent id
            
                rb.add(obs[agent_id], real_next_obs, actions[agent_id], rewards, dones[agent_id], infos)
    
            # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
            obs = next_obs
            previous_actions = actions
    
            # ALGO LOGIC: training.
            if global_step > args.learning_starts:
                data = rb.sample(args.batch_size)
                with torch.no_grad():
                    next_state_actions = target_actor(data.next_observations)
                    qf1_next_target = qf1_target(data.next_observations, next_state_actions)
                    next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (qf1_next_target).view(-1)
    
                qf1_a_values = qf1(data.observations, data.actions).view(-1)
                qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
    
                # optimize the model
                q_optimizer.zero_grad()
                qf1_loss.backward()
                q_optimizer.step()
    
                if global_step % args.policy_frequency == 0:
                    actor_loss = -qf1(data.observations, actor(data.observations)).mean()
                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()
    
                    # update the target network
                    for param, target_param in zip(actor.parameters(), target_actor.parameters()):
                        target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                    for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                        target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                
                if global_step % 100 == 0:
                    writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                    writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                    writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
                    print("SPS:", int(global_step / (time.time() - start_time))) # samples-per-second
                    writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
        
        # writer.add_scalar(f"trajectory/episode-{episode_idx}/episode_length", global_step + 1, global_step)
        trajectory['episode_length'].append(local_step + 1)
        toc = time.perf_counter()
        print(f'Epsiode {episode_idx} ran in {toc - tic:0.4f} seconds.')
    
    with open(os.path.join(SAVE_DIR, f'trajectory_{run_name}.pickle'), 'wb') as handle:
        pickle.dump(trajectory, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    env.close()
    writer.close()