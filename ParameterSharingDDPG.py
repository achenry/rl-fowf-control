# TODO
# develop DDPG algorithm with pytorch for Yaw Control X
# create multiagent WFEnv with PettingZoo
# normalize action space and observation space to be between 0 and 1 OR keep running mean of variables and standardize
# Read DDPG Paper X, Parameter Sharing Paper
# implement parameter sharing with pytorch for Yaw and Axial Induction Factor Control with observation/control space padding and shared Actor Network



# GPUs:
# 3 GPU devices (nvidea a100s) + amd CPU with 64 CPU cores on each node => can request up to all or a third ie 21
# code will run on CPU until it reaches GPU part, data is copied to GPU, then copied back
# code needs to tell system when to shift to GPU and back

# implement mini-batch normalization X
# TODO QUESTION whether to generate exploration policy by adding noise or with epsilon greedy approach?
# checkpoint X
# store training trajectory X
# write test loop X
# write plot code X
# TODO run on RC single GPU
# TODO make baseline yaw controller



import math
import random
from collections.abc import MutableMapping, Sequence
from collections import defaultdict
from itertools import count
import numpy as np
from results import plot_cumulative_reward, plot_tracking_errors

from gymnasium.spaces import Dict, Tuple

from collections import namedtuple, deque
import copy
import sys
import os

import torch
import torch.nn as nn
import torch.optim as optim
# import torch.nn.functional as F

from WFEnv import WFEnv
from constants import ENV_CONFIG, DT, SAMPLING_TIME

#https://openreview.net/forum?id=MWj_P-Lk3jC
#https://arshren.medium.com/step-by-step-guide-to-implementing-ddpg-reinforcement-learning-in-pytorch-9732f42faac9

# DEFINE CONSTANTS
BATCH_SIZE = 128 # number of transitions sampled from the replay buffer
GAMMA = 0.99 # discount factor
EPS_START = 0.9 # starting value of epsilong in greedy policy action selection
EPS_END = 0.05 # final value of epsilong in greedy policy action selection
EPS_DECAY = 1000 # rate of exponential decay in greedy policy action selection, high for slow decay
TAU = 0.005 # update rate of the target network
LR = 1e-4 # learning rate of the AdamW optimizer
REPLAY_SIZE = int(1e4) # capacity of replay memory

if sys.platform == 'darwin':
    project_dir = '/Users/aoifework/Documents/Research/rl_fowf_control/rl-fowf-control/'
elif sys.platform == 'linux':
    project_dir = f'/scratch/alpine/aohe7145/rl_wf_control/'

SAVE_DIR = os.path.join(project_dir, 'checkpoints')
FIG_DIR = os.path.join(project_dir, 'figs')

for dir in [SAVE_DIR, FIG_DIR]:
    if not os.path.exists(dir):
        os.makedirs(dir)

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


# code to convert ini_dict to flattened dictionary
# default separator '_'
def convert_flatten(d, sep='_'):
    items = []
    if isinstance(d, MutableMapping):
        d = d.values()
    for v in d:
        
        if isinstance(v, MutableMapping):
            # if this is a dictionary
            items.extend(convert_flatten(v, sep=sep))
        elif isinstance(v, Sequence):
            # if this is a list or a tuple
            items.extend(convert_flatten(v, sep=sep))
        else:
            # items.append((new_key, v))
            items.append(v)
    return items


def convert_unflatten(d_template, arr):
    d_unf = {}
    for i, (k, v) in enumerate(d_template.items()):
        if isinstance(v, Dict):
            d_unf[k] = convert_unflatten(v, arr)
        elif isinstance(v, Tuple):
            d_unf[k] = convert_unflatten(v, arr)
        else:
            d_unf[k] = arr.pop()
    return d_unf


class ReplayMemory(object):
    """ A cyclic buffer of bounded size that holds the transitions observed recently."""
    
    def __init__(self, capacity):
        # A list-like sequence optimized for data accesses near its endpoints.
        self.memory = deque([], maxlen=capacity)
    
    def push(self, *args):
        """ Save a transition """
        self.memory.append(Transition(*args))
    
    def sample(self, batch_size):
        """ Select a random batch of transitions for training"""
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)


class Actor(nn.Module):
    """
        The Actor model takes in a state observation as input and
        outputs an action, which is a continuous value.

        It consists of three fully connected linear layers with ReLU activation functions and
        a final output layer selects optimized actions for the state
        """
    
    def __init__(self, n_observations, n_actions, n_hidden_nodes):
        super(Actor, self).__init__()
        self.layer1 = nn.Linear(n_observations, n_hidden_nodes)  # first layer reads observed states, outputs 128 nodes
        self.layer2 = nn.Linear(n_hidden_nodes, n_hidden_nodes)  # middle layer reads previous layer of 128 nodes, outputs 128 nodes
        self.layer3 = nn.Linear(n_hidden_nodes, n_hidden_nodes)  # middle layer reads previous layer of 128 nodes, outputs 128 nodes
        self.layer4 = nn.Linear(n_hidden_nodes, n_actions)  # final layer reads previous 128 nodes, outputs control actions
        
        self.net = nn.Sequential(
            self.layer1,
            nn.ReLU(),
            self.layer2,
            nn.ReLU(),
            self.layer3,
            nn.ReLU(),
            self.layer4
        )
    
    def forward(self, state):
        """ Feed Forward neural ne. Called with either one element to determine next action, or a batch during optimization. Returns tensor([[ctrl_Action1, ctrl_action2, ...]])"""
        return self.net(state)


class Critic(nn.Module):
    """
    The Critic model takes in both a state observation and an action as input and
    outputs a Q-value, which estimates the expected total reward for the current state-action pair.

    It consists of four linear layers with ReLU activation functions,
    State and action inputs are concatenated before being fed into the first linear layer.

    The output layer has a single output, representing the Q-value
    """
    
    def __init__(self, n_observations, n_actions, n_hidden_nodes=128):
        super(Critic, self).__init__()
        self.layer1 = nn.Linear(n_observations + n_actions, n_hidden_nodes)  # first layer reads observed states, outputs 128 nodes
        self.layer2 = nn.Linear(n_hidden_nodes, n_hidden_nodes)  # middle layer reads previous layer of 128 nodes, outputs 128 nodes
        self.layer3 = nn.Linear(n_hidden_nodes, n_hidden_nodes)  # middle layer reads previous layer of 128 nodes, outputs 128 nodes
        self.layer4 = nn.Linear(n_hidden_nodes, n_actions)  # final layer reads previous 128 nodes, outputs control actions actions
        
        self.net = nn.Sequential(
            self.layer1,
            nn.ReLU(),
            self.layer2,
            nn.ReLU(),
            self.layer3,
            nn.ReLU(),
            self.layer4
        )
    
    def forward(self, state, action):
        return self.net(torch.cat((state, action), 1))


class OU_Noise(object):
    """Ornstein-Uhlenbeck process.
    code from :
    https://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
    The OU_Noise class has four attributes

        size: the size of the noise vector to be generated
        mu: the mean of the noise, set to 0 by default
        theta: the rate of mean reversion, controlling how quickly the noise returns to the mean
        sigma: the volatility of the noise, controlling the magnitude of fluctuations
    """
    
    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()
    
    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)
    
    def sample(self):
        """Update internal state and return it as a noise sample.
        This method uses the current state of the noise and generates the next sample
        """
        dx = self.theta * (self.mu - self.state) + self.sigma * np.array(
            [np.random.normal() for _ in range(len(self.state))])
        self.state += dx
        return self.state


class PSDDPG(object):
    def __init__(self, env, device, memory, n_hidden_nodes_actor=128, n_hidden_nodes_critic=128):
        """
        Initializes the DDPG agent.
        Takes three arguments:
               n_observations which is the dimensionality of the state space,
               n_actions which is the dimensionality of the action space, and
               max_action which is the maximum value an action can take.

        Creates a replay buffer, an actor-critic  networks and their corresponding target networks.
        It also initializes the optimizer for both actor and critic networks alog with
        counters to track the number of training iterations.
        """
        self.steps_done = 0
        self.steps_done_n = {agent_id: 0 for agent_id in env.agent_ids}
        self.device = device
        self.env = env
        self.memory = memory
        self.episode_durations = []
        
        # # find agent with
        # max_n_actions_i = np.argmax(self.env.n_actions.values())

        # for Parameter Sharing, we consider the greatest n_observation quantities over all agents
        self.max_n_observations = max(self.env.n_observations.values())
        
        # applied to the yaw agent, only first three elements are meaningful, applied to axial induction factor agent, more
        # action space of n_turbines, for yaw of type discrete, for axial induction factor of type continuous
        # self.max_action_space = Dict({agent_id: self.env.action_space[agent_id] for agent_id in self.env.agent_ids})
        # for Parameter Sharing, we consider the greatest n_actions discrete action possibilities
        # self.max_n_actions = max(self.env.n_actions.values())
        self.max_n_actions = max(self.env.n_actions.values())
        self.action_type = {agent_id: getattr(torch, agent.action_space.dtype.name) for agent_id, agent in self.env.agents.items()}
        # self.max_action_space = None

        # self.state_mean = {agent_id: np.zeros((self.env.n_observations[agent_id],)) for agent_ids in self.env.agent_ids}
        # self.state_std = {agent_id: np.zeros((self.env.n_observations[agent_id])) for agent_ids in self.env.agent_ids}
        # , add 1 to include agent indicator variable
        self.state_mean = torch.zeros((self.max_n_observations + 1,), device=self.device) # np.zeros((self.max_n_observations + 1,))
        self.state_std = torch.zeros((self.max_n_observations + 1,), device=self.device) # np.zeros((self.max_n_observations + 1,))

        # Shared actor to learn single policy for all agents, add 1 to include agent indicator variable
        self.actor = Actor(self.max_n_observations + 1, self.max_n_actions, n_hidden_nodes_actor).to(self.device)
        self.actor_target = Actor(self.max_n_observations + 1, self.max_n_actions, n_hidden_nodes_actor).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LR)
        
        # Shared critic to learn Q(s, a) for all agents, add 1 to include agent indicator variable
        self.critic = Critic(self.max_n_observations + 1, self.max_n_actions, n_hidden_nodes_critic).to(self.device)
        self.critic_target = Critic(self.max_n_observations + 1, self.max_n_actions, n_hidden_nodes_critic).to(self.device)
        self.critic_optimizer = {}
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=LR)
        
    def select_action(self, state, agent_id, greedy=False):
        """
        takes the current state as input and returns an action to take in that state.
        It uses the actor network to map the state to an action.
        """
        # global steps_done
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1 * self.steps_done_n[agent_id] / EPS_DECAY)
        # self.steps_done += 1
        if (sample > eps_threshold) or greedy:
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward (greedy).
                # print(f'policy_net(state).shape = {policy_net(state).shape}')
                # TODO QUESTION also normalize agent indication function and padded zeros?
                # state[agent_id][:, :-1] \
                #     = state[agent_id][:, :-1] \
                #       - self.state_mean[:, :-1]
                # self.state_std[self.state_std == 0] = 1
                # state[agent_id][:, :-1] = state[agent_id][:, :-1] / self.state_std[:, :-1]
                action = self.actor(state[agent_id])
                action = torch.clip(action, -1., 1.)
                return action
        else:
            # take random sample of actions if we have not exceeded the eps_threshold for greedy action selection
            action_sample = self.env.action_space(agent_id).sample()
            # convert from gym.Spaces object to list of actions and pad if necessary
            action_sample = [list(convert_flatten(action_sample)) + ([0] * (self.max_n_actions - self.env.n_actions[agent_id]))]
            # print(f'[env.action_space.sample()] = {action_sample}')
            
            return torch.tensor(action_sample, device=self.device, dtype=self.action_type[agent_id])

    def optimize_model(self):
    
        # fetch batch of samples from replay buffer if it is large enough
        if len(self.memory) < BATCH_SIZE:
            return
        transitions = self.memory.sample(BATCH_SIZE)
    
        # transpose the batch (https://stackoverflow.com/a/19343/3343043) to convert batch-arry of Transitions to Transition of batch arrays
        batch = Transition(*zip(*transitions))
        # print('len(batch.state)', len(batch.state))
        # print('batch.state[0].shape', batch.state[0].shape)
        # print('batch.action[0].shape', batch.action[0].shape)
        # print('batch.next_state[0].shape', batch.next_state[0].shape)

        # TODO QUESTION what does this mean:
        #  "we used batch normalization on the state input and all layers of the μ network and all layers of the Q network prior to the action input"
        
        # normalize the next-state variables
        # exclude agent indication variable from standardization
        # means_batch = self.state_mean.repeat(BATCH_SIZE, 1)
        # stds_batch = self.state_std.repeat(BATCH_SIZE, 1)

        
        # for i in range(len(next_state_batch)):
        #     next_state_batch[i][:, :-1] = next_state_batch[i][:, :-1] - self.state_mean[:, :-1]
        #     next_state_batch[i][:, :-1] = next_state_batch[i][:, :-1] / self.state_std[:, :-1]
        # next_state_batch = tuple(next_state_batch)
    
        # normalize the state variables
        state_batch = torch.cat(batch.state)
        # state_batch[:, :-1] = state_batch[:, :-1] - means_batch[:, :-1]
        # state_batch[:, :-1] = state_batch[:, :-1] / stds_batch[:, :-1]
        # print('state_batch.shape', state_batch.shape)
        # for i in range(len(state_batch)):
        #     state_batch[i] -= means
        #     state_batch[i] /= stds
    
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
    
        # Compute a mask of non-final states and concatenate the batch elements (a final state would have been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), # returns tuple of True values for non-None next_states
                                      device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None]) # returns list of non-None next_states

        # compute target policy mu' for y_t target computation, for non-final states (otherwise zero action)
        target_action_values = torch.zeros((BATCH_SIZE, self.max_n_actions), device=self.device)
        # Context-manager that disabled gradient calculation, when we know we will not call Tensor.backward()
        with torch.no_grad(): # TODO QUESTION do I need this
            target_action_values[non_final_mask, :] = self.actor_target(non_final_next_states)
    
        # Compute the shared target Q-function from next state, s', values in replay buffer, and target mu' action values
        # add 1 to include agent indicator variable
        batch_next_state_values = torch.zeros((BATCH_SIZE, self.max_n_observations + 1), device=self.device)
        batch_next_state_values[non_final_mask, :] = non_final_next_states
        # batch_next_state_values[non_final_mask, :-1] \
        #     = batch_next_state_values[non_final_mask, :-1] \
        #       - means_batch[non_final_mask, :-1]
        # batch_next_state_values[non_final_mask, :-1] \
        #     = batch_next_state_values[non_final_mask, :-1] \
        #       / stds_batch[non_final_mask, :-1]
        
        target_Q = self.critic_target(batch_next_state_values, target_action_values)
        target_Q = reward_batch[:, None].repeat(1, self.max_n_actions) + (GAMMA * target_Q)#.detacH(
        # print('expected_state_action_values.shape', target_Q.shape)
        
        # Get current Q estimate
        current_Q = self.critic(state_batch, action_batch)
        # print('state_action_values.shape', current_Q.shape)
        
        # compute critic loss
        criterion = nn.SmoothL1Loss()
        # actor_loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
        critic_loss = criterion(current_Q, target_Q)
        
        # optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # in-place gradient clipping
        # torch.nn.utils.clip_grad_value_(self.critic.parameters(), 100) # TODO QUESTION need this?
        self.critic_optimizer.step()
        
        # Compute the actor loss as the negative mean Q value using the critic network and the actor network
        actor_loss = -self.critic(state_batch, self.actor(state_batch)).mean()
    
        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        # in-place gradient clipping
        # torch.nn.utils.clip_grad_value_(self.actor.parameters(), 100)  # TODO need this?
        self.actor_optimizer.step()
    
    def soft_update(self):
        # Soft update of the frozen target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        critic_target_net_state_dict = self.critic_target.state_dict()
        critic_net_state_dict = self.critic.state_dict()
        for key in critic_net_state_dict:
            critic_target_net_state_dict[key] = \
                critic_net_state_dict[key] * TAU + critic_target_net_state_dict[key] * (1 - TAU)
        self.critic_target.load_state_dict(critic_target_net_state_dict)

        actor_target_net_state_dict = self.actor_target.state_dict()
        actor_net_state_dict = self.actor.state_dict()
        for key in actor_net_state_dict:
            actor_target_net_state_dict[key] = \
                actor_net_state_dict[key] * TAU + actor_target_net_state_dict[key] * (1 - TAU)
        self.actor_target.load_state_dict(actor_target_net_state_dict)

    def save(self):
        """
        Saves the state dictionaries of the actor and critic networks to files
        """
        torch.save(self.actor.state_dict(), os.path.join(SAVE_DIR, 'actor.pth'))
        torch.save(self.critic.state_dict(), os.path.join(SAVE_DIR, 'critic.pth'))
    
    def load(self):
        """
        Loads the state dictionaries of the actor and critic networks to files
        """
        self.actor.load_state_dict(torch.load(os.path.join(SAVE_DIR, 'actor.pth')))
        self.critic.load_state_dict(torch.load(os.path.join(SAVE_DIR, 'critic.pth')))

def run(wf_env, agent, num_episodes, power_ref_preview, training=False, testing=False, **kwargs):
    trajectory = {'episode_length': [], 'power_tracking_error': [], 'farm_power': [], 'turbine_power': [],
                  'rotor_thrust': [], 'yaw_travel': [],
                     'ai_factors': [], 'yaw_angles': [], 'reward_yaw_angle': [], 'reward_ai_factor': []}
    
    for episode_idx in range(num_episodes):
        print(f'Running Episode {episode_idx}')
        
        for metric in trajectory:
            if metric != 'episode_length':
                trajectory[metric].append([])
        
        # Initialize environment and get flattened, tensored state
        observation_n, _ = wf_env.reset(seed=1, options={'power_ref_preview': power_ref_preview,
                                                         'sampling_time': SAMPLING_TIME})
        state_fl = {agent_id: convert_flatten(observation_n[agent_id])
                              + ([0] * (agent.max_n_observations - wf_env.n_observations[agent_id]))
                              + [i] for i, agent_id in enumerate(wf_env.agent_ids)}
        
        # print(f'state.shape = {state.shape}')
        state_fl = {agent_id: torch.tensor(state_fl[agent_id], dtype=torch.float32, device=agent.device).unsqueeze(0)
                    for agent_id in wf_env.agent_ids}
        # print(f'tensor state.shape = {state.shape}')
        
        # for each world time-step
        for t in count():
            
            if (t * kwargs['dt']) % (3600 * 1) == 0:
                print(f'{int((t * kwargs["dt"]) / 3600)} hours passed in episode {episode_idx}')
            
            if (t * kwargs['dt']) % (3600 * 6) == 0:
                agent.save()
            
            # set actions to none if the sampling time has not passed yet time-step for an agent
            action_fl = defaultdict(None)
            action_unf = defaultdict(None)
            # for each agent i
            for agent_id in wf_env.agent_ids:
                # if it is this agent's turn to go
                if t % kwargs['sampling_time'][agent_id] == 0:
                    # input the padded observation for this agent to the shared policy network mu(s),
                    # trimming if necessary for different sized action spaces
                    # to get the action for this agent
                    action_fl[agent_id] = agent.select_action(state_fl, agent_id, greedy=testing)

                    if np.any(np.isnan(action_fl[agent_id].tolist()[0])):
                        print('oh no')
                    
                    # trim acions
                    action_fl_ls = action_fl[agent_id].cpu().tolist()[0][:wf_env.n_actions[agent_id]]
                    
                    # convert flattened action back to dict form for env step method
                    if type(wf_env.action_space(agent_id)) is Dict:
                        action_unf[agent_id] = convert_unflatten(wf_env.action_space(agent_id), action_fl_ls)
                    else:
                        action_unf[agent_id] = action_fl_ls
                    
                    agent.steps_done += 1
                    agent.steps_done_n[agent_id] += 1
                else:
                    action_fl[agent_id] = None
                    action_unf[agent_id] = None
            
            # print(f'action.shape = {action.shape}')
            # print('action.shape', action.shape)
            
            # step the env forward
            observation_n, reward_n, terminated_n, truncated_n, _ = wf_env.step(action_unf, t, kwargs['sampling_time'])

            trajectory['farm_power'][episode_idx].append(wf_env.farm_power)
            trajectory['power_tracking_error'][episode_idx].append(wf_env.power_tracking_error)
            trajectory['turbine_power'][episode_idx].append(wf_env.turbine_power)
            trajectory['rotor_thrust'][episode_idx].append(tuple(wf_env.rotor_thrust))
            trajectory['yaw_travel'][episode_idx].append(tuple(wf_env.yaw_travel))
            trajectory['ai_factors'][episode_idx].append(tuple(wf_env.eff_ai_factors))
            trajectory['yaw_angles'][episode_idx].append(tuple(wf_env.yaw_angles))
            
            for agent_id in wf_env.agent_ids:
                trajectory[f'reward_{agent_id}'][episode_idx].append(reward_n[agent_id])

            # print('observation.shape', np.array(observation['yaw_angle']).shape)
            
            # TODO is the below only considered in training? what about batch normalization
            if training:
                # for each agent i
                next_state_fl = {}
                observation_fl = {}
                for i, agent_id in enumerate(wf_env.agent_ids):
                    # if it is this agent's turn to go
                    if t % SAMPLING_TIME[agent_id] == 0:
                        reward_fl = torch.tensor([reward_n[agent_id]], device=agent.device)
                        
                        # pad shorter observation with zeros and with identity of agent
                        observation_fl[agent_id] = convert_flatten(observation_n[agent_id]) \
                                                   + ([0] * (agent.max_n_observations - wf_env.n_observations[agent_id])) \
                                                   + [i]
                        if np.any(np.isnan(observation_fl[agent_id])):
                            print('oh no')
                            
                        if terminated_n[agent_id]:
                            next_state_fl[agent_id] = None
                        else:
                            next_state_fl[agent_id] = torch.tensor(observation_fl[agent_id], dtype=torch.float32,
                                                                   device=agent.device).unsqueeze(0)
                            # print(f'next_state.shape = {next_state.shape}')
                        
                        # TODO QUESTION will the fact that the 'objective' (ie reward function) differs for different agents be a problem with parameter sharing
                        # TODO QUESTION could it be problematic if we collect the reward and observations after both yaw and angle act?
                        # store the transition for this agent in memory
                        agent.memory.push(state_fl[agent_id],
                                          action_fl[agent_id],
                                          next_state_fl[agent_id], reward_fl)
                        
                        # Move to the next state
                        state_fl[agent_id] = next_state_fl[agent_id]
                        
                        # perform one step of the optimization (on the policy network)
                        agent.optimize_model()
                        
                        # Soft update target networks for critic and actor
                        agent.soft_update()
                        
                        # maintain running average of mean and standard deviation to use for batch normalization
                        
                        agent.state_mean = agent.state_mean + ((state_fl[agent_id] - agent.state_mean) / agent.steps_done)
                        
                        agent.state_std = (agent.state_std ** 2 + (
                            (state_fl[agent_id] - agent.state_mean) ** 2 - agent.state_std ** 2) / agent.steps_done) ** 0.5
            
            # print(f'observation.shape = {observation.shape}')
            done = all(terminated_n.values()) or all(truncated_n.values())
            
            # if t == 3600:
            #     done = True
            
            if done:
                trajectory['episode_length'].append(t + 1)
                # agent.episode_durations.append(t + 1)
                # plot_durations(agent.episode_durations)
                break
    
    # TODO checkpoint state of agent
    # print('Complete')
    # durations_t = torch.tensor(agent.episode_durations, dtype=torch.float)
    # plot_durations(durations_t, show_result=True)
    # plt.ioff()
    # plt.show()
    if training:
        agent.training_trajectory = trajectory
    elif testing:
        agent.testing_trajectory = trajectory
    return trajectory


if __name__ == '__main__':
    
    # register the environment
    # gym.register('wf_env', WFEnv, kwargs=ENV_CONFIG)
    
    # create the environment
    # wf_env = gym.make("wf_env")  # WFEnv(env_config)
    wf_env = WFEnv(**ENV_CONFIG)
    
    # if gpu is to be used
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print('CUDA is available')
        num_training_episodes = 2
        num_testing_episodes = 1
    else:
        num_training_episodes = 2
        num_testing_episodes = 1

    # optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
    memory = ReplayMemory(REPLAY_SIZE)
    
    # set seeds
    wf_env.seed(0)
    torch.manual_seed(0)
    np.random.seed(0)
    
    n_observations = wf_env.n_observations
    n_actions = wf_env.n_actions

    # Exploration Noise
    # exploration_noise = 0.1
    # exploration_noise = 0.1 * max_action

    # load power reference signal
    power_ref_preview = [30 * 1e6] * int(24 * 3600 // DT)

    # Create a DDPG instance
    agent = PSDDPG(wf_env, device, memory)
   
    # TODO which to cpu for calls to step
    
    training_trajectory = run(wf_env, agent, num_training_episodes, power_ref_preview, dt=DT, sampling_time=SAMPLING_TIME, training=True)
    np.save(os.path.join('./trajectories', 'training_trajectory'), training_trajectory)
    np.load(os.path.join('./trajectories', 'training_trajectory'))
    plot_cumulative_reward(training_trajectory, ['yaw_angle'])
    plot_tracking_errors(training_trajectory, wf_env.n_turbines)
    
    testing_trajectory = run(wf_env, agent, num_testing_episodes, power_ref_preview, dt=DT, sampling_time=SAMPLING_TIME, testing=True)
