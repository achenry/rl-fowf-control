import gymnasium as gym
import math
import random
import matplotlib
from matplotlib import pyplot as plt
from collections import namedtuple, deque, MutableMapping
from itertools import count
import numpy as np
from sklearn.preprocessing import StandardScaler

from gymnasium.spaces import Dict

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import os
from WFEnv import WFEnv, ENV_CONFIG, DT

# set up matplotlib
IS_IPYTHON = 'inline' in matplotlib.get_backend()
if IS_IPYTHON:
	from IPython import display

plt.ion()  # enable interactive mode

# TODO batch normalization
# TODO terminate episode if yaw travel is exceeded or power produced too farm from reference
# TODO available wind speed should make it possible to achieve power reference signal: generate wind field preview, then generate Pref from this
# TODO need autoregressive inputs - not a Markov Process?

# DEFINE CONSTANTS
BATCH_SIZE = 128 # number of transitions sampled from the replay buffer
GAMMA = 0.99 # discount factor
EPS_START = 0.9 # starting value of epsilong in greedy policy action selection
EPS_END = 0.05 # final value of epsilong in greedy policy action selection
EPS_DECAY = 1000 # rate of exponential decay in greedy policy action selection, high for slow decay
TAU = 0.005 # update rate of the target network
LR = 1e-4 # learning rate of the AdamW optimizer
REPLAY_SIZE = int(1e4) # capacity of replay memory

# TODO test Sarsa equivalent of this?

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


# code to convert ini_dict to flattened dictionary
# default separator '_'
def convert_flatten(d, parent_key='', sep='_'):
	items = []
	for k, v in d.items():
		new_key = parent_key + sep + k if parent_key else k
		
		if isinstance(v, MutableMapping):
			items.extend(convert_flatten(v, new_key, sep=sep).items())
		else:
			items.append((new_key, v))
	return dict(items)

def convert_unflatten(d_template, arr):
	d_unf = {}
	for i, (k, v) in enumerate(d_template.items()):
		if isinstance(v, Dict):
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
		# TODO ensure contains given % of non-conventional farm layout states
		self.memory.append(Transition(*args))
	
	def sample(self, batch_size):
		""" Select a random batch of transitions for training"""
		# TODO ensure a % of sample contain variations with current farm layout
		return random.sample(self.memory, batch_size)
	
	def __len__(self):
		return len(self.memory)

class DQN(nn.Module):
	""" FF NN to predict expected return of taking each action given the current state."""
	def __init__(self, n_observations, n_actions):
		super(DQN, self).__init__()
		self.layer1 = nn.Linear(n_observations, 128) # first layer reads observed states, outputs 128 nodes
		self.layer2 = nn.Linear(128, 128) # middle layer reads previous layer of 128 nodes, outputs 128 nodes
		self.layer3 = nn.Linear(128, n_actions) # final layer reads previous 128 nodes, outputs control actions actions
	
	def forward(self, x):
		""" Feed Forward neural ne. Called with either one element to determine next action, or a batch during optimization. Returns tensor([[ctrl_Action1, ctrl_action2, ...]])"""
		x = F.relu(self.layer1(x))
		x = F.relu(self.layer2(x))
		return self.layer3(x)

class Agent(object):
	def __init__(self, device, env, optimizer, memory):
		self.steps_done = 0
		self.device = device
		self.env = env
		self.optimizer = optimizer
		self.memory = memory
		self.episode_durations = []
		self.state_mean = np.zeros((env.n_observations,))
		self.state_std = np.zeros((env.n_observations))
		
	def select_action(self, state):
		# global steps_done
		sample = random.random()
		eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1 * self.steps_done / EPS_DECAY)
		self.steps_done += 1
		if sample > eps_threshold:
			with torch.no_grad():
				# t.max(1) will return the largest column value of each row.
				# second column on max result is index of where max element was
				# found, so we pick action with the larger expected reward.
				# print(f'policy_net(state).shape = {policy_net(state).shape}')
				# TODO test
				state = state - torch.tensor(self.state_mean, dtype=torch.float32)
				# std = self.state_std.copy()
				# std[std == 0] = 1
				state = state / torch.tensor(self.state_std, dtype=torch.float32)
				return policy_net(state).max(0)[1].view(1, self.env.n_actions)
		else:
			# take random sample of actions if we have not exceeded the eps_threshold for greedy action selection
			action_sample = self.env.action_space.sample()
			action_sample = [list(convert_flatten(action_sample).values())]
			# print(f'[env.action_space.sample()] = {action_sample}')
			return torch.tensor(action_sample, device=device, dtype=torch.long)
	
	def plot_durations(self, show_result=False):
		plt.figure(1)
		durations_t = torch.tensor(self.episode_durations, dtype=torch.float)
		if show_result:
			plt.title('Result')
		else:
			plt.clf() # clear current figure
			plt.title('Training...')
		plt.xlabel('Episode')
		plt.ylabel('Duration')
		
		plt.plot(durations_t.numpy(()))
		
		# Take 100 episode averags and plot them too
		if len(durations_t) >= 100:
			means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
			means = torch.cat((torch.zeros(99), means))
			plt.plot(means.numpy())
		
		# Pause st plots are updated
		plt.pause(0.001)
		if IS_IPYTHON:
			if not show_result:
				display.dispay(plt.gcf())
				display.clear_output(wait=True)
			else:
				display.display(plt.gcf())
	
	def optimize_model(self):
		if len(self.memory) < BATCH_SIZE:
			return
		transitions = memory.sample(BATCH_SIZE)
		
		# transpose the batch (https://stackoverflow.com/a/19343/3343043) to convert batch-arry of Transitions to Transition of batch arrays
		batch = Transition(*zip(*transitions))
		
		# TODO normalize state_batch, better to do this with repmat tensor operation?
		next_state_batch = list(batch.next_state)
		means = torch.tensor(self.state_mean, dtype=torch.float32)
		stds = torch.tensor(self.state_std, dtype=torch.float32)
		for i in range(len(next_state_batch)):
			next_state_batch[i] -= means
			next_state_batch[i] /= stds
		# next_state_batch = tuple(next_state_batch)
		
		# Compute a mask of non-final states and concatenate the batch elements (a final state would have been the one after which simulation ended)
		non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, next_state_batch)), device=self.device, dtype=torch.bool)
		non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
		
		state_batch = torch.cat(batch.state)
		action_batch = torch.cat(batch.action)
		reward_batch = torch.cat(batch.reward)
		
		# TODO normalize state_batch
		for i in range(len(state_batch)):
			state_batch[i] -= means
			state_batch[i] /= stds

		# Compute Q(s_t, a) - the model computes Q(s_t), then we select the
		# columns of actions taken. These are the actions which would've been taken
		# for each batch state according to policy_net
		state_action_values = policy_net(state_batch).gather(1, action_batch)

		# Compute V(s_{t+1}) for all next states.
		# Expected values of actions for non_final_next_states are computed based
		# on the "older" target_net; selecting their best reward with max(1)[0].
		# This is merged based on the mask, such that we'll have either the expected
		# state value or 0 in case the state was final.
		next_state_values = torch.zeros(BATCH_SIZE, device=self.device)
		with torch.no_grad():
			next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
		
		# compute the expected Q values
		expected_state_action_values = reward_batch + (GAMMA * next_state_values)
		
		# compute Huber loss
		criterion = nn.SmoothL1Loss()
		loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
		
		# optimize the model
		optimizer.zero_grad()
		loss.backward()
		
		# in-place gradient clipping
		torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
		optimizer.step()

if __name__ == '__main__':
	WORKING_DIR = os.getcwd()

	# register the environment

	gym.register('wf_env', WFEnv, kwargs=ENV_CONFIG)

	# create the environment
	wf_env = gym.make("wf_env")  # WFEnv(env_config)
	
	# if gpu is to be used
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	if torch.cuda.is_available():
		print('CUDA is available')
		num_episodes = 1
	else:
		num_episodes = 1
		
	# setup NN Q function
	n_actions = wf_env.n_actions
	n_observations = wf_env.n_observations
	# state, info = wf_env.reset()
	# n_observations = len(state)
	
	policy_net = DQN(n_observations, n_actions).to(device)
	target_net = DQN(n_observations, n_actions).to(device)
	target_net.load_state_dict(policy_net.state_dict())
	
	optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
	memory = ReplayMemory(REPLAY_SIZE)
	
	agent = Agent(device, wf_env, optimizer, memory)
	# TODO MPI, data-modelled parallelization
	# TODO how to parallelize this over multiple GPUs
	for i_episode in range(num_episodes):
		print(f'Running Episode {i_episode}')
		# Initialize environment
		state, info = wf_env.reset(seed=1, options={})
		state = np.array(list(convert_flatten(state).values()))
		# print(f'state.shape = {state.shape}')
		state = torch.tensor(state, dtype=torch.float32, device=agent.device).unsqueeze(0)
		# print(f'tensor state.shape = {state.shape}')
		for t in count():
			
			if (t * DT) % 3600 == 0:
				print(f'{int((t * DT) / 3600)} hours passed in episode {i_episode}')
			
			# TODO QUESTION do we evaluate in practise by simply selecting a greedy action?
			action = agent.select_action(state)
			# print(f'action.shape = {action.shape}')
			
			# convert back to dict form
			action_dict = convert_unflatten(agent.env.action_space, action.squeeze().tolist())
			
			observation, reward, terminated, truncated, _ = agent.env.step(action_dict)
			observation = np.array(list(convert_flatten(observation).values()))
			
			# maintain running average of mean and standard deviation to use for batch normalization
			agent.state_mean += (observation - agent.state_mean) / agent.steps_done
			agent.state_std = (agent.state_std**2 + ((observation - agent.state_mean)**2 - agent.state_std**2) / agent.steps_done)**0.5
			# print(f'observation.shape = {observation.shape}')
			
			reward = torch.tensor([reward], device=agent.device)
			done = terminated or truncated
			
			if terminated:
				next_state = None
			else:
				next_state = torch.tensor(observation, dtype=torch.float32, device=agent.device).unsqueeze(0)
				# print(f'next_state.shape = {next_state.shape}')
				
			# store the transition in memory
			agent.memory.push(state, action, next_state, reward)
			
			# Move to the next state
			state = next_state
			
			# perform one step of the optimization (on the policy network)
			agent.optimize_model()
			
			# Soft update of the target network's weights
			# θ′ ← τ θ + (1 −τ )θ′
			target_net_state_dict = target_net.state_dict()
			policy_net_state_dict = policy_net.state_dict()
			for key in policy_net_state_dict:
				target_net_state_dict[key] = policy_net_state_dict[key] * TAU + target_net_state_dict[key] * (1 - TAU)
			target_net.load_state_dict(target_net_state_dict)
			
			if done:
				agent.episode_durations.append(t + 1)
				agent.plot_durations()
				break
	
	# TODO checkpoint state of agent
	print('Complete')
	agent.plot_durations(show_result=True)
	plt.ioff()
	plt.show()