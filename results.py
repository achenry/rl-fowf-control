from floridyn.tools.visualization import plot_turbines_with_fi, visualize_cut_plane
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import pandas as pd
import os
import torch
import pickle

from constants import SAVE_DIR, FIG_DIR, LEARNING_ENDS, N_TRAINING_EPISODES, N_TESTING_EPISODES

# set up matplotlib
# is_ipython = 'inline' in matplotlib.get_backend()
# if is_ipython:
# 	from IPython import display

plt.ion()

# def get_testing_episodes(trajectory):
# 	# find first index in which episode idx comprises all testing data
# 	return np.where(LEARNING_ENDS >= np.cumsum(trajectory['episode_length']))[0]
# 	# # for each episode run
# 	# for ep_length in trajectory['episode_length']:
# 	# 	testing_trajectory_start += min(ep_length, LEARNING_ENDS - testing_trajectory_start)

def plot_cumulative_reward(trajectory, reward_types, episode_indices):
	# plot mean cumulative reward per time step during training # TODO what is mean cumulative reward?
	
	fig, ax = plt.subplots(1, 1)
	# for each episode tested
	for ep_idx in episode_indices:
		if ep_idx not in episode_indices:
			continue
		for rt in reward_types:
			ax.plot(np.arange(trajectory['episode_length'][ep_idx]), np.cumsum(trajectory[f'{rt}_reward'][ep_idx]),
			        label=f"{' '.join([s.capitalize() for s in rt.split('_')])}", linestyle='--')
		ax.plot(np.arange(trajectory['episode_length'][ep_idx]), np.cumsum(trajectory[f'total_reward'][ep_idx]),
			        label=f"{' '.join([s.capitalize() for s in rt.split('_')])}", linestyle='-')
		ax.set(title=f"Cumulative Reward")
			
	fig.show()


def plot_state_trajectory(trajectory, states, episode_indices, agent_indices):
	# plot mean cumulative reward per time step during training
	
	fig, axs = plt.subplots(len(states), 1, sharex=True)
	if len(states) == 1:
		axs = [axs]
	# for each episode tested
	for ep_idx in episode_indices:
		if ep_idx not in episode_indices:
			continue
		for i, s in enumerate(states):
			data = [[all_agent_data[i] for i in agent_indices] for all_agent_data in trajectory[s][ep_idx]]
			axs[i].plot(np.arange(trajectory['episode_length'][ep_idx]), data,
			        label=f"{' '.join([x.capitalize() for x in s.split('_')])}")
			axs[i].set(title=f"{' '.join([x.capitalize() for x in s.split('_')])} Trajectory")
	
	fig.show()
def plot_inst_reward(trajectory, reward_types, episode_indices):
	# plot mean cumulative reward per time step during training
	
	fig, axs = plt.subplots(len(reward_types) + 1, 1, sharex=True)
	if len(reward_types) == 0:
		axs = [axs]
		
	# for each episode tested
	for ep_idx in episode_indices:
		if ep_idx not in episode_indices:
			continue
		for i, rt in enumerate(reward_types):
			label = f"{' '.join([s.capitalize() for s in rt.split('_')])}"
			axs[i].plot(np.arange(trajectory['episode_length'][ep_idx]), trajectory[f'{rt}_reward'][ep_idx],
			        label=label, linestyle='--')
			axs[i].set(title=f"Instantaneous {label} Reward")
		axs[-1].plot(np.arange(trajectory['episode_length'][ep_idx]), trajectory[f'total_reward'][ep_idx],
		        label="Total Reward", linestyle='-')
		axs[-1].set(title=f"Instantaneous Total Reward")
		
	
	fig.show()

def plot_tracking_errors(trajectory, episode_indices, agent_indices):
	# plot the power, yaw_travel, rotor_thrust tracking error per time-step during training/evaluation
	fig, ax = plt.subplots(3, 1, sharex=True)
	# for each episode tested
	for ep_idx in episode_indices:
		if ep_idx not in episode_indices:
			continue
			
		ax[0].plot(np.arange(trajectory['episode_length'][ep_idx]), trajectory['power_tracking_error'][ep_idx])
		ax[0].set(title="Farm Power Tracking Error")
		
		for k in agent_indices:
			ax[1].plot(np.arange(trajectory['episode_length'][ep_idx]), [ls[k] for ls in trajectory['yaw_travel'][ep_idx]], label=f'Turbine {k}')
		ax[1].set(title="Yaw Travel", xlabel='Episode Time-Step')
		# ax[1].legend()
		
		for k in agent_indices:
			ax[2].plot(np.arange(trajectory['episode_length'][ep_idx]), [ls[k] for ls in trajectory['rotor_thrust'][ep_idx]], label=f'Turbine {k}')
		ax[2].set(title="Rotor Thrust", xlabel='Episode Time-Step')
		ax[2].legend()
		
	fig.show()

def plot_wind_farm(system_fi):
	farm_fig, farm_ax = plt.subplots(1, 1)
	hor_plane = system_fi.get_hor_plane()
	im = visualize_cut_plane(hor_plane, ax=farm_ax)
	divider = make_axes_locatable(farm_ax)
	cax = divider.append_axes("right", size="2.5%", pad=0.15)
	farm_fig.colorbar(im, cax=cax)
	# plot_turbines_with_fi(farm_ax, system_fi)
	for t in system_fi.turbine_indices:
		x = system_fi.layout_x[t] - 100
		y = system_fi.layout_y[t]
		farm_ax.annotate(f'T{t}', (x, y), ha="center", va="center")
	
	farm_ax.set_xlabel('Streamwise Distance [m]')
	farm_ax.set_ylabel('Cross-\nStream\nDistance\n[m]', rotation=0, ha='right', labelpad=15.0, y=0.7)
	
	return farm_fig

def plot_durations(durations_t, show_result=False):
	plt.figure(1)
	
	if show_result:
		plt.title('Result')
	else:
		plt.clf()
		plt.title('Training...')
		
	plt.xlabel('Episode')
	plt.ylabel('Duration')
	plt.plot(durations_t.numpy())

	# Take 100 episode averages and plot them too
	if len(durations_t) >= 100:
		means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
		means = torch.cat((torch.zeros(99), means))
		plt.plot(means.numpy())
	
	plt.pause(0.001)  # pause a bit so that plots are updated
	# if is_ipython:
	# 	if not show_result:
	# 		display.display(plt.gcf())
	# 		display.clear_output(wait=True)
	# 	else:
	# 		display.display(plt.gcf())
			
if __name__ == '__main__':
	# tensorboard --logdir=runs
	# event_acc = EventAccumulator("runs/trajectory", size_guidance={'tensors': 0})
	#
	# pd.DataFrame([(w, s, t.to_numpy()) for w, s, t in event_acc.Tensors('my_metric')], 2, \
	# 	columns = ['wall_time', 'step', 'tensor'])
	# training_trajectory = np.load(os.path.join('./trajectories', 'training_trajectory.pickle'), allow_pickle=True)
	# multi_turbine_env-v0__CleanRL_PSDDPG__1__1683043106
	run_env_id = 'multi_turbine_env-v0'
	run_exp_name = 'CleanRL_PSDDPG'
	run_seed = 1
	run_time = 1683055949
	run_name = f"{run_env_id}__{run_exp_name}__{run_seed}__{run_time}.pickle"
	with open(os.path.join(SAVE_DIR, 'trajectories', run_name), 'rb') as handle:
		trajectory = pickle.load(handle)
	
	# testing_episode_idx = get_testing_episodes(trajectory)
	testing_episode_idx = np.arange(N_TRAINING_EPISODES, N_TRAINING_EPISODES + N_TESTING_EPISODES)
	testing_episode_idx = [0]
	agent_idx = [0]
	plot_state_trajectory(trajectory, ['yaw_angles', 'ai_factors'], testing_episode_idx, agent_idx)
	plot_tracking_errors(trajectory, testing_episode_idx, agent_idx)
	plot_inst_reward(trajectory, ['power_tracking', 'rotor_thrust', 'yaw_travel'], testing_episode_idx)
	# plot_inst_reward(trajectory, [], testing_episode_idx)
	plot_cumulative_reward(trajectory, ['power_tracking', 'rotor_thrust', 'yaw_travel'], testing_episode_idx)