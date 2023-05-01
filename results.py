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

# set up matplotlib
# is_ipython = 'inline' in matplotlib.get_backend()
# if is_ipython:
# 	from IPython import display

plt.ion()

def plot_cumulative_reward(trajectory, agent_ids):
	# plot mean cumulative reward per time step during training # TODO what is mean cumulative reward?
	
	fig, ax = plt.subplots(1, 1)
	# for each episode tested
	for episode_idx in range(len(trajectory['episode_length'])):
		ax.scatter(np.arange(trajectory['episode_length'][episode_idx]), np.cumsum(trajectory[f'reward'][episode_idx]))
		ax.set(title=f"Reward")
			
	fig.show()

def plot_tracking_errors(trajectory, n_turbines):
	# plot the power, yaw_travel, rotor_thrust tracking error per time-step during training/evaluation
	fig, ax = plt.subplots(3, 1)
	# for each episode tested
	for episode_idx in range(len(trajectory['episode_length'])):
		ax[0].scatter(np.arange(trajectory['episode_length'][episode_idx]), trajectory['power_tracking_error'][episode_idx])
		ax[0].set(title="Farm Power Tracking Error")
		
		for k in range(n_turbines):
			ax[1].scatter(np.arange(trajectory['episode_length'][episode_idx]), [ls[k] for ls in trajectory['yaw_travel'][episode_idx]], label=f'Turbine {k}')
		ax[1].set(title="Yaw Travel", xlabel='Episode Time-Step')
		# ax[1].legend()
		
		for k in range(n_turbines):
			ax[2].scatter(np.arange(trajectory['episode_length'][episode_idx]), [ls[k] for ls in trajectory['rotor_thrust'][episode_idx]], label=f'Turbine {k}')
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
	with open(os.path.join('./trajectories', 'training_trajectory.pickle'), 'rb') as handle:
		training_trajectory = pickle.load(handle)
	plot_tracking_errors(training_trajectory, 9)
	plot_cumulative_reward(training_trajectory, ['yaw_angle'])