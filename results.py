from floridyn.tools.visualization import plot_turbines_with_fi, visualize_cut_plane
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
	from IPython import display

plt.ion()

def plot_cumulative_reward(trajectory, agent_ids):
	# plot mean cumulative reward per time step during training # TODO what is mean cumulative reward?
	
	fig, axs = plt.subplots(len(agent_ids), 1)
	# for each episode tested
	for episode_idx in range(len(trajectory['episode_length'])):
		if len(agent_ids) == 1:
			ax = [axs]
		
		for i, agent_id in enumerate(agent_ids):
			ax[i].scatter(trajectory['episode_length'][episode_idx], np.cumsum(trajectory[f'reward_{agent_id}'][episode_idx]))
			ax[i].set(title=f"{' '.join([str.capitalize() for str in agent_id.split('_')])} Agent Reward")
			
	fig.show()

def plot_tracking_errors(trajectory, n_turbines):
	# plot the power, yaw_travel, rotor_thrust tracking error per time-step during training/evaluation
	fig, ax = plt.subplots(3, 1)
	
	# for each episode tested
	for episode_idx in range(len(trajectory['episode_length'])):
		ax[0].scatter(trajectory['episode_length'][episode_idx], trajectory['power_tracking_error'][episode_idx])
		ax[0].set(title="Farm Power Tracking Error")
		
		for k in range(n_turbines):
			ax[1].scatter(trajectory['episode_length'][episode_idx], [ls[k] for ls in trajectory['yaw_travel'][episode_idx]], label=f'Turbine {k}')
		ax[1].set(title="Yaw Travel", xlabel='Episode Time-Step')
		ax[1].legend()
		
		for k in range(n_turbines):
			ax[1].scatter(trajectory['episode_length'][episode_idx], [ls[k] for ls in trajectory['rotor_thrust'][episode_idx]], label=f'Turbine {k}')
		ax[1].set(title="Rotor Thrust", xlabel='Episode Time-Step')
		ax[1].legend()
	
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
	if is_ipython:
		if not show_result:
			display.display(plt.gcf())
			display.clear_output(wait=True)
		else:
			display.display(plt.gcf())