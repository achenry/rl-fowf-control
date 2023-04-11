from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.logger import pretty_print
from ray.tune import register_env
import os
from WFEnv import FOWFEnv
import torch

if __name__ == '__main__':
	
	env_config = {
		"floris_input_file": os.path.join(os.getcwd(), "9turb_floris_input.json"),
		"turbine_layout_std": 1,
		"offline_probability": 0.001,
	}
	
	# gym.register(
	register_env(
		"fowf_env",
		lambda config: FOWFEnv(**env_config),
	)
	num_gpus = torch.cuda.device_count()
	algo = (
		PPOConfig()
			.rollouts(num_rollout_workers=4) # number of parallel workers to collect samples from the environment,  Rollout workers query the policy to determine agent actions
			.resources(num_gpus=num_gpus)
			.environment(env="fowf_env")
			.framework(framework="torch")
			.training(train_batch_size=4000, model={"fcnet_hiddens": [64, 64]})
			.evaluation(evaluation_num_workers=4)
			.build()
	)
	
	for i in range(10):
		result = algo.train()
		print(pretty_print(result))
		
		if i % 5 == 0:
			checkpoint_dir = algo.save()
			print(f"Checkpoint saved in directory {checkpoint_dir}")