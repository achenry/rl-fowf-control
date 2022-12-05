import gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

import torch

from FOWFEnv import FOWFEnv

def create_env_and_model(env_config, policy="MultiInputPolicy"):
    env = FOWFEnv(env_config)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = PPO(policy, env, verbose=1, device=device)
    return env, model

def train(env, model, epochs=10000, save_name="ppo_fowf"):
    obs = env.reset()
    rewards = []
    for _ in range(epochs):
        done = False
        while not done:
            action, _states = model.predict(obs)
            obs, reward, done, info = env.step(action)
            rewards.append(reward)
    model.save(save_name)

def eval_policy(env, load_name="ppo_fowf"):
    model = PPO.load(load_name)
    mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
    print(f"Reward is {mean_reward} \pm {std_reward}")

    
if __name__ == "__main__":
    
    env_config = {
        "floris_input_file": "/scratch/summit/jomy5901/rl-fowf-control/9turb_floris_input.json",
        "turbine_layout_std": 1,
        "offline_probability": 0.001,
    }
    
    env, model = create_env_and_model(env_config)
    train(env, model)
    eval_policy(env, model)
    
