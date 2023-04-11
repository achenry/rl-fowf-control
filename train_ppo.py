import gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

import torch

from WFEnv import FOWFEnv

from tqdm import tqdm
import pickle
import sys

def create_env_and_model(env_config, policy="MultiInputPolicy"):
    env = FOWFEnv(env_config)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = PPO(policy, env, verbose=1, device=device)
    return env, model

def train(env, model, epochs=1000, save_name="ppo_fowf", rewards_name="rewards.pkl"):
    obs = env.reset()
    rewards = []
    for _ in tqdm(range(epochs)):
        obs = env.reset()
        done = False
        while not done:
            action, _states = model.predict(obs)
            obs, reward, done, info = env.step(action)
            rewards.append(reward)
    with open(rewards_name, 'wb') as f:
        pickle.dump(rewards, f)
    model.save(save_name)

def eval_policy(env, load_name="ppo_fowf"):
    model = PPO.load(load_name)
    mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
    print(f"Reward is {mean_reward} \pm {std_reward}")

    
if __name__ == "__main__":
    
    env_config = {
        "floris_input_file": "/scratch/alpine/jomy5901/rl-fowf-control/9turb_floris_input.json",
        "turbine_layout_std": 1,
        "offline_probability": 0.001,
    }


    save_name = sys.argv[1]
    epochs = int(sys.argv[2])
    rewards_name = sys.argv[3]
    print(save_name, epochs, rewards_name)
    
    env, model = create_env_and_model(env_config)
    train(env, model, epochs, save_name, rewards_name)
    eval_policy(env, model)
    
