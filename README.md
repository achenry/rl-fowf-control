# Code for ``Reinforcement Learning for Optimal Operation of Stochastic Floating
Wind Farm Environment"

__Authors__: Aoife Henry, Josh Myers-Dean

__CSCI 7000__: Deep Reinforcement Learning and Robotics 

Currently only PPO is working as there are known bugs when attempting to evaluate QMIX with RLLIB. To train PPO with a Floating Offshore Wind Farm,

```bash
python train_ppo.py {save name} {number of episodes} {rewards save name}
```
