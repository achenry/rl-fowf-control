import numpy as np


class FOWFAgent:
    def __init__(self, turbine_id=0, eps=0.4, eps_decay=1):
        self.turbine_id = turbine_id
        self.eps = eps
        self.eps_decay = eps_decay

    def decay(self):
        """
        Decay epsilon for epsilon-greedy learning
        """
        self.eps *= self.eps_decay

    def choose_actions(self, q_vals, fowf_env):
        curr_actions = {k: None for k, v in fowf_env.action_space.items()}
        for k in fowf_env.action_space:
            if np.random.rand() < self.eps:
                act = fowf_env.action_space[k].sample()
                act = act + (self.turbine_id * fowf_env.action_space[k].n)
            else:
                act = np.argmax(q_vals[k])
                act = act + (self.turbine_id * fowf_env.action_space[k].n)
            curr_actions[k] = act
        return curr_actions
