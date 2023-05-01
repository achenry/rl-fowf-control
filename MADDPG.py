from multiagent.environment import MultiAgentEnv
import numpy as np
from multiagent.core import World, Agent
from multiagent.scenario import BaseScenario

# https://antonai.blog/multi-agent-reinforcement-learning-openais-maddpg/
# https://github.com/Ah31/maddpg_pytorch

class Turbine(object):
    def __init__(self, id, loc):
        self.id = id
        self.loc = loc

class WindFarmWorld(object):
    def step(self):
        pass
    
class WindFarmScenario(BaseScenario):
    
    def make_world(self, agent_types):
        world = WindFarmWorld()
        # add agents
        world.agents = [Agent(agent_type) for agent_type in agent_types]
        
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d'
        # add turbines
        world.turbines = [Turbine() for i in range(1)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.25,0.25,0.25])
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.75,0.75,0.75])
        world.landmarks[0].color = np.array([0.75,0.25,0.25])
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1,+1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.random.uniform(-1,+1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

    def reward(self, agent, world):
        dist2 = np.sum(np.square(agent.state.p_pos - world.landmarks[0].state.p_pos))
        return -dist2

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        return np.concatenate([agent.state.p_vel] + entity_pos)

    

def make_env(scenario_name, benchmark=False):
    '''
    Creates a MultiAgentEnv object as env. This can be used similar to a gym
    environment by calling env.reset() and env.step().
    Use env.render() to view the environment on the screen.
    Input:
        scenario_name   :   name of the scenario from ./scenarios/ to be Returns
                            (without the .py extension)
        benchmark       :   whether you want to produce benchmarking data
                            (usually only done during evaluation)
    Some useful env properties (see environment.py):
        .observation_space  :   Returns the observation space for each agent
        .action_space       :   Returns the action space for each agent
        .n                  :   Returns the number of Agents
    '''
    from multiagent.environment import MultiAgentEnv
    # import multiagent.scenarios as scenariosd

    # load scenario from script
    # scenario = scenarios.load(scenario_name + ".py").Scenario()
    scenario = WFScenario()
    
    # create world
    world = scenario.make_world()
    
    # create multiagent environment
    if benchmark:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    return env

if __name__ == '__main__':
	wf_env = make_env("wf_env")
    print(wf_env.action_space)