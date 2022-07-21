from environments.runner import DDQN_GameAgent
from utils import tools
import numpy as np
import yaml

with open("config_ddqn.yaml", 'r') as stream:
    running_config = yaml.safe_load(stream)

np.random.seed(seed=running_config['RANDOM_SEED'])

tools.mkdir('model/q_networks')
tools.mkdir('results/Default')
tools.mkdir('results/DR')



agent = DDQN_GameAgent(config=running_config)
agent.train(mode='Default', n_games=500)
# agent.run(mode='Default')

# agent.train(mode='DR', n_games=3000)
# agent.run(mode='DR')