from environments.play_ddqn import DDQN_GameAgent
from utils import tools
import numpy as np


config = tools.load_config("configs/config_ddqn.yaml")

tools.setup_seed(config['RANDOM_SEED'])

tools.mkdir('model/q_networks')
tools.mkdir('results/Default')
tools.mkdir('results/DR')

agent = DDQN_GameAgent(config=config, network='Default')
# agent.train(mode='Default', n_games=300)
agent.run(mode='Default')

# agent.train(mode='DR', n_games=3000)
# agent.run(mode='DR')