from trainer.Q_Learning.ddqn_main import DDQN_GameAgent
from utils import tools
import numpy as np

def init_working_dir():
    tools.mkdir('model/q_networks')
    tools.mkdir('results/Default')
    tools.mkdir('results/DR')

def ddqn():
    print('=====================DDQN======================')
    config = tools.load_config("configs/config_ddqn.yaml")
    tools.setup_seed(config['RANDOM_SEED'])
    # print(config)
    # agent = DDQN_GameAgent(config=config, network='Default')
    # agent.train(mode='Default', n_games=300)
    # agent.evaluate(mode='Default')
    # agent.train(mode='DR', n_games=3000)
    # agent.evaluate(mode='DR')
def ppo():
    print('=====================PPO=======================')
    config = tools.load_config("configs/config_ppo_default.yaml")
    tools.setup_seed(config['random_seed'])
    args = tools.dict2class(config)
    print(args.random_seed)

if __name__ == '__main__':
    init_working_dir()
    # ddqn()
    ppo()

    
   