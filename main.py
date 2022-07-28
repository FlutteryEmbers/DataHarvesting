from trainer.DDQN.ddqn_main import DDQN_GameAgent
from trainer.PPO.PPO_continuous_main import PPO_GameAgent
from utils import tools
from loguru import logger
import sys
import torch as T

def init_working_dir():
    tools.mkdir('model/q_networks')
    tools.mkdir('model/ppo')

def ddqn():
    logger.critical('Start DDQN Session')
    config = tools.load_config("configs/config_ddqn.yaml")
    tools.setup_seed(config['RANDOM_SEED'])
    ## network: trainning algorithm using: MLP/CNN network 
    agent = DDQN_GameAgent(config=config, network='MLP')

    ## trainning_mode:
    ## - Default: Trainning for a specific environment;
    ## - DR: Trainning with randomized initial states
    # agent.train(env_type='Default', n_games=10)
    agent.evaluate(env_type='Default')
    # agent.train(env_type='DR', n_games=1000)
    # agent.evaluate(env_type='DR')

def ppo():
    logger.critical('Start PPO Session')
    mode = 'Default'
    config = tools.load_config("configs/config_ppo_{}.yaml".format(mode.lower()))
    tools.setup_seed(config['random_seed'])
    args = tools.dict2class(config)
    agent = PPO_GameAgent(args = args)
    # agent.train(env_type=mode)
    agent.evaluate_policy(args=args, load_model=mode)

if __name__ == '__main__':
    
    tools.set_logger_level(1)

    init_working_dir()
    ddqn()
    # ppo()

    
   