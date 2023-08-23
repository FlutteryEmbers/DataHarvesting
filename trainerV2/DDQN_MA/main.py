import sys, os
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/../..'))

from trainerV2.DDQN_MA.scripts import HER_Batch_Trainer
from utils import graph, tools
from loguru import logger

def init_working_dir():
    tools.mkdir('cache/model/q_networks')
    tools.mkdir('cache/model/ppo')
    tools.mkdir('cache/model/her_q_networks')

def her_ddqn():
    logger.critical('Start HER_DDQN Session')
    config = tools.load_config("configs/config_ddqn_2.yaml")
    # tools.setup_seed(config['RANDOM_SEED'])
    agent = HER_Batch_Trainer.GameAgent(config=config, network='MLP')
    # agent = PHER_Batch_Trainer.GameAgent(config=config, network='MLP')
    agent.batch_train('Default')

if __name__ == '__main__':
    tools.set_logger_level(3)
    init_working_dir()
    her_ddqn()