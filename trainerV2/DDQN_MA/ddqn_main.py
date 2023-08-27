import sys, os
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/../..'))

from trainerV2.DDQN_MA.ddqn import ddqn_batch_trainer
from utils import graph, tools
from loguru import logger

if __name__ == '__main__':
    logger.critical('Start DDQN Session')
    config = tools.load_config("configs/config_ddqn.yaml")
    agent = ddqn_batch_trainer.GameAgent(config=config, network='MLP')
    agent.batch_train('Default')