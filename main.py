from trainer.DDQN_HER import HER_Batch_Trainer
# from trainer.DDQN import ddqn_batch_trainer
# from trainer.DDQN_PHER import PHER_Batch_Trainer
from utils import tools, graphic
from loguru import logger

def init_working_dir():
    tools.mkdir('model/q_networks')
    tools.mkdir('model/ppo')
    tools.mkdir('model/her_q_networks')

def ddqn():
    logger.critical('Start DDQN Session')
    config = tools.load_config("configs/config_ddqn.yaml")
    agent = ddqn_batch_trainer.GameAgent(config=config, network='MLP')
    agent.batch_train('Default')

def her_ddqn():
    logger.critical('Start HER_DDQN Session')
    config = tools.load_config("configs/config_ddqn.yaml")
    # tools.setup_seed(config['RANDOM_SEED'])
    agent = HER_Batch_Trainer.GameAgent(config=config, network='MLP')
    agent.batch_train('Default') 
    # agent = PHER_Batch_Trainer.GameAgent(config=config, network='MLP')
    # agent.batch_train('Default')  

if __name__ == '__main__':
    tools.set_logger_level(3)
    init_working_dir()
    # ddqn()
    her_ddqn()

    
   