from trainer.DDQN.ddqn_main import DDQN_GameAgent
from trainer.PPO.PPO_continuous_main import PPO_GameAgent
from trainer.DDQN_HER import HER_ddqn_main
from trainer.SAC import sac_main
from utils import tools, graphic
from loguru import logger
from trainer.DDPG_HER import ddpg_main

def init_working_dir():
    tools.mkdir('model/q_networks')
    tools.mkdir('model/ppo')
    tools.mkdir('model/her_q_networks')

def ddqn():
    logger.critical('Start DDQN Session')
    config = tools.load_config("configs/config_ddqn.yaml")
    tools.setup_seed(config['RANDOM_SEED'])
    ## network: trainning algorithm using: MLP/CNN network 
    agent = DDQN_GameAgent(config=config, network='MLP')

    ## trainning_mode:
    ## - Default: Trainning for a specific environment;
    ## - DR: Trainning with randomized initial states
    # agent.train(env_type='Default', n_games=2000)
    # agent.evaluate(env_type='Default')
    # agent.train(env_type='DR', n_games=10000)
    agent.evaluate(env_type='DR')

def ppo():
    logger.critical('Start PPO Session')
    mode = 'Default'
    config = tools.load_config("configs/config_ppo_{}.yaml".format(mode.lower()))
    tools.setup_seed(config['random_seed'])
    args = tools.dict2class(config)
    agent = PPO_GameAgent(args = args)
    # agent.train(env_type=mode)
    agent.evaluate_policy(args=args, load_model=mode)

def her_ddqn():
    logger.critical('Start HER_DDQN Session')
    config = tools.load_config("configs/config_ddqn.yaml")
    tools.setup_seed(config['RANDOM_SEED'])
    ## network: trainning algorithm using: MLP/CNN network 
    agent = HER_ddqn_main.GameAgent(config=config, network='MLP')
    # agent.train(env_type='Default', n_games=1000)
    # agent.evaluate(env_type='Default')
    agent.batch_evaluation(env_type='Default')

def ddpg():
    logger.critical('Start DDPG Session')
    config = tools.load_config("configs/config_ddqn.yaml")
    tools.setup_seed(config['RANDOM_SEED'])
    ## network: trainning algorithm using: MLP/CNN network 
    agent = ddpg_main.GameAgent()
    agent.train(n_games=1000)
    # agent.evaluate(env_type='Default')
    # agent.train(env_type='DR', n_games=10000)
    # agent.evaluate(env_type='DR')

def sac():
    logger.critical('Start SAC Session')
    config = tools.load_config("configs/config_ddqn.yaml")
    tools.setup_seed(config['RANDOM_SEED'])
    ## network: trainning algorithm using: MLP/CNN network 
    agent = sac_main.Agent()
    agent.train(1000)


if __name__ == '__main__':
    tools.set_logger_level(3)
    init_working_dir()
    # ddqn()
    # ppo()
    # her_ddqn()
    # ddpg()
    sac()
    # graphic.plot_result_path(x_limit=10, y_limit=10, tower_locations=[[0, 1], [4, 7], [9, 3]], paths=[[0, 0], [1, 2], [2, 3], [7, 9]])

    
   