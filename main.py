from trainer.Q_Learning.ddqn_main import DDQN_GameAgent
from trainer.PPO.PPO_continuous_main import PPO_GameAgent
from utils import tools

def init_working_dir():
    tools.mkdir('model/q_networks')
    tools.mkdir('model/ppo')
    tools.mkdir('results/Default')
    tools.mkdir('results/DR')

def ddqn():
    print('=====================DDQN======================')
    config = tools.load_config("configs/config_ddqn.yaml")
    tools.setup_seed(config['RANDOM_SEED'])
    ## network: trainning algorithm using: MLP/CNN network 
    agent = DDQN_GameAgent(config=config, network='MLP')

    ## trainning_mode:
    ## - Default: Trainning for a specific environment;
    ## - DR: Trainning with randomized initial states
    # agent.train(env_type='Default', n_games=100)
    agent.evaluate(env_type='Default')
    # agent.train(env_type='DR', n_games=100)
    # agent.evaluate(env_type='DR')

def ppo():
    print('=====================PPO=======================')
    config = tools.load_config("configs/config_ppo_default.yaml")
    tools.setup_seed(config['random_seed'])
    args = tools.dict2class(config)
    agent = PPO_GameAgent(args = args)
    # agent.train(env_type='Default')
    agent.evaluate_policy(args=args, load_model='Default')
    # agent.train(env_type='DR')
    # agent.evaluate_policy(args=args, load_model='DR')

if __name__ == '__main__':
    init_working_dir()
    # ddqn()
    ppo()

    
   