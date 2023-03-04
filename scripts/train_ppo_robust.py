import sys, os
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/' + '..'))

from trainerV2.Robust_PPO.PPO_continuous_main import PPO_GameAgent
# from scripts.continuous.test_moving import env_list
from scripts.data.test_stationary import env_list
from utils import tools

SEED = 10
RUN_NAME = 'ppo_stationary_robust_l2'

if __name__ == "__main__":    
    save_dir = 'cache/results/' + RUN_NAME
    tools.mkdir(save_dir)
    tools.setup_seed(SEED)
    args = tools.load_config("configs/config_ppo_default.yaml")
    args = tools.dict2class(args)
    args.train_adv = True
    args.delta = 0.05
    args.run_name = RUN_NAME

    PPO_agent = PPO_GameAgent(args=args, output_dir=save_dir, train_mode=True)
    PPO_agent.train(env_list.environment_list[0])
    # PPO_agent.evaluate(env_list.environment_list[0])