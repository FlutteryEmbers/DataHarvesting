import sys, os
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/' + '..'))

from trainerV2.Robust_PPO.PPO_continuous_main import PPO_GameAgent
# from scripts.continuous.test_moving import env_list
from scripts.data.test_stationary import env_list
from utils import tools

if __name__ == "__main__":
    tools.setup_seed(10)
    save_dir = 'cache/results/ppo_stationary_robust_2'
    tools.mkdir(save_dir)
    args = tools.load_config("configs/config_ppo_default.yaml")
    args = tools.dict2class(args)
    args.train_adv = True
    args.delta = 0.05

    PPO_agent = PPO_GameAgent(args=args, output_dir=save_dir, train_mode=True)
    PPO_agent.train(env_list.environment_list[0])
    # PPO_agent.evaluate(env_list.environment_list[0])