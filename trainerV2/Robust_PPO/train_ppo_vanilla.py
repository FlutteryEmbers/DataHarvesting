import sys, os
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/' + '..'))

from trainerV2.Robust_PPO.PPO_continuous_main import PPO_GameAgent
# from scripts.continuous.test_moving import env_list
from scripts.data.test_stationary import env_list
from utils import tools

SEED = 15
RUN_NAME = 'ppo_stationary_vanilla'
if __name__ == "__main__":
    tools.setup_seed(SEED)
    save_dir = 'cache/results/'
    tools.mkdir(save_dir)
    args = tools.load_config("configs/config_ppo_default.yaml")
    args = tools.dict2class(args)
    args.seed = SEED
    args.train_adv = False
    args.delta = 0
    args.run_name = RUN_NAME
    PPO_agent = PPO_GameAgent(args=args, output_dir=save_dir, train_mode=True)
    PPO_agent.train(env_list.environment_list[0])
    # PPO_agent.evaluate(env_list.environment_list[0])