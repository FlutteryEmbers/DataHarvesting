import sys, os
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/../..'))

from trainerV2.Robust_PPO.scripts.PPO_continuous_main import PPO_GameAgent
# from scripts.continuous.test_moving import env_list
from trainerV2.Robust_PPO.data.test_stationary import env_list
from utils import tools

# 10, 15, 243, 10030, 255000
SEED = 10
RUN_NAME = 'ppo_stationary_vanilla'

if __name__ == "__main__":
    args = tools.load_config("configs/config_ppo_default.yaml")
    args = tools.dict2class(args)
    args.seed = SEED
    args.train_adv = False
    args.delta = 0
    args.run_name = RUN_NAME
    args.type_reward = 'Lagrangian'
    
    save_dir = 'cache/results/seed_{}/'.format(SEED)
    tools.mkdir(save_dir)
    tools.setup_seed(SEED)

    PPO_agent = PPO_GameAgent(args=args, output_dir=save_dir, train_mode=True)
    PPO_agent.train(env_list.environment_list[0])
    # PPO_agent.evaluate(env_list.environment_list[0])