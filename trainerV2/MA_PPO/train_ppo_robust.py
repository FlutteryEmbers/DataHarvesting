import sys, os
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/../..'))

from trainerV2.MA_PPO.scripts.PPO_continuous_main import PPO_GameAgent
from trainerV2.MA_PPO.data.ma_env_list import env_list
from utils import tools
import argparse

# 10, 15, 243, 10030, 255000
SEED = 10
RUN_NAME = 'ppo_stationary_robust_KL'

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='RobustPPO', description='train')
    parser.add_argument('--seed', type=int, default=SEED)
    parsed_args = parser.parse_args()

    args = tools.load_config("configs/config_ppo_ma.yaml")
    args = tools.dict2class(args)
    args.train_adv = True
    
    args.delta = 0.05
    args.run_name = RUN_NAME
    args.adv_type = 'KL'
    args.actor_adv_step_size = 30
    args.adv_lr = 0.002
    args.type_reward = 'Lagrangian'

    
    seed_list = [10, 15, 243, 10030, 255000]
    for i in range(len(env_list)):
        for seed in seed_list:
            args.seed = seed
            save_dir = 'cache/results/{}/seed_{}/'.format(env_list[i].instance_name, args.seed)
            tools.mkdir(save_dir)
            tools.setup_seed(seed)
            PPO_agent = PPO_GameAgent(args=args, output_dir=save_dir, train_mode=True)
            PPO_agent.train(env_list[i].environment)
    # PPO_agent.evaluate(env_list.environment_list[0])