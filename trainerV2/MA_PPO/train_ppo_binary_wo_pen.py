import sys, os
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/../..'))

from trainerV2.MA_PPO.scripts.PPO_continuous_main import PPO_GameAgent
# from scripts.continuous.test_moving import env_list
# from trainerV2.MA_PPO.data.ma_env import env_list
from trainerV2.MA_PPO.data.ma_env_list import env_list
from utils import tools

# 10, 15, 243, 10030, 255000
SEED = 10
RUN_NAME = 'ppo_stationary_vanilla'

if __name__ == "__main__":
    for j in range(1, 4):
        args = tools.load_config("configs/config_ppo_ma{}.yaml".format(j))
        args = tools.dict2class(args)
        args.seed = SEED
        args.train_adv = False
        args.delta = 0
        args.run_name = RUN_NAME
        args.type_reward = 'MA_Binary_WO_Pen'
        
        # seed_list = [10, 15, 243, 10030, 255000]
        seed_list = [10, 15, 243]
        for i in range(len(env_list)):
            for seed in seed_list:
                args.seed = seed
                save_dir = 'cache/results/{}_binary_wo_pen/config_{}/seed_{}/'.format(env_list[i].instance_name, j, args.seed)
                tools.mkdir(save_dir)
                tools.setup_seed(seed)
                PPO_agent = PPO_GameAgent(args=args, output_dir=save_dir, train_mode=True)
                PPO_agent.train(env_list[i].environment)
