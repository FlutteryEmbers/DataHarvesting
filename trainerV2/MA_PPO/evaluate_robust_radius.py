import sys, os
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/../..'))

from trainerV2.MA_PPO.scripts.PPO_continuous_main import PPO_GameAgent
# from scripts.continuous.test_moving import env_list
from trainerV2.MA_PPO.data.ma_env_list import env_list
from utils import tools, graph
from utils import io
import numpy as np

SEED = 10
config_name = 'config_5'
vanilla_model_name = 'Aug18-21_02-ppo_stationary_vanilla'
robust_model_name = 'Aug23-04_49-ppo_stationary_robust_KL' ## for performance
# robust_model_name = 'Aug17-22_12-ppo_stationary_robust_KL'

vanilla_model_dir = 'cache/results/{}/seed_{}/{}/model/'.format(config_name, SEED, vanilla_model_name)
robust_model_dir = 'cache/results/{}/seed_{}/{}/model/'.format(config_name, SEED, robust_model_name)
adv_model_dir = 'cache/results/{}/seed_{}/{}/model/'.format(config_name, SEED, robust_model_name)
output_dir = 'cache/results/{}/seed_{}/ppo_stationary_evaluation/'.format(config_name, SEED)

if __name__ == "__main__":
    tools.setup_seed(10)
    summary = []
    args = tools.load_config("configs/config_ppo_default.yaml")
    args = tools.dict2class(args)
    # args.delta = 1
    args.adv_lr = 0.005
    args.type_reward = 'Lagrangian'
    
    # output_dir = output_dir + 'delta_{}/'.format(args.delta)
    noise_level = np.linspace(0.0, 0.2, num=101)
    vanilla_mean = []
    vanilla_std = []
    robust_mean = []
    robust_std = []

    for i in range(len(noise_level)):
        print('vanilla {}'.format(i))
        dirs = {}
        dirs['actor'] = vanilla_model_dir
        dirs['critic'] = vanilla_model_dir

        args.delta = noise_level[i]
        PPO_agent = PPO_GameAgent(args=args, output_dir=vanilla_model_dir, train_mode=False)
        reward, var_reward, min_reward, step, var_steps, max_steps = PPO_agent.evaluate_robust(env=env_list[0].environment, dirs=dirs, noise_type='random',  seed = 1000, plot=False)
        vanilla_mean.append(step)
        vanilla_std.append(var_steps)
        print(step, var_steps)

    for i in range(len(noise_level)):
        print('robust {}'.format(i))
        dirs = {}
        dirs['actor'] = robust_model_dir
        dirs['critic'] = robust_model_dir

        args.delta = noise_level[i]
        PPO_agent = PPO_GameAgent(args=args, output_dir=robust_model_dir, train_mode=False)
        reward, var_reward, min_reward, step, var_steps, max_steps = PPO_agent.evaluate_robust(env=env_list[0].environment, dirs=dirs, noise_type='random',  seed = 1000, plot=False)
        robust_mean.append(step)
        robust_std.append(var_steps)
        print(step, var_steps)


    graph.plot_robust_radius(name_arr=['non-smooth', 'smooth'], noise_level=noise_level, mean_arr=[np.array(vanilla_mean), np.array(robust_mean)], std_arr=[np.array(vanilla_std), np.array(robust_std)])
        #summary.append('mode: {}, final rewards: {}, var_reward: {}, final steps: {}, var_steps:{}, model_dir: {}, adv_dir: {}, perturb:{}'
        #    .format(mode, round(reward, 4), round(var_reward, 4), round(step, 4), round(var_steps, 4), eval_info[mode]['model_dir'], eval_info[mode]['adv_model'], args.delta))
        
        