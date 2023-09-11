import sys, os
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/../..'))

from trainerV2.MA_PPO.scripts.PPO_continuous_main import PPO_GameAgent
# from scripts.continuous.test_moving import env_list
from trainerV2.MA_PPO.data.ma_env_list import env_list
from utils import tools
from utils import io

SEED = 10
config_name = 'config_5'
vanilla_model_name = 'Aug18-21_02-ppo_stationary_vanilla'
# robust_model_name = 'Aug23-04_49-ppo_stationary_robust_KL' ## for performance
robust_model_name = 'Aug17-22_12-ppo_stationary_robust_KL'

vanilla_model_dir = 'cache/results/{}/seed_{}/{}/model/'.format(config_name, SEED, vanilla_model_name)
robust_model_dir = 'cache/results/{}/seed_{}/{}/model/'.format(config_name, SEED, robust_model_name)
adv_model_dir = 'cache/results/{}/seed_{}/{}/model/'.format(config_name, SEED, robust_model_name)
output_dir = 'cache/results/{}/seed_{}/ppo_stationary_evaluation/'.format(config_name, SEED)

eval_info = {'vanilla_no_noise': {'noise': None, 'model_dir': vanilla_model_dir, 'adv_model': ''}, 
        'vanilla_adv_noise': {'noise': 'adv', 'model_dir': vanilla_model_dir, 'adv_model': adv_model_dir}, 
        'vanilla_random_noise': {'noise': 'random', 'model_dir': vanilla_model_dir, 'adv_model': ''}, 
        'robust_no_noise': {'noise': None, 'model_dir': robust_model_dir, 'adv_model': ''}, 
        'robust_adv_noise': {'noise': 'adv', 'model_dir': robust_model_dir, 'adv_model': adv_model_dir}, 
        'robust_random_noise': {'noise': 'random', 'model_dir': robust_model_dir, 'adv_model': ''}}

evals_modes = ['vanilla_no_noise', 'vanilla_adv_noise', 'vanilla_random_noise', 'robust_no_noise', 'robust_adv_noise', 'robust_random_noise']

if __name__ == "__main__":
    tools.setup_seed(10)
    summary = []
    args = tools.load_config("configs/config_ppo_default.yaml")
    args = tools.dict2class(args)
    args.delta = 1
    args.adv_lr = 0.005
    args.type_reward = 'Lagrangian'
    
    output_dir = output_dir + 'delta_{}/'.format(args.delta)
    
    for i in range(len(evals_modes)):
        mode = evals_modes[i]
        dirs = {}
        dirs['actor'] = eval_info[mode]['model_dir']
        dirs['critic'] = eval_info[mode]['model_dir']
        dirs['adv_net'] = eval_info[mode]['adv_model']

        # sub_output_dir = output_dir + mode
        # tools.mkdir(sub_output_dir)
        sub_output_dir = output_dir + mode
        tools.mkdir(sub_output_dir)

        PPO_agent = PPO_GameAgent(args=args, output_dir=sub_output_dir, train_mode=False)
        reward, var_reward, min_reward, step, var_steps, max_steps = PPO_agent.evaluate_robust(env=env_list[0].environment, dirs=dirs, noise_type=eval_info[mode]['noise'],  seed = 1000)
        #summary.append('mode: {}, final rewards: {}, var_reward: {}, final steps: {}, var_steps:{}, model_dir: {}, adv_dir: {}, perturb:{}'
        #    .format(mode, round(reward, 4), round(var_reward, 4), round(step, 4), round(var_steps, 4), eval_info[mode]['model_dir'], eval_info[mode]['adv_model'], args.delta))
        summary.append([mode, round(reward, 4), round(var_reward, 4), round(min_reward, 4), round(step, 4), round(var_steps, 4), round(max_steps, 4), eval_info[mode]['model_dir'], eval_info[mode]['adv_model'], args.delta])
        header = ['mode', 'final rewards', 'var_reward', 'min_reward', 'final steps', 'var_steps', 'max_steps','model_dir', 'adv_dir', 'perturb']
    # io.save_log(output_dir=output_dir, logs=summary)
    io.save_csv(output_dir=output_dir, name='evaluation_result', headers=header, logs=summary)
