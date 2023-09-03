import sys, os
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/../..'))

from trainerV2.Robust_PPO.scripts.PPO_continuous_main import PPO_GameAgent
# from scripts.continuous.test_moving import env_list
from trainerV2.MA_PPO.data.ma_env_list import env_list
from utils import tools
from utils import io

SEED = 10
vanilla_model_name = 'Aug16-15_37-ppo_stationary_vanilla'
# robust_model_name = 'Mar09-17_14-ppo_stationary_robust_KL'

vanilla_model_dir = 'cache/results/seed_{}/config_5s/{}/model/'.format(SEED, vanilla_model_name)
#robust_model_dir = 'cache/results/seed_{}/{}/model/'.format(SEED, robust_model_name)
# adv_model_dir = 'cache/results/seed_{}/{}/model/'.format(SEED, robust_model_name)
output_dir = 'cache/results/seed_{}/ppo_stationary_evaluation/'.format(SEED)

eval_info = {'vanilla_no_noise': {'noise': None, 'model_dir': vanilla_model_dir, 'adv_model': ''}, 
        'vanilla_random_noise': {'noise': 'random', 'model_dir': vanilla_model_dir, 'adv_model': ''}}

evals_modes = ['vanilla_no_noise', 'vanilla_random_noise']

if __name__ == "__main__":
    tools.setup_seed(10)
    summary = []
    for i in range(len(evals_modes)):
        mode = evals_modes[i]
        dirs = {}
        dirs['actor'] = eval_info[mode]['model_dir']
        dirs['critic'] = eval_info[mode]['model_dir']
        dirs['adv_net'] = eval_info[mode]['adv_model']

        args = tools.load_config("configs/config_ppo_default.yaml")
        args = tools.dict2class(args)
        args.delta = 1.0
        output_dir = output_dir + 'delta_{}/'.format(args.delta)
        sub_output_dir = output_dir + mode
        tools.mkdir(sub_output_dir)
        args.adv_lr = 0.01
        args.type_reward = 'Lagrangian'

        PPO_agent = PPO_GameAgent(args=args, output_dir=sub_output_dir, train_mode=False)
        reward, var_reward, step, var_steps = PPO_agent.evaluate_robust(env=env_list[0].environment, dirs=dirs, noise_type=eval_info[mode]['noise'])
        summary.append('mode: {}, final rewards: {}, var_reward: {}, final steps: {}, var_steps:{}, model_dir: {}, adv_dir: {}, perturb:{}'
            .format(mode, round(reward, 4), round(var_reward, 4), round(step, 4), round(var_steps, 4), eval_info[mode]['model_dir'], eval_info[mode]['adv_model'], args.delta))
        
    io.save_log(output_dir=output_dir, logs=summary)