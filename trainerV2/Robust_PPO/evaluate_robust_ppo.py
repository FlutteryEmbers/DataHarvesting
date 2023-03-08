import sys, os
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/../..'))

from trainerV2.Robust_PPO.scripts.PPO_continuous_main import PPO_GameAgent
# from scripts.continuous.test_moving import env_list
from trainerV2.Robust_PPO.data.test_stationary import env_list
from utils import tools
from utils import io

SEED = 10030
vanilla_model_name = 'Mar07-07_31-ppo_stationary_vanilla'
robust_model_name = 'Mar06-08_07-ppo_stationary_robust_KL'

vanilla_model_dir = 'cache/results/seed_{}/{}/model/'.format(SEED, vanilla_model_name)
robust_model_dir = 'cache/results/seed_{}/{}/model/'.format(SEED, robust_model_name)
adv_model_dir = 'cache/results/seed_{}/{}/model/'.format(SEED, robust_model_name)
output_dir = 'cache/results/seed_{}/ppo_stationary_evaluation/'.format(SEED)

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
    for i in range(len(evals_modes)):
        mode = evals_modes[i]
        dirs = {}
        dirs['actor'] = eval_info[mode]['model_dir']
        dirs['critic'] = eval_info[mode]['model_dir']
        dirs['adv_net'] = eval_info[mode]['adv_model']

        sub_output_dir = output_dir + mode
        tools.mkdir(sub_output_dir)
        args = tools.load_config("configs/config_ppo_default.yaml")
        args = tools.dict2class(args)
        args.delta = 0.05

        PPO_agent = PPO_GameAgent(args=args, output_dir=sub_output_dir, train_mode=False)
        reward, step = PPO_agent.evaluate_robust(env=env_list.environment_list[0], dirs=dirs, noise_type=eval_info[mode]['noise'])
        summary.append('mode: {}, final rewards: {}, final steps: {}, args:{}, model_dir: {}, adv_dir: {}'.format(mode, reward, step, 
                            eval_info[mode]['model_dir'], eval_info[mode]['adv_model'], args.delta))
        
    io.save_log(output_dir=output_dir, logs=summary)
