from trainerV2.Robust_PPO.PPO_continuous_main import PPO_GameAgent
# from scripts.continuous.test_moving import env_list
from scripts.continuous.test_stationary import env_list
from utils import tools


def run():
    tools.setup_seed(10)
    save_dir = 'cache/results/ppo_stationary_robust'
    tools.mkdir(save_dir)
    args = tools.load_config("configs/config_ppo_default.yaml")
    args = tools.dict2class(args)
    PPO_agent = PPO_GameAgent(args=args, output_dir=save_dir)
    PPO_agent.train(env_list.environment_list[0])
    # PPO_agent.evaluate(env_list.environment_list[0])