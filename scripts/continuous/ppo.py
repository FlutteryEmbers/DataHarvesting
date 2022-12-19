# from trainerV2.PPO_HER.PPO_continuous_main import PPO_GameAgent
from trainerV2.PPO.PPO_continuous_main import PPO_GameAgent
from scripts.continuous.test1 import env_list
from utils import tools


def run():
    tools.mkdir('cache/results/ppo_test')
    args = tools.load_config("configs/config_ppo_default.yaml")
    args = tools.dict2class(args)
    PPO_agent = PPO_GameAgent(args=args, output_dir='cache/results/ppo_test')
    PPO_agent.train(env_list.environment_list[0])