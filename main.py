from environments.runner import DDQN_GameAgent
from utils import tools

tools.mkdir('model/q_networks')
tools.mkdir('results/Default')
tools.mkdir('results/DR')

agent = DDQN_GameAgent()
# agent.train(mode='Default', n_games=500)
agent.run(mode='Default')

# agent.train(mode='DR', n_games=3000)
# agent.run(mode='DR')