from environments.runner import DDQN_GameAgent

agent = DDQN_GameAgent()
agent.train(mode='Default', n_games=300)
# agent.run(mode='Default')

# agent.train(mode='DR', n_games=300)
# agent.run(mode='DR')