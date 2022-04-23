from environments.environment import Environment

if __name__ == "__main__":
    env = Environment()
    action_n = env.action_space().n()
    print(action_n)
