from environments.DQN_Environment import DQN_Environment

if __name__ == "__main__":
    board = [[0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]]

    env = DQN_Environment(board=board)
    action_n = env.action_space().n()
    startAt = [0,0]
    arrivalAt = [1,1]
    data_volumn = [10, 10, 10]
    env.init(startAt=startAt, arrivalAt=arrivalAt, data_volume=data_volumn)
    state = env.get_state()
    print(state)
    # action = env.action_space().sample()
    # print(action)
    action = [1, 0]
    print(env.step(action=action))
    action = [0, 1]
    print(env.step(action=action))
    action = [1, 1]
    print(env.step(action=action))

