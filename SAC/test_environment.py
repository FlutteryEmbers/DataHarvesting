from environments.DQN_Environment import DQN_Environment

if __name__ == "__main__":
    board = [[0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]]

    env = DQN_Environment(board=board)
    action_n = env.get_action_space().n()
    print(env.action_space.n())
    startAt = [0,0]
    arrivalAt = [1,1]
    data_volumn = [10, 10, 10]
    env.init(startAt=startAt, arrivalAt=arrivalAt, data_volume=data_volumn)
    state = env.get_state()
    print(state)
    # action = env.action_space().sample()
    # print(action)
    '''
    action = [1, 0]
    print(env.step(action=action))
    action = [1, 0]
    print(env.step(action=action))
    action = [1, 0]
    print(env.step(action=action))
    action = [1, 0]
    print(env.step(action=action))
    action = [1, 0]
    state, _, _, _ = env.step(action=action)
    '''
    # print(env.step(action=action))
    print(env.get_action_space().get_available_actions(state.current_position))
    print(env.get_action_space().sample(state.current_position))

