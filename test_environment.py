from environments.environment import Environment

if __name__ == "__main__":
    env = Environment()
    action_n = env.action_space().n()
    board = [[0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0]]
    startAt = [0,0]
    arrivalAt = [1,1]
    data_volumn = [10, 10, 10]
    env.init(board=board, startAt=startAt, arrivalAt=arrivalAt, data_volume=data_volumn)
    state = env.get_state()
    print(state)
