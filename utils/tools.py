def ListAddition(L1, L2):
    next_position = []
    for (c1, c2) in zip(L1, L2):
        next_position.append(c1+c2)
    return next_position