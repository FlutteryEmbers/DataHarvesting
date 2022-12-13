from textwrap import fill
from matplotlib import pyplot as plt
import matplotlib.path as mpath
import matplotlib.patches as mpatches
from info import env_list

curved_path = False

dir = env_list.instance_name
# dir = dir + "/batch_train_ddqn_her/"
dir = dir + "/batch_train_ddqn/"
for i in range(len(env_list.environment_list)):
    env = env_list.environment_list[i]
    dir_i = dir + "/{}/random_seed_0/".format(i)
    x_limit = env.x_limit
    y_limit = env.y_limit
    tower_locations = env.tower_locations
    start_at = env.start_at
    end_at = env.arrival_at
    tower_range = env.signal_range

    stopping_time = {}
    with open(dir_i + 'position_t.txt') as f:
        paths = [start_at]
        for line in f: # read rest of lines
            # print(line)
            x, y, z = line.split()
            point = [float(y), float(z)]
            paths.append(point)
            if tuple(point) in stopping_time:
                stopping_time[tuple(point)] += 1
            else:
                stopping_time[tuple(point)] = 0


    plt.figure()
    fig, ax = plt.subplots()
    ax.axis([-2, x_limit, -2, y_limit])

    i = 0
    color_wheel = ["#cc99ff50", "#ff99ff50", "#ffb36650", "#ff4d9450", "#80b3ff50"]
    for x, y in tower_locations:
        plt.plot(x, y, marker="*", markersize=10, markeredgecolor="red", markerfacecolor='red')
        circle = plt.Circle( (x, y ), tower_range[i], color=color_wheel[i] )
        ax.add_artist( circle )
        i += 1
        plt.text(x-3, y+0.5, 'target {}'.format(i), fontsize=15)

    x, y = start_at
    plt.plot(x, y, marker="o", markersize=10, markeredgecolor="#003d99", markerfacecolor='#003d99')
    plt.text(x-2, y-1, 'start', fontsize=15)

    x, y = end_at
    plt.plot(x, y, marker="o", markersize=10, markeredgecolor="#003d99", markerfacecolor='#003d99')
    plt.text(x-2, y, 'end', fontsize=15)

    for point in stopping_time:
        if stopping_time[point] > 0:
            x, y = point
            # plt.text(x+0.5, y, 'stop at {} {} {} steps'.format(x, y, stopping_time[point]), fontsize=8)
            # plt.plot(x, y, marker="*", markersize=10, markeredgecolor="green")
            # plt.text(x+0.5, y, 'stop {} steps'.format(stopping_time[point]), fontsize=12)
            print('stop at {} {} {} steps'.format(x, y, stopping_time[point]))

    style = mpath.Path.CURVE4 if curved_path else mpath.Path.LINETO

    string_path_data = [(mpath.Path.MOVETO, tuple(paths[0]))]

    for i in range(1, len(paths)):
        string_path_data.append((style, tuple(paths[i])))

    codes, verts = zip(*string_path_data)
    string_path = mpath.Path(verts, codes)
    patch = mpatches.PathPatch(string_path, facecolor="none", lw=2)

    ax.add_patch(patch)
    ax.set_aspect(1)
    # plt.savefig(dir_i + "path.eps", format='eps')
    plt.savefig(dir_i + "path.png", bbox_inches='tight', pad_inches=0, format='png', dpi=300)
