import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.patches as mpatches
import matplotlib.cm as cm

def plot_learning_curve(x, scores, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    plt.plot(x, running_avg)
    plt.title('Running average of previous 100 scores')
    plt.savefig(figure_file)

def plot_curve(x, y, figure_file):
    plt.figure()
    plt.plot(x, y)
    # plt.title('Running average of previous 100 scores')
    plt.savefig(figure_file)
    plt.close('all')

def plot_result_path(x_limit, y_limit, tower_locations, paths, curved_path = False):
    # w = 4
    # h = 3
    # d = 70
    # plt.figure(figsize=(w, h), dpi=d)
    plt.figure()
    fig, ax = plt.subplots()
    ax.axis([-2, x_limit, -2, y_limit])

    i = 0
    for x, y in tower_locations:
        i += 1
        plt.plot(x, y, marker="x", markersize=10, markeredgecolor="red")
        plt.text(x-0.5, y+0.5, 'tower {}'.format(i), fontsize=8)

    style = mpath.Path.CURVE4 if curved_path else mpath.Path.LINETO

    string_path_data = [(mpath.Path.MOVETO, tuple(paths[0]))]

    for i in range(1, len(paths)):
        string_path_data.append((style, tuple(paths[i])))

    codes, verts = zip(*string_path_data)
    string_path = mpath.Path(verts, codes)
    patch = mpatches.PathPatch(string_path, facecolor="none", lw=2)

    ax.add_patch(patch)
    plt.savefig("out.png")
    plt.close('all')

def plot_path(x_limit, y_limit, start_at, end_at, tower_locations, agent_paths, signal_range, dir):
    tower_range = signal_range
    plt.figure()
    fig, ax = plt.subplots()
    ax.axis([-2, x_limit, -2, y_limit])

    color_wheel = ["#cc99ff50", "#ff99ff50", "#ffb36650", "#ff4d9450", "#80b3ff50"]
    cmap = cm.get_cmap(name='rainbow')
    for i, (x, y) in enumerate(tower_locations):
        plt.plot(x, y, marker="*", markersize=10, markeredgecolor="red", markerfacecolor='red')
        circle = plt.Circle( (x, y ), tower_range[i], color=color_wheel[i])
        ax.add_artist( circle )
        plt.text(x-3, y+0.5, 'target {}'.format(i+1), fontsize=15)

    for i, (x, y) in enumerate(start_at):
        #x, y = start_at
        plt.plot(x, y, marker="o", markersize=10, markeredgecolor=cmap(i*10+5), markerfacecolor=cmap(i*35+5))
        # plt.text(x-2, y-1, '{}'.format(i), fontsize=15)
        # plt.plot(x, y, marker="o", markersize=10, markeredgecolor="#003d99", markerfacecolor='#003d99')
        # plt.text(x-2, y-1, 'start', fontsize=15)

    for i, (x, y) in enumerate(end_at):
        plt.plot(x, y, marker="X", markersize=10, markeredgecolor=cmap(i*10+5), markerfacecolor=cmap(i*35+5))
        # plt.text(x-2, y, 'end', fontsize=15)

    '''
    for point in stopping_time:
        if stopping_time[point] > 0:
            x, y = point
            # plt.text(x+0.5, y, 'stop at {} {} {} steps'.format(x, y, stopping_time[point]), fontsize=8)
            # plt.plot(x, y, marker="*", markersize=10, markeredgecolor="green")
            # plt.text(x+0.5, y, 'stop {} steps'.format(stopping_time[point]), fontsize=12)
            print('stop at {} {} {} steps'.format(x, y, stopping_time[point]))
    '''

    curved_path = False
    style = mpath.Path.CURVE4 if curved_path else mpath.Path.LINETO
    for index, paths in enumerate(agent_paths):
        string_path_data = [(mpath.Path.MOVETO, tuple(start_at[index]))]

        for i in range(0, len(paths)):
            string_path_data.append((style, tuple(paths[i].tolist())))

        codes, verts = zip(*string_path_data)
        string_path = mpath.Path(verts, codes)
        patch = mpatches.PathPatch(string_path, facecolor="none", lw=2, edgecolor=cmap(index*35+5))

        ax.add_patch(patch)
    ax.set_aspect(1)
    # plt.savefig(dir_i + "path.eps", format='eps')
    plt.savefig(dir + "path.png", bbox_inches='tight', pad_inches=0, format='png', dpi=300)
    plt.close('all')


def plot_robust_radius(name_arr, noise_level, mean_arr, std_arr):
    # trail_names = ['weights_normalization/ci50', 'weights_normalization/ci200', 'weights_normalization/ci487', 'weights_normalization/ci4790', 'weights_normalization/uncap', 'vanilla/ci1327']
    colors = ['#ff8c1a', '#0066ff', '#d4ac0d', '#922b21', '#76448a', '#117a65', '#3498db']
    face_colors = ['#ffb366', '#66a3ff', '#f7dc6f', '#d98880', '#af7ac5', '#73c6b6', '#a9cce3']

    seeds = [10]
    fig, ax = plt.subplots()
    # ax.set_ylim(bottom=50, top=1500)
    for i in range(len(name_arr)):
        ax.plot(noise_level, mean_arr[i], label=name_arr[i], color=colors[i])
        ax.fill_between(noise_level, mean_arr[i]-std_arr[i], mean_arr[i]+std_arr[i], alpha=0.3, facecolor=colors[i])

    plt.legend(loc='upper right')
    plt.xlabel("noise_level")
    # plt.yscale('log')

    plt.ylabel("steps")
    plt.savefig('{}.png'.format('cache/robust_radius'), bbox_inches='tight', pad_inches=0.1, format='png', dpi=300)