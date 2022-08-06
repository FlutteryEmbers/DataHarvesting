import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.patches as mpatches

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