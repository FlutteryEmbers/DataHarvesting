from utils import graphic
from utils import io
from matplotlib import pyplot as plt
import matplotlib.path as mpath
import matplotlib.patches as mpatches

x_limit = 10
y_limit = 10
tower_locations = [[7, 2], [1, 8]]
start_at = [0, 0]
end_at = [2, 5]
curved_path = False

stopping_time = {}
with open('position_t.txt') as f:
    paths = [[0, 0]]
    for line in f: # read rest of lines
        # print(line)
        x, y, z = line.split()
        point = [float(y), float(z)]
        paths.append(point)
        if tuple(point) in stopping_time:
            stopping_time[tuple(point)] += 1
        else:
            stopping_time[tuple(point)] = 0


# print(paths)

#graphic.plot_result_path(x_limit=10, y_limit=10, 
#    tower_locations=[[0, 1], [4, 7], [9, 3]], paths=path)
plt.figure()
fig, ax = plt.subplots()
ax.axis([-2, x_limit, -2, y_limit])

i = 0
for x, y in tower_locations:
    i += 1
    plt.plot(x, y, marker="x", markersize=10, markeredgecolor="red")
    plt.text(x-0.5, y+0.5, 'tower {}'.format(i), fontsize=18)

x, y = start_at
plt.plot(x, y, marker="o", markersize=10, markeredgecolor="green")
plt.text(x-0.5, y+0.5, 'start', fontsize=18)

x, y = end_at
plt.plot(x, y, marker="o", markersize=10, markeredgecolor="green")
plt.text(x-1.5, y, 'end', fontsize=18)

for point in stopping_time:
    if stopping_time[point] > 0:
        x, y = point
        # plt.text(x+0.5, y, 'stop at {} {} {} steps'.format(x, y, stopping_time[point]), fontsize=8)
        print('stop at {} {} {} steps'.format(x, y, stopping_time[point]))

style = mpath.Path.CURVE4 if curved_path else mpath.Path.LINETO

string_path_data = [(mpath.Path.MOVETO, tuple(paths[0]))]

for i in range(1, len(paths)):
    string_path_data.append((style, tuple(paths[i])))

codes, verts = zip(*string_path_data)
string_path = mpath.Path(verts, codes)
patch = mpatches.PathPatch(string_path, facecolor="none", lw=2)

ax.add_patch(patch)
plt.savefig("out.png")