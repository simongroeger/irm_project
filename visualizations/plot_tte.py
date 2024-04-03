import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("visualizations/trajectory_tracking_error.txt", delimiter=",")

tte_d = {}
steps_d = {}

for line in data:
    if not 100*line[0] in tte_d:
        tte_d[100*line[0]] = []
        steps_d[100*line[0]] = []

    tte_d[100*line[0]].append(1000*line[1])
    steps_d[100*line[0]].append(line[2] / 240)

lookahead_distance = np.array(list(tte_d.keys()))
tte = []
for k in tte_d.keys():
    arr = tte_d[k]
    m = np.mean(np.array(arr))
    tte.append(m)
tte = np.array(tte)
steps = []
for k in steps_d.keys():
    arr = steps_d[k]
    m = np.mean(np.array(arr))
    steps.append(m)
steps = np.array(steps)


#lookahead_distance = np.array([1, 2, 3, 5, 7, 9, 11, 13, 15, 17, 19])
#tte = np.array([1.24, 0.96, 1.02, 1.64, 2.78, 4.42, 6.24, 8.56, 10.91, 13.89, 16.61])
#steps = np.array([500, 260, 190, 140, 120, 97, 89, 87, 85, 80, 75]) / 240 

print(lookahead_distance)
print(tte)
print(steps)

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
ax1.set_xlabel('Lookahead distance (cm)')

color = 'tab:red'
ax2.set_ylabel('Trajectory tracking error (mm)', color=color)
ax2.plot(lookahead_distance, tte, color=color)
ax2.tick_params(axis='y', labelcolor=color)


color = 'tab:blue'
ax1.set_ylabel('Duration (s)', color=color)  # we already handled the x-label with ax1
ax1.plot(lookahead_distance, steps, color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax1.set_ylim([0.2, 2.2])
#ax2.set_ylim([0.8, 17.8])
ax2.set_xlim([1, 19])

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()


