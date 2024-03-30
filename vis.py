
import matplotlib.pyplot as plt

sim = None
fig, axs = plt.subplots(1,4)
plot_arr = []
for i in range(17):
    plot_arr.append([])

fig.suptitle("Position Estimates")
axs[0].set_title("x")
axs[1].set_title("y")
axs[2].set_title("z")
axs[3].set_title("size")

plot_title = ["time", "o1_x", "o1_y", "o1_z", "o1_size", "o2_x", "o2_y", "o2_z", "o2_size", "kfa_x", "kfa_y", "kfa_z", "kfa_size", "kfb_x", "kfb_y", "kfb_z", "kfb_size"]


def plot_kf_error(time_step, obstacle_tracking):
    global fig, axs, plot_arr

    plot_arr[0].append(time_step)

    for j, obstacle in enumerate(sim.obstacles):
        #print("Ostacle pos", obstacle.get_pos(), obstacle.scaling)
        for i in range(3):
            plot_arr[1+4*j+i].append(obstacle.get_pos()[i])
        plot_arr[1+4*j+3].append(obstacle.scaling/2)

    for i in range(4):
       plot_arr[9+i].append(obstacle_tracking.kf["a"].x[i])
       plot_arr[13+i].append(obstacle_tracking.kf["b"].x[i])

    for ax in axs:
       ax.clear()

    for i in range(1, len(plot_arr)):
       if len(plot_arr[0]) == len(plot_arr[i]):
           axs[(i-1)%4].plot(plot_arr[0], plot_arr[i], label=plot_title[i])

    for ax in axs:
       ax.legend()

    plt.draw()
    plt.pause(0.001)



def plot_trajectory(trajectory, trajectory_support_points, obstacle_tracking, nextTarget, currentEE):
    plt.clf()

    for obstacle in obstacle_tracking:
        plt.plot(obstacle[0], obstacle[1], 'o', color='red')

    for obstacle in sim.obstacles:
        plt.plot(obstacle.get_pos()[0], obstacle.get_pos()[1], 'o', color='purple')

    if len(trajectory_support_points)  > 0: plt.plot(trajectory_support_points[:, 0], trajectory_support_points[:, 1], color="grey")
    if len(trajectory)  > 0: plt.plot(trajectory[:, 0], trajectory[:, 1], color="black")

    plt.plot(currentEE[0], currentEE[1], 'o', color='yellow')

    plt.plot(nextTarget[0], nextTarget[1], 'o', color='orange')


    plt.draw()
    plt.pause(0.001)

