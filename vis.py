
import matplotlib.pyplot as plt


def plot(trajectory, trajectory_support_points, obstacle_tracking, nextTarget, currentEE):
    plt.clf()

    plt.plot(obstacle_tracking.kf["a"].x[0], obstacle_tracking.kf["a"].x[1], 'o', color='red')
    plt.plot(obstacle_tracking.kf["b"].x[0], obstacle_tracking.kf["b"].x[1], 'o', color='red')

    if len(trajectory_support_points)  > 0: plt.plot(trajectory_support_points[:, 0], trajectory_support_points[:, 1], color="grey")
    if len(trajectory)  > 0: plt.plot(trajectory[:, 0], trajectory[:, 1], color="black")

    plt.plot(currentEE[0], currentEE[1], 'o', color='yellow')

    plt.plot(nextTarget[0], nextTarget[1], 'o', color='orange')


    plt.draw()
    plt.pause(0.001)