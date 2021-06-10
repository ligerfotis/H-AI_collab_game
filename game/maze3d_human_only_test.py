# Virtual environment
from maze3D_new.Maze3DEnv import Maze3D as Maze3D_v2
from maze3D_new.assets import *
from maze3D_new.utils import save_logs_and_plot

import sys
from experiment import Experiment
from plot_utils.plot_utils import get_config


def main(argv):
    # get configuration
    # sets goal, whether the input is discrete or continuous
    # and the number of episodes and timesteps
    config = get_config(argv[0])

    # creating environment
    maze = Maze3D_v2(config_file=argv[0])

    # create the experiment

    experiment = Experiment(maze, config=config)

    # set the goal
    goal = config["game"]["goal"]

    # Test loop
    experiment.test_human(goal)

    pg.quit()


if __name__ == '__main__':
    main(sys.argv[1:])
    exit(0)
