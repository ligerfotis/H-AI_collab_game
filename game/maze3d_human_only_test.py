import time
from datetime import timedelta
from experiment import Experiment
from maze3D_new.Maze3DEnv import Maze3D
from maze3D_new.assets import *
from rl_models.sac_agent import Agent
from rl_models.sac_discrete_agent import DiscreteSACAgent
from rl_models.utils import get_config, get_plot_and_chkpt_dir, get_sac_agent, get_test_plot_and_chkpt_dir
from maze3D_new.utils import save_logs_and_plot
import sys


def main(argv):
    # get configuration
    # sets goal, whether the input is discrete or continuous
    # and the number of episodes and timesteps
    test_config = get_config(argv[0])

    # creating environment
    maze = Maze3D(config_file=argv[0])

    # create the experiment
    experiment = Experiment(maze, config=test_config, discrete=test_config['train_game_number']['discrete'])

    # set the goal
    goal = test_config["train_game_number"]["goal"]

    # Test mode
    experiment.test_human(goal)

    pg.quit()


if __name__ == '__main__':
    main(sys.argv[1:])
    exit(0)