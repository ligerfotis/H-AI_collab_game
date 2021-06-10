# Virtual environment
from maze3D_new.Maze3DEnv import Maze3D as Maze3D_v2
from maze3D_new.assets import *
from maze3D_new.utils import save_logs_and_plot

# Experiment
from experiment import Experiment

# RL modules
from plot_utils.plot_utils import get_config, get_plot_and_chkpt_dir
from rl_models.utils import get_sac_agent

import sys
import time
from datetime import timedelta

"""
The code of this work is based on the following github repos:
https://github.com/kengz/SLM-Lab
https://github.com/EveLIn3/Discrete_SAC_LunarLander/blob/master/sac_discrete.py
"""

def main(argv):
    # get configuration
    config = get_config(argv[0])

    # creating environment
    maze = Maze3D_v2(config_file=argv[0])

    chkpt_dir, load_checkpoint_name, plot_dir = [None, None, None]
    if config["game"]["save"]:
        # create the checkpoint and plot directories for this experiment
        chkpt_dir, plot_dir, load_checkpoint_name = get_plot_and_chkpt_dir(config)

    # create the SAC agent
    sac = get_sac_agent(config, maze, chkpt_dir)

    # create the experiment
    experiment = Experiment(maze, sac, config=config)

    start_experiment = time.time()

    # max_games_mode runs for maximum number of games
    # max_interactions_mode runs with maximum interactions (human-agent actions)
    loop = config['Experiment']['mode']
    if loop == 'max_games_mode':
        experiment.max_games_mode()
    elif loop == 'max_interactions_mode':
        experiment.max_interactions_mode()
    else:
        print("Unknown training mode")
        exit(1)

    end_experiment = time.time()
    experiment_duration = timedelta(seconds=end_experiment - start_experiment - experiment.duration_pause_total)

    print('Total Experiment time: {}'.format(experiment_duration))

    if config["game"]["save"]:
        # save training logs to a pickle file
        experiment.df.to_pickle(plot_dir + '/training_logs.pkl')

        if not config['game']['test_model']:
            # save rest of the experiment logs and plot them
            save_logs_and_plot(experiment, chkpt_dir, plot_dir, experiment.max_games)
            experiment.save_info(chkpt_dir, experiment_duration, experiment.max_games)
    pg.quit()


if __name__ == '__main__':
    main(sys.argv[1:])
    exit(0)
