import numpy as np
import os
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from statistics import stdev, mean
import yaml

from math import sqrt
from pip._vendor.distlib._backport import shutil


def get_config(config_file='config_sac.yaml'):
    try:
        with open(config_file) as file:
            yaml_data = yaml.safe_load(file)
    except Exception as e:
        print('Error reading the config file')

    return yaml_data

def plot_mean_sem(test_games_per_session, data, figure_file, title):
    fig, ax = plt.subplots()
    means = [mean(data[i:i + test_games_per_session]) for i in range(0, len(data), test_games_per_session)]
    sem = [stdev(data[i:i + test_games_per_session]) / sqrt(len(data[i:i + test_games_per_session])) for i in
           range(0, len(data), test_games_per_session)]
    x_axis = [i for i in range(len(means))]
    means = np.asarray(means)
    sem = np.asarray(sem)

    ax.plot(x_axis, means, marker='*')
    ax.fill_between(x_axis, means - sem, means + sem, alpha=0.5)
    plt.plot(x_axis, means)
    plt.title(title)
    plt.savefig(figure_file)


def plot_learning_curve(x, scores, figure_file):
    plt.plot(x, scores)
    plt.title('Total Rewards per Episode')
    plt.savefig(figure_file)


def plot_actions(x, actions, figure_file):
    plt.figure()
    plt.plot(x, actions)
    plt.title('Actions')
    plt.savefig(figure_file)


def plot(data, figure_file, x=None, title=None):
    plt.figure()
    if x is None:
        x = [i + 1 for i in range(len(data))]
    plt.plot(x, data)
    if title:
        plt.title(title)
    plt.savefig(figure_file)


def plot_test_score(data, figure_file, title=None):
    fig, ax = plt.subplots()
    clrs = sns.color_palette("husl", 5)
    means, stds, x_axis = [], [], []
    for i in range(0, len(data), 10):
        means.append(mean(data[i:i + 10]))
        stds.append(stdev(data[i:i + 10]))
        x_axis.append(i + 10)
    means, stds, x_axis = np.asarray(means), np.asarray(stds), np.asarray(x_axis)
    with sns.axes_style("darkgrid"):
        # meanst = np.array(means.ix[i].values[3:-1], dtype=np.float64)
        # sdt = np.array(stds.ix[i].values[3:-1], dtype=np.float64)
        ax.plot(x_axis, means, c=clrs[0])
        ax.fill_between(x_axis, means - stds, means + stds, facecolor='blue', alpha=0.5)
    if title:
        plt.title(title)
    plt.savefig(figure_file)
    # plt.show()


def get_plot_and_chkpt_dir(config, participant_name, config_file_name):
    load_checkpoint, load_checkpoint_name, discrete = [config['game']['load_checkpoint'],
                                                       config['game']['checkpoint_name'], config['SAC']['discrete']]
    mode = str(config['Experiment']['mode'])
    total_number_updates = config['Experiment'][mode]['total_update_cycles']
    participant = participant_name
    learn_every = config['Experiment'][mode]['learn_every_n_games']
    reward_function = config['SAC']['reward_function']
    allocation = config['Experiment']['scheduling']

    alg = 'O_O_a' if config['Experiment']['online_updates'] else 'O_a'
    human_input = 'discrete' if config['game']['discrete_input'] else 'continuous'

    plot_dir = None
    if not load_checkpoint:
        if 'chkpt_dir' in config["SAC"].keys():
            chkpt_dir = 'results/tmp/' + config['SAC']['chkpt_dir']
            plot_dir = 'results/plots/' + config['SAC']['chkpt_dir']
        else:
            chkpt_dir = 'results/tmp/' + mode + '_' + human_input + '_' + alg + '_' + str(int(
                total_number_updates / 1000)) + 'K_every' + str(learn_every) + '_' \
                        + reward_function + '_' + allocation + '_' + participant

            plot_dir = 'results/plots/' + mode + '_' + alg + '_' + str(int(
                total_number_updates / 1000)) + 'K_every' + str(learn_every) + '_' \
                       + reward_function + '_' + allocation + '_' + participant
        i = 1
        while os.path.exists(chkpt_dir + '_' + str(i)):
            i += 1
        os.makedirs(chkpt_dir + '_' + str(i))
        chkpt_dir = chkpt_dir + '_' + str(i)

        j = 1
        while os.path.exists(plot_dir + '_' + str(j)):
            j += 1
        os.makedirs(plot_dir + '_' + str(j))
        plot_dir = plot_dir + '_' + str(j)

        shutil.copy(config_file_name, chkpt_dir)
    else:
        print("Loading Model from checkpoint {}".format(load_checkpoint_name))
        chkpt_dir = load_checkpoint_name

    return chkpt_dir, plot_dir, load_checkpoint_name


def get_test_plot_and_chkpt_dir(test_config):
    # get the config from the train folder
    config = None

    load_checkpoint_name = test_config['checkpoint_name']

    print("Loading Model from checkpoint {}".format(load_checkpoint_name))
    participant = test_config['participant']
    test_plot_dir = 'train_game_number/test_game_number/sac_discrete_' + participant + "/"
    if not os.path.exists(test_plot_dir):
        os.makedirs(test_plot_dir)

    assert os.path.exists(load_checkpoint_name)

    return test_plot_dir, load_checkpoint_name, config


def save_logs_and_plot(experiment, chkpt_dir, plot_dir, max_episodes):
    x = [i + 1 for i in range(len(experiment.score_history))]
    np.savetxt(chkpt_dir + '/scores.csv', np.asarray(experiment.score_history), delimiter=',')

    actions = np.asarray(experiment.action_history)
    # action_main = actions[0].flatten()
    # action_side = actions[1].flatten()
    x_actions = [i + 1 for i in range(len(actions))]
    # Save logs in files
    np.savetxt(chkpt_dir + '/actions.csv', actions, delimiter=',')
    # np.savetxt('tmp/sac_' + timestamp + '/action_side.csv', action_side, delimiter=',')
    np.savetxt(chkpt_dir + '/epidode_durations.csv', np.asarray(experiment.episode_duration_list), delimiter=',')
    np.savetxt(chkpt_dir + '/avg_length_list.csv', np.asarray(experiment.length_list), delimiter=',')
    np.savetxt(chkpt_dir + '/grad_updates_durations.csv', experiment.grad_updates_durations, delimiter=',')

    # np.savetxt(chkpt_dir + '/epidode_durations.csv', np.asarray(experiment.game_duration_list), delimiter=',')

    np.savetxt(chkpt_dir + '/game_step_durations.csv', np.asarray(experiment.step_duration_list), delimiter=',')
    np.savetxt(chkpt_dir + '/online_update_durations.csv', np.asarray(experiment.online_update_duration_list),
               delimiter=',')
    np.savetxt(chkpt_dir + '/total_fps.csv', np.asarray(experiment.total_fps_list), delimiter=',')
    np.savetxt(chkpt_dir + '/train_fps.csv', np.asarray(experiment.train_fps_list), delimiter=',')
    np.savetxt(chkpt_dir + '/test_fps.csv', np.asarray(experiment.test_fps_list), delimiter=',')

    # test_game_number logs
    np.savetxt(chkpt_dir + '/test_episode_duration_list.csv', experiment.test_episode_duration_list, delimiter=',')
    np.savetxt(chkpt_dir + '/test_score_history.csv', experiment.test_score_history, delimiter=',')
    np.savetxt(chkpt_dir + '/test_length_list.csv', experiment.test_length_list, delimiter=',')

    plot_learning_curve(x, experiment.score_history, plot_dir + "/scores.png")
    # plot_actions(x_actions, action_main, plot_dir + "/action_main.png")
    # plot_actions(x_actions, action_side, plot_dir + "/action_side.png")
    plot(experiment.length_list, plot_dir + "/length.png", x=[i + 1 for i in range(max_episodes)])
    plot(experiment.episode_duration_list, plot_dir + "/epidode_durations.png",
         x=[i + 1 for i in range(max_episodes)])
    plot(experiment.grad_updates_durations, plot_dir + "/grad_updates_durations.png",
         x=[i + 1 for i in range(len(experiment.grad_updates_durations))])

    plot(experiment.step_duration_list, plot_dir + "/game_step_durations.png",
         x=[i + 1 for i in range(len(experiment.step_duration_list))])
    plot(experiment.test_step_duration_list, plot_dir + "/test_game_step_durations.png",
         x=[i + 1 for i in range(len(experiment.test_step_duration_list))])

    plot(experiment.online_update_duration_list, plot_dir + "/online_updates_durations.png",
         x=[i + 1 for i in range(len(experiment.online_update_duration_list))])
    plot(experiment.total_fps_list, plot_dir + "/total_fps.png",
         x=[i + 1 for i in range(len(experiment.total_fps_list))])
    plot(experiment.train_fps_list, plot_dir + "/train_fps.png",
         x=[i + 1 for i in range(len(experiment.train_fps_list))])
    plot(experiment.test_fps_list, plot_dir + "/test_fps.png",
         x=[i + 1 for i in range(len(experiment.test_fps_list))])

    # plot test_game_number logs
    x = [i + 1 for i in range(len(experiment.test_length_list))]
    plot_test_score(experiment.test_score_history, plot_dir + "/test_scores.png")
    # plot_actions(x_actions, action_main, plot_dir + "/action_main.png")
    # plot_actions(x_actions, action_side, plot_dir + "/action_side.png")
    plot(experiment.test_length_list, plot_dir + "/test_length.png",
         x=x)
    plot(experiment.test_episode_duration_list, plot_dir + "/test_episode_duration.png",
         x=x)
    try:
        plot_test_score(experiment.score_history, plot_dir + "/test_scores_mean_std.png")
    except:
        print("An exception occurred while plotting")
