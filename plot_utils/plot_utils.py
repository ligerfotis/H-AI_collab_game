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
    sem = [stdev(data[i:i + test_games_per_session]) / sqrt(len(data[i:i + test_games_per_session])) for i in range(0, len(data), test_games_per_session)]
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


def plot_test_score(data, figure_file, average_cycle, title=None):
    fig, ax = plt.subplots()
    clrs = sns.color_palette("husl", 5)
    means, stds, x_axis = [], [], []
    for i in range(0, len(data), average_cycle):
        means.append(mean(data[i:i + average_cycle]))
        stds.append(stdev(data[i:i + average_cycle]))
        x_axis.append(i + average_cycle)
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
            header = mode + '_' + human_input + '_' + alg + '_' + str(int(
                total_number_updates / 1000)) + 'K_every' + str(learn_every) + '_' \
                        + reward_function + '_' + allocation + '_' + participant
            chkpt_dir = 'results/tmp/' + header

            plot_dir = 'results/plots/' + header
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


def save_metrics(experiment, save_dir):
    """
    Saves the metrics related to the experiment
    :param experiment: the experiment instance
    :param save_dir: the directory to save the metrics in
    :return:
    """
    # Train/TestScores(.csv)
    np.savetxt(save_dir + '/train_scores.csv', experiment.train_scores, delimiter=',')
    np.savetxt(save_dir + '/test_scores.csv', experiment.test_scores, delimiter=',')

    # Train/Test Game durations(.csv)
    np.savetxt(save_dir + '/train_game_durations.csv', experiment.train_game_durations, delimiter=',')
    np.savetxt(save_dir + '/test_game_durations.csv', experiment.test_game_durations, delimiter=',')

    # Offline Gradient Updates Durations(.csv)
    np.savetxt(save_dir + '/offline_update_durations.csv', experiment.offline_update_durations, delimiter=',')

    # Online Gradient Updates Durations(.csv)
    np.savetxt(save_dir + '/online_update_durations.csv', experiment.online_update_durations, delimiter=',')

    # Train/Test Reward per game
    np.savetxt(save_dir + '/train_rewards.csv', experiment.train_rewards, delimiter=',')
    np.savetxt(save_dir + '/test_rewards.csv', experiment.test_rewards, delimiter=',')

    # Train/Test Game Success Rate
    np.savetxt(save_dir + '/train_game_success_rates.csv', experiment.train_game_success_rates, delimiter=',')
    np.savetxt(save_dir + '/test_game_success_rates.csv', experiment.test_game_success_rates, delimiter=',')

    # Train/Test Agent-Human Action pairs
    # Train/Test Distance Travelled by ball per game
    np.savetxt(save_dir + '/train_distance_traveled.csv', experiment.train_distance_traveled, delimiter=',')
    np.savetxt(save_dir + '/test_distance_traveled.csv', experiment.test_distance_traveled, delimiter=',')

    # Train/Test Dataframe of transitions
    # Train/Test step duration per game
    # q1 , q2, policy and entropy losses

    # Train/Test fps
    np.savetxt(save_dir + '/total_fps.csv', experiment.total_fps_list, delimiter=',')
    np.savetxt(save_dir + '/train_fps.csv', experiment.train_fps_list, delimiter=',')
    np.savetxt(save_dir + '/test_fps.csv', experiment.test_fps_list, delimiter=',')

    # Train/Test Steps per Game
    np.savetxt(save_dir + '/train_steps_per_game.csv', experiment.train_steps_per_game, delimiter=',')
    np.savetxt(save_dir + '/test_steps_per_game.csv', experiment.test_steps_per_game, delimiter=',')

    # Internet Delay
    np.savetxt(save_dir + '/internet_delay.csv', experiment.env.internet_delay, delimiter=',')

    # learning metrics
    np.savetxt(save_dir + '/policy_loss_list.csv', experiment.policy_loss_list, delimiter=',')
    np.savetxt(save_dir + '/q1_loss_list.csv', experiment.q1_loss_list, delimiter=',')
    np.savetxt(save_dir + '/q2_loss_list.csv', experiment.q2_loss_list, delimiter=',')
    np.savetxt(save_dir + '/entropy_loss_list.csv', experiment.entropy_loss_list, delimiter=',')

def plot_metrics(experiment, plot_dir):
    """
    Plots the metrics related to the experiment
    :param experiment: the experiment instance
    :param plot_dir: the directory to save the ploted metrics in
    :return:
    """
    # plot the score mean and sem
    plot_mean_sem(experiment.train_interval, experiment.train_scores, plot_dir + "/train_score_mean_sem.png",
                  "Training Scores")
    plot_mean_sem(experiment.test_max_games, experiment.test_scores, plot_dir + "/test_score_mean_sem.png",
                  "Testing Scores")

    plot(experiment.train_game_durations, plot_dir + "/train_game_durations.png",
         x=[i + 1 for i in range(len(experiment.train_game_durations))])
    plot(experiment.test_game_durations, plot_dir + "/test_game_durations.png",
         x=[i + 1 for i in range(len(experiment.test_game_durations))])

    plot(experiment.train_step_duration_list, plot_dir + "/train_game_step_durations.png",
         x=[i + 1 for i in range(len(experiment.train_step_duration_list))])
    plot(experiment.test_step_duration_list, plot_dir + "/test_game_step_durations.png",
         x=[i + 1 for i in range(len(experiment.test_step_duration_list))])

    plot(experiment.offline_update_durations, plot_dir + "/offline_grad_updates_durations.png",
         x=[i + 1 for i in range(len(experiment.offline_update_durations))])

    plot(experiment.online_update_durations, plot_dir + "/online_update_durations.png",
         x=[i + 1 for i in range(len(experiment.online_update_durations))])

    plot(experiment.total_fps_list, plot_dir + "/total_fps.png",
         x=[i + 1 for i in range(len(experiment.total_fps_list))])
    plot(experiment.train_fps_list, plot_dir + "/train_fps.png",
         x=[i + 1 for i in range(len(experiment.train_fps_list))])
    plot(experiment.test_fps_list, plot_dir + "/test_fps.png",
         x=[i + 1 for i in range(len(experiment.test_fps_list))])

    plot(experiment.train_rewards, plot_dir + "/train_rewards.png",
         x=[i + 1 for i in range(len(experiment.train_rewards))])
    plot(experiment.test_rewards, plot_dir + "/test_rewards.png",
         x=[i + 1 for i in range(len(experiment.test_rewards))])

    plot(experiment.train_game_success_rates, plot_dir + "/train_game_success_rates.png",
         x=[i + 1 for i in range(len(experiment.train_game_success_rates))])
    plot(experiment.test_game_success_rates, plot_dir + "/test_game_success_rates.png",
         x=[i + 1 for i in range(len(experiment.test_game_success_rates))])

    plot(experiment.env.internet_delay, plot_dir + "/internet_delay.png",
         x=[i + 1 for i in range(len(experiment.env.internet_delay))])

    # plot the reward mean and sem
    plot_mean_sem(experiment.train_interval, experiment.train_rewards, plot_dir + "/train_reward_mean_sem.png",
                  "Training Scores")
    plot_mean_sem(experiment.test_interval, experiment.test_rewards, plot_dir + "/test_reward_mean_sem.png",
                  "Testing Scores")

    # plot number of steps per game
    plot(experiment.train_steps_per_game, plot_dir + "/train_steps_per_game.png", x=[i + 1 for i in range(len(experiment.train_steps_per_game))])
    plot(experiment.test_steps_per_game, plot_dir + "/test_steps_per_game.png", x=[i + 1 for i in range(len(experiment.test_steps_per_game))])

    # plot learning curves
    plot(experiment.policy_loss_list, plot_dir + "/policy_loss_list.png", x=[i + 1 for i in range(len(experiment.policy_loss_list))])
    plot(experiment.q1_loss_list, plot_dir + "/q1_loss_list.png", x=[i + 1 for i in range(len(experiment.q1_loss_list))])
    plot(experiment.q2_loss_list, plot_dir + "/q2_loss_list.png", x=[i + 1 for i in range(len(experiment.q2_loss_list))])
    plot(experiment.entropy_loss_list, plot_dir + "/entropy_loss_list.png", x=[i + 1 for i in range(len(experiment.entropy_loss_list))])
