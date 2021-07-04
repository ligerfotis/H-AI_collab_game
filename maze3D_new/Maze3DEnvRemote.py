import random
import time
import traceback

import numpy as np
import requests as requests

from plot_utils.plot_utils import get_config, plot_learning_curve, plot_test_score, plot, plot_mean_sem


def reward_function_timeout_penalty(goal_reached, timedout):
    # for every timestep -1
    # timed out -50
    # reach goal +100
    if goal_reached:
        return 100
    if timedout:
        return -50
    return -1


class ActionSpace:
    def __init__(self):
        self.actions = list(range(3))
        self.shape = 2
        self.actions_number = len(self.actions)
        self.high = self.actions[-1]
        self.low = self.actions[0]

    def sample(self):
        return np.random.randint(self.low, self.high + 1, 2)


class Maze3D:
    def __init__(self, config=None, config_file=None):
        print("Init Maze3D")
        self.config = get_config(config_file) if config_file is not None else config
        self.ip_host = "https://maze-server.app.orbitsystems.gr"
        # self.host = "http://79.129.14.204:8080"
        self.host = "http://maze3d.duckdns.org:8080"
        # self.host = 'http://panos-server.duckdns.org:8080'
        # self.host = "http://localhost:8080"
        self.action_space = ActionSpace()
        self.fps = 60
        self.done = False
        self.set_host()
        self.send_config()
        self.agent_ready()
        self.observation, _ = self.reset()
        self.observation_shape = (len(self.observation),)
        self.internet_delay = []

    def send_config(self):
        config = {}

        while True:
            try:
                config['discrete_input'] = self.config['game']['discrete_input']
                config['max_duration'] = self.config['Experiment']['max_games_mode']['max_duration']
                config['action_duration'] = self.config['Experiment']['max_games_mode']['action_duration']
                config['human_speed'] = 30
                config['agent_speed'] = 30
                config['discrete_angle_change'] = 10
                config['human_assist'] = False
                config['start_up_screen_display_duration'] = self.config['GUI']['start_up_screen_display_duration']
                config['popup_window_time'] = self.config['GUI']['popup_window_time']
                print(config)
                requests.post(self.host + "/config", json=config).json()
                return
            except Exception as e:
                print("/agent_ready not returned", e)
                time.sleep(1)

    def set_host(self):
        while True:
            try:
                requests.get(self.host + "/set_server_host/" + self.host).json()
                break
            except Exception as e:
                # print("/agent_ready not returned", e)
                time.sleep(0.1)

    def agent_ready(self):
        while True:
            try:
                res = requests.get(self.host + "/agent_ready").json()
                if 'command' in res and res['command'] == "player_ready":
                    break
            except Exception as e:
                # print("/agent_ready not returned", e)
                time.sleep(0.1)

    def send(self, namespace, method="GET", data=None):
        while True:
            try:
                if method == " GET":
                    res = requests.get(self.host + namespace).json()
                else:
                    res = requests.post(self.host + namespace, json=data).json()

                if 'command' in res and res['command'] == "player_ready":
                    continue
                return res
            except Exception as e:
                # in here when wrong request is given
                # traceback.print_exc()
                self.agent_ready()
                time.sleep(0.1)

    def reset(self):
        # print("reset")
        start_time = time.time()
        res = self.send("/reset")
        print("reset time:", time.time() - start_time)
        return np.array(res['observation']), res['setting_up_duration']

    def training(self, cycle, total_cycles):
        self.send("/training", "POST", {'cycle': cycle, 'total_cycles': total_cycles})

    def finished(self):
        print("finished")
        self.send("/finished", "GET")

    def step(self, action_agent, timed_out, action_duration, mode):
        """
        Performs the action of the agent to the environment for action_duration time.
        Simultaneously, receives input from the user via the keyboard arrows.

        :param action_agent: the action of the agent. gives -1 for down, 0 for nothing and 1 for up
        :param timed_out: used
        :param action_duration: the duration of the agent's action on the game
        :param mode: training or test
        :return: a transition [observation, reward, done, timeout, train_fps, duration_pause, action_list]
        """
        # print("step", timed_out)
        # if timed_out:
        #     print("timeout", timed_out, int(time.time()))

        payload = {
            'action_agent': action_agent,
            'action_duration': action_duration,
            'timed_out': timed_out,
            'mode': mode
        }
        start_time = time.time()
        res = self.send("/step", method="POST", data=payload)
        delay = time.time() - start_time
        self.internet_delay.append(delay)
        print(delay)
        self.observation = np.array(res['observation'])
        self.done = res['done']  # true if goal_reached OR timeout
        fps = res['fps']
        human_action = res['human_action']
        agent_action = res['agent_action']
        duration_pause = res['duration_pause']

        reward = reward_function_timeout_penalty(self.done, timed_out)

        return self.observation, reward, self.done, fps, duration_pause, [agent_action, human_action]


if __name__ == '__main__':
    """Dummy execution"""
    while True:
        try:
            maze = Maze3D()
            while True:
                maze.step(random.randint(-1, 1), None, None, 200)
        except:
            traceback.print_exc()


def save_logs_and_plot(experiment, chkpt_dir, plot_dir, max_games):
    # score_history a list with the reward for each episode
    x = [i + 1 for i in range(len(experiment.score_history))]
    np.savetxt(chkpt_dir + '/scores.csv', np.asarray(experiment.score_history), delimiter=',')

    # action_history as returned by get_action_pair: a dyad agent and human {-1,0,1}
    actions = np.asarray(experiment.action_history)

    x_actions = [i + 1 for i in range(len(actions))]
    # Save logs in files
    np.savetxt(chkpt_dir + '/actions.csv', actions, delimiter=',')
    # np.savetxt('tmp/sac_' + timestamp + '/action_side.csv', action_side, delimiter=',')
    np.savetxt(chkpt_dir + '/epidode_durations.csv', np.asarray(experiment.game_duration_list), delimiter=',')

    np.savetxt(chkpt_dir + '/game_step_durations.csv', np.asarray(experiment.train_step_duration_list), delimiter=',')
    np.savetxt(chkpt_dir + '/online_update_durations.csv', np.asarray(experiment.online_update_duration_list),
               delimiter=',')
    np.savetxt(chkpt_dir + '/total_fps.csv', np.asarray(experiment.total_fps_list), delimiter=',')
    np.savetxt(chkpt_dir + '/train_fps.csv', np.asarray(experiment.train_fps_list), delimiter=',')
    np.savetxt(chkpt_dir + '/test_fps.csv', np.asarray(experiment.test_fps_list), delimiter=',')

    np.savetxt(chkpt_dir + '/distance_travel.csv', np.asarray(experiment.distance_travel_list), delimiter=',')
    np.savetxt(chkpt_dir + '/distance_travel_test.csv', np.asarray(experiment.test_distance_travel_list), delimiter=',')
    np.savetxt(chkpt_dir + '/pure_rewards.csv', experiment.reward_list, delimiter=',')
    np.savetxt(chkpt_dir + '/pure_rewards_test.csv', experiment.test_reward_list, delimiter=',')

    np.savetxt(chkpt_dir + '/grad_updates_durations.csv', experiment.grad_updates_durations, delimiter=',')

    # test_game_number logs
    np.savetxt(chkpt_dir + '/test_episode_duration_list.csv', experiment.test_game_duration_list, delimiter=',')
    np.savetxt(chkpt_dir + '/test_score_history.csv', experiment.test_score_history, delimiter=',')
    np.savetxt(chkpt_dir + '/test_length_list.csv', experiment.test_length_list, delimiter=',')

    # plot_learning_curve(x, experiment.score_history, plot_dir + "/train_scores.png")
    plot(experiment.length_list, plot_dir + "/train_length.png", x=[i + 1 for i in range(max_games)])
    plot(experiment.game_duration_list, plot_dir + "/train_game_durations.png", x=[i + 1 for i in range(max_games)])

    plot(experiment.train_step_duration_list, plot_dir + "/train_game_step_durations.png",
         x=[i + 1 for i in range(len(experiment.train_step_duration_list))])
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

    plot(experiment.grad_updates_durations, plot_dir + "/grad_updates_durations.png",
         x=[i + 1 for i in range(len(experiment.grad_updates_durations))])

    # plot game logs
    # todo: not working properly
    plot_test_score(experiment.test_score_history, plot_dir + "/test_scores.png")
    plot(experiment.test_length_list, plot_dir + "/test_length.png",
         x=[i + 1 for i in range(len(experiment.test_length_list))])
    plot(experiment.test_game_duration_list, plot_dir + "/test_game_duration.png",
         x=[i + 1 for i in range(len(experiment.test_game_duration_list))])

    # todo: not working properly
    x = [i + 1 for i in range(experiment.max_games)]
    plot_learning_curve(x, experiment.reward_list, plot_dir + "/rewards_train.png")
    x = [i + 1 for i in range(int(experiment.test_max_games * experiment.max_games / experiment.test_interval))]
    plot_learning_curve(x, experiment.test_reward_list, plot_dir + "/rewards_test.png")

    plot_mean_sem(experiment.test_max_games, experiment.test_score_history, plot_dir + "/score_mean_sem.png",
                  "Testing Scores")
    try:
        # todo: not working properly
        plot_test_score(experiment.test_score_history, plot_dir + "/test_scores_mean_std.png")
    except:
        print("An exception occurred while plotting")
