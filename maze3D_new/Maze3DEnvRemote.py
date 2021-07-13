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
        self.network_config = get_config("game/network_config.yaml")
        self.ip_host = self.network_config["ip_distributor"]
        self.outer_host = self.network_config["maze_server"]
        self.host = self.network_config["maze_rl"]

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
                config['human_speed'] = 35
                config['agent_speed'] = 35
                config['discrete_angle_change'] = 3
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
                requests.post(self.ip_host + "/set_server_host",json={'server_host':self.outer_host}).json()
                break
            except Exception as e:
                print("ip host offline", e)
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
                if method == "GET":
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