import math
from datetime import timedelta

column_names = ["actions_x", "actions_y", "tray_rot_x", "tray_rot_y", "tray_rot_vel_x", "tray_rot_vel_y",
                "ball_pos_x", "ball_pos_y", "ball_vel_x", "ball_vel_y"]


def get_distance_traveled(dist_travel, prev_observation, observation):
    """
    compounds the distance travelled by the ball
    :param dist_travel: previous distance travelled
    :param prev_observation: previous observation
    :param observation: next observation
    :return: the total travelled distance
    """
    dist_travel += math.sqrt((prev_observation[0] - observation[0]) * (prev_observation[0] - observation[0]) +
                             (prev_observation[1] - observation[1]) * (prev_observation[1] - observation[1]))
    return dist_travel


def get_row_to_store(prev_observation):
    # constructs a row to add in a dataframe
    return {"ball_pos_x": prev_observation[0], "ball_pos_y": prev_observation[1],
            "ball_vel_x": prev_observation[2], "ball_vel_y": prev_observation[3],
            "tray_rot_x": prev_observation[4], "tray_rot_y": prev_observation[5],
            "tray_rot_vel_x": prev_observation[6], "tray_rot_vel_y": prev_observation[7]}


def get_env_action(agent_action, discrete):
    # convert agent's action to an environment-compatible one
    tmp_agent_action = agent_action
    if discrete:
        tmp_agent_action = -1 if agent_action == abs(2) else agent_action
    return tmp_agent_action


def print_logs(verbose, test_model, total_steps, game, best_score, running_reward, avg_length, log_interval,
               avg_ep_duration):
    """print logs during training"""
    if verbose:
        if not test_model:
            if game % log_interval == 0:
                avg_length = int(avg_length / log_interval)
                log_reward = int((running_reward / log_interval))

                print('Episode {}\tTotal timesteps {}\tavg length: {}\tTotal reward(last {} episodes): {}\tBest '
                      'Score: {}\tavg '
                      'episode duration: {}'.format(game, total_steps, avg_length, log_interval, log_reward, best_score,
                                                    timedelta(seconds=avg_ep_duration)))
                running_reward = 0
                avg_length = 0
            return running_reward, avg_length


def test_print_logs(avg_score, avg_length, best_score, duration):
    """print logs during testing"""
    print('Avg Score: {}\tAvg length: {}\tBest Score: {}\tTest duration: {}'.format(avg_score, avg_length, best_score,
                                                                                    timedelta(seconds=duration)))


def get_agent_only_action(agent_action):
    """convert agent's action to an environment-compatible one whe agent is acting alone on the board"""
    # up: 0, down:1, left:2, right:3, upleft:4, upright:5, downleft: 6, downright:7
    if agent_action == 0:
        return [1, 0]
    elif agent_action == 1:
        return [-1, 0]
    elif agent_action == 2:
        return [0, -1]
    elif agent_action == 3:
        return [0, 1]
    elif agent_action == 4:
        return [1, -1]
    elif agent_action == 5:
        return [1, 1]
    elif agent_action == 6:
        return [-1, -1]
    elif agent_action == 7:
        return [-1, 1]
    else:
        print("Invalid agent action")
