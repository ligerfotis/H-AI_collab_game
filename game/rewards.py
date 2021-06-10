from maze3D_new.utils import get_distance_from_goal


def main(config):
	global reward_type
	reward_type = config['SAC']['reward_function'] if 'SAC' in config.keys() else None

def reward_function_maze(goal_reached, timedout, ball, goal=None):
	if reward_type in ["Timeout", "timeout"]:
		return reward_function_timeout_penalty(goal_reached, timedout)
	elif reward_type in ["Distance", "distance"]:
		return reward_function_distance(goal_reached, timedout, ball=ball, goal=goal)
	elif reward_type in ["Shafti", "shafti"]:
		return reward_function_shafti(goal_reached)

def reward_function_timeout_penalty(goal_reached, timedout):
	# for every timestep -1
	# timed out -50
	# reach goal +100
	if goal_reached:
		return 100
	if timedout:
		return -50
	return -1

def reward_function_shafti(goal_reached):
	# For every timestep -1
	# Reach goal +100
	if goal_reached:
		return 100
	return -1

def reward_function_distance(goal_reached, timedout, ball, goal=None):
	# for every timestep -target_distance
	# timed_out -50
	# reach goal +100
	if goal_reached:
		return 100
	if timedout:
		return -50
	return get_distance_from_goal(ball, goal)
	

def reward_function(goal_reached, timedout, goal=None):
	# Construct here the mathematical reward function
  	# The reward function during the train_game_number can be static or it can depend on time, the distance of the ball to the goal, etc.
   	# The reward when the goal reached can also be defined as well as the penalty if the train_game_number times out.
	# Once you write the function you need to do two things.
	# First, set it as an option in the `reward_function_maze` function and set the correspoding key
	# Secondly, choose this key in the config files where you select which reward function to choose
	# See an example below
	if goal_reached:
		return 100
	if timedout:
		return -5
	#Static reward function
	return -1

if __name__ == "__main__":
	main(config)
