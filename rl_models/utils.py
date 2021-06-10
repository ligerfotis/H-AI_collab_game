from rl_models.sac_agent import Agent
from rl_models.sac_discrete_agent import DiscreteSACAgent


def get_sac_agent(config, env, chkpt_dir=None):
    discrete = config['SAC']['discrete']
    if discrete:
        if config['Experiment']['mode'] == 'max_games_mode':
            buffer_max_size = config['Experiment']['max_games_mode']['buffer_memory_size']
            update_interval = config['Experiment']['max_games_mode']['learn_every_n_games']
            scale = config['Experiment']['max_games_mode']['reward_scale']
        elif config['Experiment']['mode'] == 'max_interactions_mode':
            buffer_max_size = config['Experiment']['max_interactions_mode']['buffer_memory_size']
            update_interval = config['Experiment']['max_interactions_mode']['learn_every_n_timesteps']
            scale = config['Experiment']['max_interactions_mode']['reward_scale']
        else:
            print("Unknown experiment mode")
            exit(1)

        if config['game']['agent_only']:
            # up: 1, down:2, left:3, right:4, upleft:5, upright:6, downleft: 7, downright:8
            action_dim = pow(2, env.action_space.actions_number)
        else:
            action_dim = env.action_space.actions_number
        sac = DiscreteSACAgent(config=config, env=env, input_dims=env.observation_shape,
                               n_actions=action_dim,
                               chkpt_dir=chkpt_dir, buffer_max_size=buffer_max_size, update_interval=update_interval,
                               reward_scale=scale)
    else:
        sac = Agent(config=config, env=env, input_dims=env.observation_shape, n_actions=env.action_space.shape,
                    chkpt_dir=chkpt_dir)
    return sac
