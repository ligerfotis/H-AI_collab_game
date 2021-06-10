import math


class UpdatesScheduler:
    def __init__(self):
        self.counter = 0

    def schedule(self, max_game_duration, action_duration, mode, max_games, update_cycles, update_interval, config):
        update_list = [22000, 1000, 1000, 1000, 1000, 1000, 1000]
        total_update_cycles = config['Experiment'][mode]['total_update_cycles']
        online_updates = 0
        if config['Experiment']['online_updates']:
            online_updates = max_game_duration / action_duration * (
                    max_games - config['Experiment'][mode]['start_training_step_on_game'])

        if update_cycles is None:
            update_cycles = total_update_cycles - online_updates

        if config['Experiment']['scheduling'] == "descending":
            self.counter += 1
            if not (math.ceil(max_games / update_interval) == self.counter):
                update_cycles /= 2

        elif config['Experiment']['scheduling'] == "big_first":
            if config['Experiment']['online_updates']:
                if self.counter == 1:
                    update_cycles = update_list[self.counter]
                else:
                    update_cycles = 0
            else:
                update_cycles = update_list[self.counter]

            self.counter += 1

        else:
            update_cycles = (total_update_cycles - online_updates) / math.ceil(
                max_games / update_interval)

        return math.ceil(update_cycles)
