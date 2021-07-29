# MazeRL v2
##Collaborative Reinforcement Learning on Human-Computer shared task

### Description
A human- RL agent collaborative game on a [graphical environment](https://github.com/ligerfotis/MazeUnity). This is an extension of [this work](https://github.com/ligerfotis/maze3d_collaborative).
The environment is build in Unity and communicates with the experiment via a HTTP server.
Collaborative Learning is achieved through Deep Reinforcement Learning (DRL). The Soft-Actor Critic (SAC) algorithm is used [2] with modifications for discrete action space [3].

### Experiment Set Up
The set-up consists of 3 components:
1. The [Maze Server](https://github.com/panos-stavrianos/maze_server): Dedicated HTTP server that takes data from the experiment (mazeRL) and passes them to the Unity environment (mazeUnity) and vice versa.
2. The online version of the [MazeRL](https://github.com/ligerfotis/maze_RL_online) experiment: Includes the training loops, the RL agent and different configuration files.
3. The graphical environment [MazeUnity](https://github.com/ligerfotis/MazeUnity): A simulation of the real world experiment from Shafti et al. (2020) [1]

The pipeline to start an experiment is described bellow:
* Start the dedicated [Maze Server](https://github.com/panos-stavrianos/maze_server)
  * Can be started after MazeRL has started
  * If started before MazeRL has it will wait for it to connect.
  * Receives a configuration file from MazeRL and delivers it to MazeUnity upon startup of the latter.
  
* Start the experiment [MazeRL](https://github.com/ligerfotis/maze_RL_online) (See [Run MazeRL](#run-mazerl))

* Open the graphical environment [MazeUnity](https://github.com/ligerfotis/MazeUnity)
### Installation
* Run `source install_dependencies/install.sh`. 
  - A python virtual environment will be created and the necessary libraries will be installed.
  - Furthermore, the directory of the repo will be added to the `PYTHONPATH` environmental variable.

### Run MazeRL

* Run `python game/sac_maze3d_train.py game/config/<config_sac> <participant_name>` for human-agent game.
  * Example:
    
        python game/sac_maze3d_train.py game/config/config_sac_60K_O-O-a_descending.yaml participant_1

  * Notes before training: 
     * Set the <participant_name> to the name of the participant.
     * The program will create a `/tmp` and a `/plot` folder (if they do not exist) in the `results/` folder. The `/tmp` folder contains CSV files with information of the game. The `/plot` folder contains figures for tha game. See [here](#Experiment-Result-Output-Files) for more details.
     * The program will automatically create an identification number after your name on each folder name created
    
### Configuration
* In the game/config folder several YAML files exist for the configuration of the experiment. The main parameters are listed below.
    * `game/discrete_input`: True if the keyboard input is discrete (False for continuous). Details regarding the discrete and continuous human input mode can be found [here](https://github.com/ligerfotis/maze_RL_v2/blob/master/game)
    * `SAC/reward_function`: Type of reward function. Details about the predefined reward functions and how to define a new one can be found [here](https://github.com/ligerfotis/maze_RL_v2/blob/master/game).
    * `Experiment/mode`: Choose how the game will be terminated; either when a number of games, or a number of interactions is completed.
    * `SAC/discrete`: Discrete or normal SAC (Currently only the discrete SAC is compatible with the game)
  
### Play
Directions of how to play the game are given in [MazeUnity](https://github.com/ligerfotis/MazeUnity).

## Citation

If you use this repository in your publication please cite below:
```
Fotios Lygerakis, Maria Dagioglou, and Vangelis Karkaletsis. 2021. Accelerating Human-Agent Collaborative Reinforcement Learning. InThe 14th PErvasive Technologies Related to Assistive Environments Conference (PETRA2021), June 29-July 2, 2021, Corfu, Greece.ACM, New York, NY, USA, 3 pages.https://doi.org/10.1145/3453892.3454004
```
### Experiment Result Output Files
Contents of a`/tmp` folder.
  * `<test/train>_scores.csv` The total score for each training _game_.
  * `<test/train>_time_scores.csv` The time score ([max game duration] - [game_duration]) for each training _game_.
  * `<test/train>_rewards.csv` The cumulative reward for each _game_.
  * `<test/train>_game_durations.csv` The total duration _game_.
  * `<test/train>_game_success_rate.csv` The success rate ([games that the goal was reached]/[total games played in the session]) for each session (every <update_interval> games).
  * `<test/train>_step_durations.csv` The duration _game_step_.
  * `<test/train>_steps_per_game.csv` The total number of steps per _game_.
  * `<test/train>_logs.pkl` A pandas dataframe containing tuples of (s<sub>previous</sub>, a<sup>agent</sup><sub>real</sub>, 
    a<sup>agent</sup><sub>environment</sub>, a<sup>human</sup>, s,r)
    * a<sup>agent</sup><sub>real</sub>: The real agent action predicted [0, 1 or 2]
    * a<sup>agent</sup><sub>environment</sub>: The agent action compatible to the environment (0 ->0, 1->1 and 2->-1)
    * `config_sac_***.yaml`: The configuration file used for this experiment. It's purpose it to be able to replicate this experiment.
  * `actor_sac/critic_sac` The network weights.
  * `<test/train>_distance_travelled.csv` The distance travelled by the ball for each _game_.
  * `<test/train>_fps.csv` The frames per second that the game was played on the screen.
    
  * `rest_info.csv`: internet delay statistics, goal position, total experiment duration, best score achieved, the _game_ that achieved the best score, the best reward achieved, the length of the game trial with the best score, the total amount of time steps for the whole experiment, the total number of _games_ played, the fps the game run on and the average offline gradient update duration over all sessions.

Contents of a`/plot` folder: .png figures of the logs saved in `/tmp` folder.

### References
[1] Shafti, Ali, et al. "Real-world human-robot collaborative reinforcement learning." arXiv preprint arXiv:2003.01156 (2020).

[2] https://github.com/kengz/SLM-Lab

[3] Christodoulou, Petros. "Soft actor-critic for discrete action settings." arXiv preprint arXiv:1910.07207 (2019).

[4] Fotios Lygerakis, Maria Dagioglou, and Vangelis Karkaletsis. 2021. Accelerating Human-Agent Collaborative Reinforcement Learning. InThe 14th PErvasive Technologies Related to Assistive Environments Conference (PETRA2021), June 29-July 2, 2021, Corfu, Greece.ACM, New York, NY, USA, 3 pages.https://doi.org/10.1145/3453892.3454004