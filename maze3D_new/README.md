# Graphical Environment Web Wrapper

The functionality of communicatting with the MazeUnity environment (via Maze-Server) is implemented in `Maze3DEnvRemote.py`.

Types of requests, posted to Maze-Server:
* _reset_.
    * Carries the following information
        * Nothing
    * Response includes:
        * the initial observation of the environment
    
* _step_. 
    * Carries the following information
        * a<sup>agent</sup>
        * action duration (the duration we want a<sup>agent</sup> to be executed in the environment before getting next one)
        * timeout (if game has timed out)
        * mode (the experiment mode; _train_ or _test_)
    * Response includes:
        * the observation of the environment
        * if game has finished
        * the running fps
        * the duration of a pause, if game was paused by the user
        * distance from goal
        * a<sup>human<sup/>
        * a<sup>agent<sup/>
    
* _training_.
    * Carries the following information
        * current cycle(epoch) of Gradient Updates
        * total number of cycle(epoch) of Gradient Updates during this session.
    * Response includes:
        * nothing
    
* _finished_.
    * Carries the following information
        * nothing
    * Response includes:
        * nothing

