# NavigationController

## Spawn Test Environment
```
from simulations.maze_env import MazeEnv
from simulations.point import PointEnv
from simulations.maze_task import CustomGoalReward4Rooms
env = MazeEnv(PointEnv, CustomRewardGoal4Rooms)
```

## Experiment Checklist

- Check and update `def step(**kwargs)` and `def _set_action_space(**kwargs)` in [simulations/point.py](simulations/point.py)
- Check and update `def _get_obs(**kwargs)` and `def _set_observation_space(**kwargs)` in [simulations/maze_env.py](simulations/maze_env.py) and [simulations/point.py](simulations/point.py) 
- Update Reward and Info Computation Pipeline
- Check and update `class Actor` and `class Critic` in [utils/rtd3_utils.py](utils/rtd3_utils.py)
- Check and update `class Autoencoder`, `class VisualCortexV4`, `class EncoderBody` and `class FeatureExtractionBackbone` in [bg/models.py](bg/models.py)
- Check and update `def update_policy(**kwargs)` and `def _sample_action(**kwargs)` in [utils/rtd3_utils.py](utils/rtd3_utils.py)
- Check and update `def predict(**kwargs)` in [utils/rtd3_utils.py](utils/rtd3_utils.py)
- Ensure `squash_output` is False in `class Actor` if using inverted gradients method
- Check and update `seed`, `batch_size`, `learning_starts`, `imitation_steps` and `debug` in `params` from [constant.py](constants.py)
- Check and update other hyperparameters for tuning
- Set `LOGDIR`, `ENV_TYPE`, `TIMESTEPS`, `MAX_EPISODE_SIZE`, `HISTORY_STEPS` and `TASK_VERSION` in [train.sh](train.sh)
- Check and update `net_arch` and `n_critics` in [learning/explore.py](learning/explore.py)
- Run `sh train.sh` in terminal for debugging. Run `nohup sh train.sh >> assets/out/models/train.log &` in terminal for GPU execution.
- Commit and Push to Github. Update local `experiments_log`

## Installation on docker

- Run `docker build -t neuroengineering-tools ./docker/cpu/` in the root folder of the repository.
- Run `docker run -dit --name sample neuroengineering-tools` to spawn  a new container.
- Run desired command with the tools in the container.
- Run `docker stop neuroengineering-tools` to stop running container.



