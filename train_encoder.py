from utils.rtd3 import train_autoencoder
from simulations.maze_env import MazeEnv
from simulations.point import PointEnv
from simulations.maze_task import CustomGoalReward4Rooms
import stable_baselines3 as sb3
from utils import set_seeds
from constants import params

if __name__ == '__main__':
    set_seeds(117)
    VecTransposeImage = sb3.common.vec_env.vec_transpose.VecTransposeImage
    env = VecTransposeImage(
        sb3.common.vec_env.dummy_vec_env.DummyVecEnv([
            lambda : sb3.common.monitor.Monitor(MazeEnv(
                PointEnv,
                CustomGoalReward4Rooms,
                max_episode_size = params['max_episode_size'],
                n_steps = params['max_seq_len'] * params['seq_sample_freq'],
            ))
        ])
    )

    logdir = '/content/drive/MyDrive/CNS/exp22/autoencoder/exp' 
    if params['debug']:
        logdir = 'assets/out/models/autoencoder/'

    train_autoencoder(
        logdir,
        env, 
        n_epochs = 100,
        batch_size = 100,
        learning_rate = 1e-3,
        save_freq = 5,
        eval_freq = 5,
    )
