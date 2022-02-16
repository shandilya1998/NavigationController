from utils.rtd3_utils import train_autoencoder
from simulations.maze_env import MazeEnv
from simulations.point import PointEnv
from simulations.maze_task import CustomGoalReward4Rooms
import stable_baselines3 as sb3

if __name__ == '__main__':
    VecTransposeImage = sb3.common.vec_env.vec_transpose.VecTransposeImage
    env = VecTransposeImage(
        sb3.common.vec_env.dummy_vec_env.DummyVecEnv([
            lambda : sb3.common.monitor.Monitor(MazeEnv(
                PointEnv,
                CustomGoalReward4Rooms,
                max_episode_size = 750,
                n_steps = 15
            ))
        ])
    )

    #logdir = '/content/drive/MyDrive/CNS/exp22/autoencoder' 
    logdir = 'assets/out/models/exp22/autoencoder/'

    train_autoencoder(
        logdir,
        env, 
        n_epochs = 1000,
        batch_size = 64,
        learning_rate = 1e-3,
        save_freq = 100,
        eval_freq = 50,
        max_episode_size = 750,
    )
