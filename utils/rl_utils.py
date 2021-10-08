import stable_baselines3 as sb3
import gym
from constants import params
from typing import Any, Dict
import torch
import numpy as np
import matplotlib.pyplot as plt

class SaveOnBestTrainingRewardCallback(sb3.common.callbacks.BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """
    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

          # Retrieve training reward
          x, y = sb3.common.results_plotter.ts2xy(load_results(self.log_dir), 'timesteps')
          if len(x) > 0:
              # Mean training reward over the last 100 episodes
              mean_reward = np.mean(y[-100:])
              if self.verbose > 0:
                print("Num timesteps: {}".format(self.num_timesteps))
                print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(self.best_mean_reward, mean_reward))

              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  if self.verbose > 0:
                    print("Saving new best model to {}".format(self.save_path))
                  self.model.save(self.save_path)
        return True

class CustomCallback(sb3.common.callbacks.BaseCallback):
    def __init__(self,
        eval_env: gym.Env,
        render_freq: int,
        n_eval_episodes: int = 1,
        deterministic: bool = True
    ):
        """
        Records a video of an agent's trajectory traversing ``eval_env`` and logs it to TensorBoard

        :param eval_env: A gym environment from which the trajectory is recorded
        :param render_freq: Render the agent's trajectory every eval_freq call of the callback.
        :param n_eval_episodes: Number of episodes to render
        :param deterministic: Whether to use deterministic or stochastic policy
        """
        super().__init__()
        self._eval_env = eval_env
        self._render_freq = render_freq
        self._n_eval_episodes = n_eval_episodes
        self._deterministic = deterministic

    def _on_step(self) -> bool:
        if self.n_calls % self._render_freq == 0:
            screens = []

            def grab_screens(
                _locals: Dict[str, Any],
                _globals: Dict[str, Any]
            ) -> None:
                """
                Renders the environment in its current state,
                    recording the screen in the captured `screens` list

                :param _locals:
                    A dictionary containing all local variables of the callback's scope
                :param _globals:
                    A dictionary containing all global variables of the callback's scope
                """
                screen = self._eval_env.render(mode="rgb_array")
                # PyTorch uses CxHxW vs HxWxC gym (and tensorflow) image convention
                screens.append(screen.transpose(2, 0, 1))
            
            sb3.common.evaluation.evaluate_policy(
                self.model,
                self._eval_env,
                callback=grab_screens,
                n_eval_episodes=self._n_eval_episodes,
                deterministic=self._deterministic,
            )
            self.logger.record(
                "trajectory/video",
                sb3.common.logger.Video(torch.ByteTensor([screens]), fps=40),
                exclude=("stdout", "log", "json", "csv"),
            )
            for item in params['track_list']:
                ITEM = np.stack(self._eval_env.env.env._track_item[item], 0)
                fig, ax = plt.subplots(
                    1, ITEM.shape[-1],
                    figsize = (7.5 * ITEM.shape[-1], 7.5)
                )
                T = np.arange(ITEM.shape[0]) * self._eval_env.dt
                if ITEM.shape[-1] > 1:
                    for j in range(ITEM.shape[-1]):
                        ax[j].plot(T, ITEM[:, j], color = 'r',
                            label = '{}_{}'.format(item, j),
                            linestyle = '--')
                        ax[j].set_xlabel('time', fontsize = 12)
                        ax[j].set_ylabel('{}'.format(item), fontsize = 12)
                        ax[j].legend(loc = 'upper left')
                else:
                    ax.plot(T, ITEM[:, 0], color = 'r',
                        label = '{}_{}'.format(item, 0), 
                        linestyle = '--')
                    ax.set_xlabel('time', fontsize = 12) 
                    ax.set_ylabel('{}'.format(item), fontsize = 12) 
                    ax.legend(loc = 'upper left')
                self.logger.record("trajectory/{}".format(item), 
                    sb3.common.logger.Figure(
                        fig, close=True
                    ),
                    exclude = ("stdout", "log", "json", "csv")
                )
                plt.close()
        return True
