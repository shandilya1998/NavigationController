from random import randint
import os
import warnings
import cv2
import gym
import stable_baselines3 as sb3
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


class Callback(sb3.common.callbacks.EventCallback):
    """
    Callback for evaluating an agent. Records videos and logs mean reward.
    .. warning::
      When using multiple environments, each call to  ``env.step()``
      will effectively correspond to ``n_envs`` steps.
      To account for that, you can use ``eval_freq = max(eval_freq // n_envs, 1)``

    :param eval_env: The environment used for initialization
    :type eval_env: Union[gym.Env, sb3.common.vec_env.VecEnv]
    :param logdir: Path to logging directory.
    :type logdir: str
    :param callback_on_new_best: Callback to trigger
        when there is a new best model according to the ``mean_reward``
    :type callback_on_new_best: Optional[sb3.common.callbacks.BaseCallback] 
    :param n_eval_episodes: The number of episodes to test the agent
    :type n_eval_episodes: int
    :param eval_freq: Evaluate the agent every ``eval_freq`` call of the callback.
    :param render_every: Frequency per agent evaluation.
    :type render_every: int
    :param log_path: Path to a folder where the evaluations (``evaluations.npz``)
        will be saved. It will be updated at each evaluation.
    :type log_path: str
    :param best_model_save_path: Path to a folder where the best model
        according to performance on the eval env will be saved.
    type best_model_save_path: str
    :param deterministic: Whether the evaluation should
        use a stochastic or deterministic actions.
    :type deterministic: bool
    :param render: Whether to render or not the environment during evaluation
    :type render: bool
    :param verbose: Verbosity Level.
    :rype verbose: int
    :param warn: Passed to ``evaluate_policy`` (warns if ``eval_env`` has not been
        wrapped with a Monitor wrapper)
    """

    def __init__(
        self,
        eval_env: Union[gym.Env, sb3.common.vec_env.VecEnv],
        logdir: str,
        callback_on_new_best: Optional[sb3.common.callbacks.BaseCallback] = None,
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        render_every: int = 10,
        image_size: Tuple[int] = (1024, 1024),
        log_path: str = None,
        best_model_save_path: str = None,
        deterministic: bool = True,
        render: bool = False,
        verbose: int = 1,
        warn: bool = True,
    ):
        super(Callback, self).__init__(callback_on_new_best, verbose=verbose)
        self.logdir = logdir
        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.render_freq = render_every * self.eval_freq
        self.image_size = image_size
        self.best_mean_reward = -np.inf
        self.last_mean_reward = -np.inf
        self.deterministic = deterministic
        self.render = render
        self.warn = warn

        # Convert to VecEnv for consistency
        if not isinstance(eval_env, sb3.common.vec_env.VecEnv):
            eval_env = sb3.common.vec_env.dummy_vec_env.DummyVecEnv([lambda: eval_env])

        self.eval_env = eval_env
        self.best_model_save_path = best_model_save_path
        # Logs will be written in ``evaluations.npz``
        if log_path is not None:
            log_path = os.path.join(log_path, "evaluations")
        self.log_path = log_path
        self.evaluations_results = []
        self.evaluations_timesteps = []
        self.evaluations_length = []
        # For computing success rate
        self._is_success_buffer = []
        self.evaluations_successes = []

    def _init_callback(self) -> None:
        # Does not work in some corner cases, where the wrapper is not the same
        if not isinstance(self.training_env, type(self.eval_env)):
            warnings.warn("Training and eval env are not of the same type" f"{self.training_env} != {self.eval_env}")

        # Create folders if needed
        if self.best_model_save_path is not None:
            os.makedirs(self.best_model_save_path, exist_ok=True)
        if self.log_path is not None:
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

    def _log_success_callback(self, locals_: Dict[str, Any], globals_: Dict[str, Any]) -> None:
        """
        Callback passed to the  ``evaluate_policy`` function
        in order to log the success rate (when applicable),
        for instance when using HER.
        :param locals_:
        :param globals_:
        """
        info = locals_["info"]

        if locals_["done"]:
            maybe_is_success = info.get("is_success")
            if maybe_is_success is not None:
                self._is_success_buffer.append(maybe_is_success)

    def _on_step(self) -> bool:

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Sync training and eval env if there is VecNormalize
            sb3.common.vec_env.sync_envs_normalization(self.training_env, self.eval_env)

            # Reset success rate buffer
            self._is_success_buffer = []

            video = None
            callback = self._log_success_callback
            if self.n_calls % self.render_freq == 0:
                video = cv2.VideoWriter(
                    os.path.join(self.logdir, 'model_{}_evaluation.avi'.format(int(self.n_calls))),
                    cv2.VideoWriter_fourcc(*"MJPG"), 10, self.image_size, isColor=True
                )
                REWARDS = []
                COMPONENTS = {}
                COLORS = []
                fig, ax = plt.subplots(1, 1, figsize=(6.5, 6.5))
                fig1, ax1 = plt.subplots(1, 1, figsize=(6.5, 6.5))
                canvas = FigureCanvas(fig)
                canvas1 = FigureCanvas(fig1)
                ax.set_xlabel('steps')
                ax.set_ylabel('reward')
                ax1.set_xlabel('steps')
                ax1.set_ylabel('reward')

                def callback(
                    _locals: Dict[str, Any],
                    _globals: Dict[str, Any]
                ) -> None:
                    self._log_success_callback(_locals, _globals)
                    """
                    Renders the environment in its current state,
                        recording the screen in the captured `screens` list

                    :param _locals:
                        A dictionary containing all local variables of the callback's scope
                    :param _globals:
                        A dictionary containing all global variables of the callback's scope
                    """
                    REWARDS.append(_locals['rewards'][0])
                    if len(COMPONENTS) == 0 and len(COLORS) == 0:
                        for component in _locals['infos'][0]['reward_keys']:
                            COMPONENTS[component] = [_locals['infos'][0][component]]
                            COLORS.append('#%06X' % randint(0, 0xFFFFFF))
                    elif len(COMPONENTS) > 0 and len(COLORS) == 0:
                        for component in _locals['infos'][0]['reward_keys']:
                            COMPONENTS[component].append(_locals['infos'][0][component])
                            COLORS.append('#%06X' % randint(0, 0xFFFFFF))
                    else:
                        for component in _locals['infos'][0]['reward_keys']:
                            COMPONENTS[component].append(_locals['infos'][0][component])
                    screen = self.eval_env.render(mode="rgb_array")
                    size = screen.shape[:2]
                    if 'frame_t' in _locals['observations'].keys():
                        screen = cv2.cvtColor(screen, cv2.COLOR_RGB2BGR)
                        # PyTorch uses CxHxW vs HxWxC gym (and tensorflow) image convention
                        frame_t = cv2.resize(
                            _locals['observations']['frame_t'][0, :3].transpose(1, 2, 0),
                            size
                        )


                        ax.clear()
                        ax.plot(REWARDS, color='r', linestyle='--')
                        canvas.draw()
                        image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
                        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                        image = cv2.resize(image, size)

                        ax1.clear()
                        for i, (component, lst) in enumerate(COMPONENTS.items()):
                            ax1.plot(lst, color=COLORS[i], linestyle='--', label=component)
                        ax1.legend(loc='upper left')
                        canvas1.draw()
                        image1 = np.frombuffer(canvas1.tostring_rgb(), dtype='uint8')
                        image1 = image1.reshape(fig1.canvas.get_width_height()[::-1] + (3,))
                        image1 = cv2.resize(image1, size)

                        observation = np.concatenate([
                            np.concatenate([screen, image], 0),
                            np.concatenate([frame_t, image1], 0),
                        ], 1).astype(np.uint8)
                        observation = cv2.cvtColor(observation, cv2.COLOR_RGB2BGR)

                        video.write(observation)
                    if _locals['done']:
                        REWARDS.clear()
                        COLORS.clear()
                        for component in COMPONENTS.keys():
                            COMPONENTS[component].clear()



            episode_rewards, episode_lengths = sb3.common.evaluation.evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                render=self.render,
                deterministic=self.deterministic,
                return_episode_rewards=True,
                warn=self.warn,
                callback=callback,
            )

            if self.n_calls % self.render_freq == 0:
                cv2.destroyAllWindows()
                video.release()
                plt.close()

            if self.log_path is not None:
                self.evaluations_timesteps.append(self.num_timesteps)
                self.evaluations_results.append(episode_rewards)
                self.evaluations_length.append(episode_lengths)

                kwargs = {}
                # Save success log if present
                if len(self._is_success_buffer) > 0:
                    self.evaluations_successes.append(self._is_success_buffer)
                    kwargs = dict(successes=self.evaluations_successes)

                np.savez(
                    self.log_path,
                    timesteps=self.evaluations_timesteps,
                    results=self.evaluations_results,
                    ep_lengths=self.evaluations_length,
                    **kwargs,
                )

            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
            mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
            self.last_mean_reward = mean_reward

            if self.verbose > 0:
                print(f"Eval num_timesteps={self.num_timesteps}, " f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
                print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")
            # Add to current Logger
            self.logger.record("eval/mean_reward", float(mean_reward))
            self.logger.record("eval/mean_ep_length", mean_ep_length)

            if len(self._is_success_buffer) > 0:
                success_rate = np.mean(self._is_success_buffer)
                if self.verbose > 0:
                    print(f"Success rate: {100 * success_rate:.2f}%")
                self.logger.record("eval/success_rate", success_rate)

            # Dump log so the evaluation results are printed with the correct timestep
            self.logger.record("time/total timesteps", self.num_timesteps, exclude="tensorboard")
            self.logger.dump(self.num_timesteps)

            if mean_reward > self.best_mean_reward:
                if self.verbose > 0:
                    print("New best mean reward!")
                if self.best_model_save_path is not None:
                    self.model.save(os.path.join(self.best_model_save_path, "best_model"))
                self.best_mean_reward = mean_reward
                # Trigger callback if needed
                if self.callback is not None:
                    return self._on_event()

        return True

    def update_child_locals(self, locals_: Dict[str, Any]) -> None:
        """
        Update the references to the local variables.
        :param locals_: the local variables during rollout collection
        """
        if self.callback:
            self.callback.update_locals(locals_)
