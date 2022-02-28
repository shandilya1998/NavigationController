from utils.td3 import TD3, FeaturesExtractor, TD3Policy
from constants import params
from simulations.maze_env import MazeEnv
from simulations.point import PointEnv
from simulations.maze_task import CustomGoalReward4Rooms
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union, NamedTuple
import os
import cv2
import gym
import stable_baselines3 as sb3
import numpy as np
import torch
from utils import set_seeds

def linear_schedule(initial_value, final_value):
    """ 
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining):
        """ 
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * (initial_value - final_value) + final_value

    return func

def evaluate_policy(
    model: "base_class.BaseAlgorithm",
    env: Union[gym.Env, sb3.common.vec_env.VecEnv],
    n_eval_episodes: int = 10,
    deterministic: bool = True,
    render: bool = False,
    callback: Optional[Callable[[Dict[str, Any], Dict[str, Any]], None]] = None,
    reward_threshold: Optional[float] = None,
    return_episode_rewards: bool = False,
    warn: bool = True,
) -> Union[Tuple[float, float], Tuple[List[float], List[int]]]:
    """
    Runs policy for ``n_eval_episodes`` episodes and returns average reward.
    If a vector env is passed in, this divides the episodes to evaluate onto the
    different elements of the vector env. This static division of work is done to
    remove bias. See https://github.com/DLR-RM/stable-baselines3/issues/402 for more
    details and discussion.
    .. note::
        If environment has not been wrapped with ``Monitor`` wrapper, reward and
        episode lengths are counted as it appears with ``env.step`` calls. If
        the environment contains wrappers that modify rewards or episode lengths
        (e.g. reward scaling, early episode reset), these will affect the evaluation
        results as well. You can avoid this by wrapping environment with ``Monitor``
        wrapper before anything else.
    :param model: The RL agent you want to evaluate.
    :param env: The gym environment or ``VecEnv`` environment.
    :param n_eval_episodes: Number of episode to evaluate the agent
    :param deterministic: Whether to use deterministic or stochastic actions
    :param render: Whether to render the environment or not
    :param callback: callback function to do additional checks,
        called after each step. Gets locals() and globals() passed as parameters.
    :param reward_threshold: Minimum expected reward per episode,
        this will raise an error if the performance is not met
    :param return_episode_rewards: If True, a list of rewards and episode lengths
        per episode will be returned instead of the mean.
    :param warn: If True (default), warns user about lack of a Monitor wrapper in the
        evaluation environment.
    :return: Mean reward per episode, std of reward per episode.
        Returns ([float], [int]) when ``return_episode_rewards`` is True, first
        list containing per-episode rewards and second containing per-episode lengths
        (in number of steps).
    """
    is_monitor_wrapped = False
    # Avoid circular import
    from stable_baselines3.common.monitor import Monitor

    if not isinstance(env, sb3.common.vec_env.VecEnv):
        env = sb3.common.vec_env.dummy_vec_env.DummyVecEnv([lambda: env])

    is_monitor_wrapped = sb3.common.vec_env.is_vecenv_wrapped(env, sb3.common.vec_env.VecMonitor) or env.env_is_wrapped(Monitor)[0]

    if not is_monitor_wrapped and warn:
        warnings.warn(
            "Evaluation environment is not wrapped with a ``Monitor`` wrapper. "
            "This may result in reporting modified episode lengths and rewards, if other wrappers happen to modify these. "
            "Consider wrapping environment first with ``Monitor`` wrapper.",
            UserWarning,
        )

    n_envs = env.num_envs
    episode_rewards = []
    episode_lengths = []

    episode_counts = np.zeros(n_envs, dtype="int")
    # Divides episodes among different sub environments in the vector as evenly as possible
    episode_count_targets = np.array([(n_eval_episodes + i) // n_envs for i in range(n_envs)], dtype="int")

    current_rewards = np.zeros(n_envs)
    current_lengths = np.zeros(n_envs, dtype="int")
    observations = env.reset()
    states = None
    while (episode_counts < episode_count_targets).any():
        [actions, [gen_image, depth]], states = model.predict(observations, state=states, deterministic=deterministic)
        observations, rewards, dones, infos = env.step(actions)
        current_rewards += rewards
        current_lengths += 1
        for i in range(n_envs):
            if episode_counts[i] < episode_count_targets[i]:

                # unpack values so that the callback can access the local variables
                reward = rewards[i]
                done = dones[i]
                info = infos[i]

                if callback is not None:
                    callback(locals(), globals())

                if dones[i]:
                    if is_monitor_wrapped:
                        # Atari wrapper can send a "done" signal when
                        # the agent loses a life, but it does not correspond
                        # to the true end of episode
                        if "episode" in info.keys():
                            # Do not trust "done" with episode endings.
                            # Monitor wrapper includes "episode" key in info if environment
                            # has been wrapped with it. Use those rewards instead.
                            episode_rewards.append(info["episode"]["r"])
                            episode_lengths.append(info["episode"]["l"])
                            # Only increment at the real end of an episode
                            episode_counts[i] += 1
                    else:
                        episode_rewards.append(current_rewards[i])
                        episode_lengths.append(current_lengths[i])
                        episode_counts[i] += 1
                    current_rewards[i] = 0
                    current_lengths[i] = 0
                    if states is not None:
                        states[i] *= 0

        if render:
            env.render()

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    if reward_threshold is not None:
        assert mean_reward > reward_threshold, "Mean reward below threshold: " f"{mean_reward:.2f} < {reward_threshold:.2f}"
    if return_episode_rewards:
        return episode_rewards, episode_lengths
    return mean_reward, std_reward

class Callback(sb3.common.callbacks.EventCallback):
    """
    Callback for evaluating an agent.
    .. warning::
      When using multiple environments, each call to  ``env.step()``
      will effectively correspond to ``n_envs`` steps.
      To account for that, you can use ``eval_freq = max(eval_freq // n_envs, 1)``
    :param eval_env: The environment used for initialization
    :param callback_on_new_best: Callback to trigger
        when there is a new best model according to the ``mean_reward``
    :param n_eval_episodes: The number of episodes to test the agent
    :param eval_freq: Evaluate the agent every ``eval_freq`` call of the callback.
    :param log_path: Path to a folder where the evaluations (``evaluations.npz``)
        will be saved. It will be updated at each evaluation.
    :param best_model_save_path: Path to a folder where the best model
        according to performance on the eval env will be saved.
    :param deterministic: Whether the evaluation should
        use a stochastic or deterministic actions.
    :param render: Whether to render or not the environment during evaluation
    :param verbose:
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
        image_size : Tuple[int] = (1024, 1024),
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
                    cv2.VideoWriter_fourcc(*"MJPG"), 10, self.image_size, isColor = True
                )   
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
                    screen = self.eval_env.render(mode="rgb_array")
                    size = screen.shape[:2]
                    # PyTorch uses CxHxW vs HxWxC gym (and tensorflow) image convention
                    scale_1 = cv2.resize(
                        _locals['observations']['scale_1'][0, :3].transpose(1, 2, 0),
                        size
                    )
                    scale_2 = cv2.resize(
                        _locals['observations']['scale_2'][0, :3].transpose(1, 2, 0),
                        size
                    )
                    scale_3 = cv2.resize(
                        _locals['observations']['scale_3'][0, :3].transpose(1, 2, 0),
                        size
                    )
                    #print(_locals['gen_image'].shape)
                    gen_scale_1 = cv2.resize(
                        _locals['gen_image'][0, :3].transpose(1, 2, 0) * 255,
                        size
                    )

                    gen_scale_2 = cv2.resize(
                        _locals['gen_image'][0, 3:6].transpose(1, 2, 0) * 255,
                        size
                    )

                    gen_scale_3 = cv2.resize(
                        _locals['gen_image'][0, 6:].transpose(1, 2, 0) * 255,
                        size
                    )
                    gen_scale_1 = gen_scale_1.astype(np.uint8)
                    gen_scale_2 = gen_scale_2.astype(np.uint8)
                    gen_scale_3 = gen_scale_3.astype(np.uint8)

                    depth = _locals['observations']['depth'][0].transpose(1, 2, 0) * 255
                    depth = depth.astype(np.uint8)
                    depth = cv2.cvtColor(cv2.resize(
                        depth,
                        size
                    ), cv2.COLOR_GRAY2RGB)

                    gen_depth =  _locals['depth'][0].transpose(1, 2, 0) * 255
                    gen_depth = gen_depth.astype(np.uint8)
                    gen_depth = cv2.cvtColor(cv2.resize(
                        gen_depth,
                        size
                    ), cv2.COLOR_GRAY2RGB)

                    observation = np.concatenate([
                        np.concatenate([screen, scale_1, gen_scale_1], 0),
                        np.concatenate([depth, scale_2, gen_scale_2], 0),
                        np.concatenate([gen_depth, scale_3, gen_scale_3], 0)
                    ], 1).astype(np.uint8)
                    observation = cv2.cvtColor(observation, cv2.COLOR_RGB2BGR)

                    video.write(observation)

            episode_rewards, episode_lengths = evaluate_policy(
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

if __name__ == '__main__':

    set_seeds(params['seed'])
    logdir = '/content/drive/MyDrive/CNS/exp22'
    #logdir = 'assets/out/models/exp22'

    train_env = sb3.common.vec_env.vec_transpose.VecTransposeImage(
        sb3.common.vec_env.dummy_vec_env.DummyVecEnv([
            lambda : sb3.common.monitor.Monitor(
                MazeEnv(
                    PointEnv, CustomGoalReward4Rooms, 
                    params['max_episode_size'],
                    params['history_steps']
                )
            )
        ])
    )
    n_actions = train_env.action_space.sample().shape[-1]
    action_noise = sb3.common.noise.OrnsteinUhlenbeckActionNoise(
        params['OU_MEAN'] * np.ones(n_actions),
        params['OU_SIGMA'] * np.ones(n_actions),
        dt = params['dt']
    )

    policy_kwargs = {
        'net_arch' : params['net_arch'],
        'activation_fn' : torch.nn.Tanh,
        'features_extractor_class' : FeaturesExtractor,
        'features_extractor_kwargs' : {
            'features_dim' : params['num_ctx']
        },
        'normalize_images' : True,
        'optimizer_class' : torch.optim.Adam,
        'optimizer_kwargs' : None,
        'n_critics' : params['n_critics'],
        'share_features_extractor' : True
    }

    model = TD3(
        policy = TD3Policy,
        env = train_env,
        learning_rate = linear_schedule(params['lr'], params['final_lr']),
        buffer_size = params['buffer_size'],
        learning_starts = params['learning_starts'],
        batch_size = params['batch_size'],
        tau = params['tau'],
        gamma = params['gamma'],
        train_freq = (1, 'episode'),
        gradient_steps = -1,
        action_noise = action_noise,
        replay_buffer_class = sb3.common.buffers.DictReplayBuffer,
        replay_buffer_kwargs = None,
        optimize_memory_usage = False,
        policy_delay = params['policy_delay'],
        target_policy_noise = 0.2,
        target_noise_clip = 0.5,
        tensorboard_log = logdir,
        create_eval_env = False,
        policy_kwargs = policy_kwargs,
        seed = params['seed'],
        device = 'auto',
        _init_setup_model = True,
        verbose = 2,
    )

    env = MazeEnv(
        PointEnv, CustomGoalReward4Rooms,
        params['max_episode_size'],
        params['history_steps']
    )
    image_size = ( 
        int(3 * env.top_view_size * len(env._maze_structure[0])),
        int(3 * env.top_view_size * len(env._maze_structure))
    )

    eval_env = sb3.common.vec_env.vec_transpose.VecTransposeImage(
        sb3.common.vec_env.dummy_vec_env.DummyVecEnv([
            lambda : sb3.common.monitor.Monitor(env)
        ])
    )

    callbacks = sb3.common.callbacks.CallbackList([
        Callback(
            eval_env = eval_env,
            logdir = logdir,
            callback_on_new_best = None,
            n_eval_episodes = 5,
            eval_freq = params['eval_freq'],
            render_every = 2,
            image_size = image_size,
            log_path = logdir,
            best_model_save_path = None,
            deterministic = True,
            render = False,
            verbose = 2,
            warn = True,
        ),
        sb3.common.callbacks.CheckpointCallback(
            save_freq = params['save_freq'],
            save_path = logdir,
            name_prefix = 'rl_model',
            verbose = 2
        )
    ])

    model.learn(
        total_timesteps = params['total_timesteps'],
        callback = callbacks
    )

    print('Training Done.')
