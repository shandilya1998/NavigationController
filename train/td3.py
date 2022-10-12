import os
import shutil
from typing import Callable, Type, Dict, Union
from neurorobotics.simulations.maze_env import Environment
from neurorobotics.simulations.agent_model import AgentModel
from neurorobotics.utils.callbacks import Callback
import stable_baselines3 as sb3

def train(
        env_class: Type[Environment],
        agent_class: Type[AgentModel],
        task_generator: Callable,
        policy_class: Union[str, sb3.common.policies.BasePolicy],
        params: Dict,
        lr_schedule: sb3.common.type_aliases.Schedule,
        action_noise: sb3.common.noise.ActionNoise,
        policy_kwargs: Dict,
        device: str = 'auto',
        logdir: str = 'assets/outputs/',
        ):
    """Executes training scripts according to provided config
    
    :param env_class: class of env to spawn
    :type env_class: Type[Environment]
    :param policy_class: policy class to train
    :type policy_class:
    :param params: parameters to be fed to the method. imported from neurorobotics/constants.py
    :param type: Dict
    :param lr_schedule: learning rate schedule for training model
    :type lr_schedule: sb3.common.type_aliases.Schedule
    :param action_noise: action noise for td3 model
    :type action_noise: sb3.common.noise.ActionNoise
    :param logdir: path of output directory
    :type param: str
    """
    if os.path.exists(logdir):
        shutil.rmtree(logdir)
    os.makedirs(logdir)
    os.mkdir(os.path.join(logdir, 'models'))
    os.mkdir(os.path.join(logdir, 'plots'))
    os.mkdir(os.path.join(logdir, 'videos'))

    train_env = env_class(
           model_cls=agent_class,
           maze_task_generator=task_generator,
           max_episode_size=params['max_episode_size'],
           n_steps=params['history_steps'],
           frame_skip=params['frame_skip']
    )

    
    image_size = ( 
        int(2 * train_env.top_view_size * len(train_env._maze_structure[0])),
        int(2 * train_env.top_view_size * len(train_env._maze_structure))
    )

    train_env = sb3.common.vec_env.vec_transpose.VecTransposeImage(
            sb3.common.vec_env.dummy_vec_env.DummyVecEnv([
                    lambda: sb3.common.monitor.Monitor(train_env)
                    ])
            )


    eval_env = env_class(
           model_cls=agent_class,
           maze_task_generator=task_generator,
           max_episode_size=params['max_episode_size'],
           n_steps=params['history_steps'],
           frame_skip=params['frame_skip']
    )

    eval_env = sb3.common.vec_env.vec_transpose.VecTransposeImage(
            sb3.common.vec_env.dummy_vec_env.DummyVecEnv([
                    lambda: sb3.common.monitor.Monitor(eval_env)
                    ])
            )

    model = sb3.TD3(
            policy=policy_class,
            env=train_env,
            learning_rate=lr_schedule,
            buffer_size=params['buffer_size'],
            learning_starts=params['learning_starts'],
            batch_size=params['batch_size'],
            tau=params['tau'],
            gamma=params['gamma'],
            train_freq=(1, 'episode'),
            gradient_steps=-1,
            action_noise=action_noise,
            replay_buffer_class=sb3.common.buffers.DictReplayBuffer,
            replay_buffer_kwargs=None,
            optimize_memory_usage=False,
            policy_delay=params['policy_delay'],
            target_policy_noise=0.2,
            target_noise_clip=0.5,
            tensorboard_log=logdir,
            create_eval_env=False,
            policy_kwargs=policy_kwargs,
            seed=params['seed'],
            device=device,
            _init_setup_model=True,
            verbose=2,
    )

    callbacks = sb3.common.callbacks.CallbackList([
        sb3.common.callbacks.CheckpointCallback(
            save_freq=params['save_freq'],
            save_path=os.path.join(logdir, 'models'),
            name_prefix='td3_model',
            verbose=2
        ),
        Callback(
            eval_env=eval_env,
            logdir=os.path.join(logdir, 'videos'),
            n_eval_episodes=5,
            eval_freq=params['eval_freq'],
            render_every=2,
            image_size=image_size,
            log_path=os.path.join(logdir, 'evaluations'),
            best_model_save_path=os.path.join(logdir, 'models'),
            verbose=2,
            warn=True)
    ])

    model.learn(
        total_timesteps=params['total_timesteps'],
        callback=callbacks
    )
    model.save(os.path.join(logdir, "final_model"))

    return model
