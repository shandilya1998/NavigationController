from learning.imitate import Imitate
from learning.explore import Explore
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--logdir',
        type = str,
        help = 'relative path to directory with data'
    )
    parser.add_argument(
        '--timesteps',
        type = int,
        help = 'number of timesteps to run learning for'
    )
    parser.add_argument(
        '--max_episode_size',
        type = int,
        help = 'maximum episode size'
    )
    parser.add_argument(
        '--learning_type',
        type = str,
        help = 'choose between imitate and explore'
    )
    parser.add_argument(
        '--policy_version',
        type = int,
        help = 'policy version to be used'
    )
    parser.add_argument(
        '--env_type',
        type = str,
        help = 'choose between maze or collision. choice modifies reward'
    )
    parser.add_argument(
        '--history_steps',
        type = int,
        help = 'number of images in observation input to policy',
        default = 4
    )
    parser.add_argument(
        '--task_version',
        type = int,
        help = 'choose between the different obstacle tasks'
    )
    parser.add_argument(
        '--n_steps',
        type = int,
        help = 'number of steps of for n step TD return'
    )
    parser.add_argument(
        '--lmbda',
        type = float,
        help = 'value of lambda to choose between TD(0), TD(1) or TD(λ) learning'
    )
    args = parser.parse_args()
    if args.learning_type == 'imitate':
        model = Imitate(
            logdir = args.logdir,
            max_episode_size = args.max_episode_size,
            policy_version = args.policy_version,
            env_type = args.env_type,
            history_steps = args.n_steps,
            task_version = args.task_version,
            n_steps = args.n_steps,
            lmbda = args.lmbda
        )
        model.learn(args.timesteps)
    elif args.learning_type == 'explore':
        model = Explore(
            logdir = args.logdir,
            max_episode_size = args.max_episode_size,
            policy_version = args.policy_version,
            env_type = args.env_type,
            history_steps = args.n_steps,
            task_version = args.task_version,
            n_steps = args.n_steps,
            lmbda = args.lmbda
        )
        model.learn(args.timesteps)
    else:
        raise ValueError('Expected one of `explore` or `imitate`, got {}'.format(args.learning_type))
