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
    args = parser.parse_args()
    if args.learning_type == 'imitate':
        model = Imitate(
            logdir = args.logdir,
            max_episode_size = args.max_episode_size
        )
        model.learn(args.timesteps)
    elif args.learning_type == 'explore':
        model = Explore(
            logdir = args.logdir,
            max_episode_size = args.max_episode_size
        )
        model.learn(args.timesteps)
    else:
        raise ValueError('Expected one of `explore` or `imitate`, got {}'.format(args.learning_type))
