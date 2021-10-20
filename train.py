from learning.imitate import Imitate

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
        '--batch_size',
        type = int,
        help = 'batch size'
    )
    parser.add_argument(
        '--max_episode_size',
        type = int,
        help = 'maximum episode size'
    )
    args = parser.parse_args()
    model = Imitate(
        logdir = args.logdir,
        batch_size = args.batch_size,
        max_episode_size = args.max_episode_size
    )
    model.learn(args.timesteps)
