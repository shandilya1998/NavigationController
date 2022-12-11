import shutil
import cv2
import matplotlib.pyplot as plt
import  numpy as np
import argparse
from neurorobotics.constants import params
import os


def test_env(out_path, env, render = True):
    t = 0
    ac = env.action_space.sample()
    omega = 3.55
    if env.gait == 'trot':
        if env._action_dim == 2:
            mu = (np.random.random() * 1.5 ) % 1
            omega = 3.55
            ac = np.array([ omega / (2 * np.pi), mu])
        elif env._action_dim == 4:
            omega = 3.55
            mu1 = np.random.random()
            mu2 = np.random.uniform(low = mu1, high = 1.0)
            if env.direction == 'left':
                mu = np.array([mu2, mu1], dtype = np.float32)
                ac = np.array([(omega) / (2 * np.pi), mu[0], omega / (2 * np.pi), mu[1]], dtype = np.float32)
            elif env.direction == 'right':
                mu = np.array([mu1, mu2], dtype = np.float32)
                ac = np.array([omega / (2 * np.pi), mu[0], (omega) / (2 * np.pi), mu[1]], dtype = np.float32)
    if 'crawl' in env.gait:
        if env._action_dim == 2:
            mu = 0.5
            omega = 1.6
            ac = np.array([omega / (2 * np.pi), mu])
        elif env._action_dim == 4:
            omega = 1.6
            mu1 = np.random.random()
            mu2 = np.random.uniform(low = mu1, high = 1.0)
            if env.direction == 'left':
                mu = np.array([mu2, mu1], dtype = np.float32)
                ac = np.array([omega / (2 * np.pi), mu[0], (omega)/ (2 * np.pi), mu[1]], dtype = np.float32)
            elif env.direction == 'right':
                mu = np.array([mu1, mu2], dtype = np.float32)
                ac = np.array([omega / (2 * np.pi), mu[0], omega / (2 * np.pi), mu[1]], dtype = np.float32)
    frame_rate = 10
    frame_width = 320
    frame_height = 240
    writer = cv2.VideoWriter(
        os.path.join(out_path, "video.avi"),
        cv2.VideoWriter_fourcc(*'MJPG'),
        frame_rate,
        (frame_width, frame_height)
    )
    while True:
        _ = env.step(ac)
        if render:
            frame = env.render(mode='rgb_array', width=frame_width, height=frame_height)
            writer.write(frame)
        t += 1
        if t > 100:
            break
    writer.release()
    fig, axes = plt.subplots(4,3, figsize = (15, 20))
    i = 0
    joint_pos = np.nan_to_num(np.vstack(env._track_item['joint_pos']))
    true_joint_pos = np.nan_to_num(np.vstack(env._track_item['true_joint_pos']))
    num_joints = joint_pos.shape[-1]
    t = np.arange(joint_pos.shape[0], dtype = np.float32) * env.dt
    while True:
        if i >= num_joints:
            break
        num_steps = int(np.pi / (omega * params['dt']))
        axes[int(i / 3)][i % 3].plot(t[:num_steps], joint_pos[:num_steps, i], color = 'r', label = 'Input')
        axes[int(i / 3)][i % 3].plot(t[:num_steps], true_joint_pos[:num_steps, i], color = 'b', linestyle = '--', label = 'Response')
        axes[int(i / 3)][i % 3].set_title('Joint {}'.format(i))
        axes[int(i / 3)][i % 3].set_xlabel('time (s)')
        axes[int(i / 3)][i % 3].set_ylabel('joint position (radian)')
        axes[int(i / 3)][i % 3].legend()
        i += 1
    fig.savefig(os.path.join(out_path, 'ant_joint_pos.png'))
    env.reset()
    return env


if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_path", type=str, help="output path of saved files.")
    parser.add_argument("--gait", type=str, help="gait to perform, one of trot, ds_crawl, ls_crawl, trot")
    parser.add_argument("--task", type=str, help="task to perform, one of turn, straight, rotate")
    parser.add_argument("--direction", type=str, help="direction of motion, one of left, right, forward, backward")
    args = parser.parse_args()

    from neurorobotics.simulations import Quadruped
    env = Quadruped(
        model_path = 'quadruped.xml',
        frame_skip = 3,
        render = True,
        gait = args.gait,
        task = args.task,
        direction = args.direction,
    )
    #env = QuadrupedV2()
    out_path = os.path.join(args.out_path, "quadruped")
    if os.path.exists(out_path):
        shutil.rmtree(out_path)
    os.mkdir(out_path)
    env = test_env(out_path, env, True)
