from simulations.maze_env import MazeEnv
from simulations.maze_task import CustomGoalReward4Rooms
from simulations.point import PointEnv
from constants import params
import stable_baselines3 as sb3
from utils.rtd3 import DictReplayBuffer
import torch
import numpy as np
from tqdm import tqdm
import cv2
import time

_env = MazeEnv(
    PointEnv,
    CustomGoalReward4Rooms,
    params['max_episode_size'],
    params['history_steps']
)

env = sb3.common.vec_env.vec_transpose.VecTransposeImage(
    sb3.common.vec_env.dummy_vec_env.DummyVecEnv([
        lambda: sb3.common.monitor.Monitor(
            _env
        )
    ])
)


state_spec = [((100,), np.float32)]
buff = DictReplayBuffer(
    buffer_size=int(1e4),
    observation_space=env.observation_space,
    action_space=env.action_space,
    device='cpu',
    n_envs=1,
    state_spec=state_spec,
    max_seq_len=100,
    burn_in_seq_len=40
)

count = 0
for i in tqdm(range(5)):
    done = np.array([0])
    ob = env.reset()
    while not done:
        states = np.random.uniform(low = 0, high = 1, size=(1,) + state_spec[0][0])
        next_ob, reward, done, info = env.step(ob['sampled_action'])
        buff.add(
            obs=ob,
            next_obs=next_ob,
            action=ob['sampled_action'],
            reward=reward,
            done=done,
            states=states,
            infos=info
        )
        ob = next_ob
        count += 1

for i in range(count):
    scale_1 = cv2.resize(
        buff.observations['scale_1'][i, 0].transpose(1,2,0),
        (320, 320)
    )
    scale_2 = cv2.resize(
        buff.observations['scale_1'][i, 0].transpose(1,2,0),
        (320, 320)
    )
    cv2.imshow('scale_1', scale_1)
    cv2.imshow('scale_2', scale_2)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break


for j in range(int(1e5)):
    replay = buff.sample(batch_size=1)
    total_steps = replay.prev_observations['scale_1'][0].shape[0] + replay.observations['scale_1'][0].shape[0]
    scale_1 = replay.prev_observations['scale_1'][0].cpu().detach().numpy().transpose(0, 2, 3, 1)
    scale_2 = replay.prev_observations['scale_2'][0].cpu().detach().numpy().transpose(0, 2, 3, 1)
    steps = scale_1.shape[0]

    bar = tqdm(total = total_steps)
    for i in range(steps):
        scale_1_ = cv2.resize(scale_1[i], (320, 320))
        cv2.imshow('scale_1', scale_1_)
        scale_2_ = cv2.resize(scale_2[i], (320, 320))
        cv2.imshow('scale_2', scale_2_)
        bar.update(1)
        if cv2.waitKey(1) & 0xff == ord('q'):
            break

    scale_1 = replay.observations['scale_1'][0].cpu().detach().numpy().transpose(0, 2, 3, 1)
    scale_2 = replay.observations['scale_2'][0].cpu().detach().numpy().transpose(0, 2, 3, 1)
    steps = scale_1.shape[0]

    for i in range(steps):
        scale_1_ = cv2.resize(scale_1[i], (320, 320))
        cv2.imshow('scale_1', scale_1_)
        scale_2_ = cv2.resize(scale_2[i], (320, 320))
        cv2.imshow('scale_2', scale_2_)
        bar.update(1)
        if cv2.waitKey(1) & 0xff == ord('q'):
            break
    bar.close()


cv2.destroyAllWindows()
