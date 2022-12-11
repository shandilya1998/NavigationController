import torch
import numpy as np
import os

debug = False

params = {
    'error_threshold'             : 0.01,
    'input_size_low_level_control': 6,
    'track_list'                  : [
                                        'joint_pos',
                                        'action',
                                        'velocity',
                                        'position',
                                        'true_joint_pos',
                                        'sensordata',
                                        'qpos',
                                        'qvel',
                                        'achieved_goal',
                                        'observation',
                                        'heading_ctrl',
                                        'omega',
                                        'z',
                                        'mu',
                                        'd1',
                                        'd2',
                                        'd3',
                                        'stability',
                                        'omega_o',
                                        'reward',
                                        'rewards'
                                    ],
    'max_simulation_time'         : 5.0,
    'min_simulation_time'         : 1.0,
    'show_animation'              : False,
    'dt'                          : 0.1,
    'frame_skip'                  : 1,
    'learning_starts'             : int(1e2),
    'staging_steps'               : int(1e4),
    'imitation_steps'             : int(2e4),
    'render_freq'                 : int(4e4),
    'save_freq'                   : int(4e4),
    'eval_freq'                   : int(2e4),
    'buffer_size'                 : int(3e4),
    'max_episode_size'            : int(5e2),
    'max_seq_len'                 : 5,
    'seq_sample_freq'             : 5,
    'burn_in_seq_len'             : 5,
    'total_timesteps'             : int(1e6),
    'history_steps'               : 10,
    'net_arch'                    : [200, 150],
    'n_critics'                   : 3,
    'ds'                          : 0.01,
    'motor_cortex'                : [256, 128],
    'snc'                         : [256, 1],
    'af'                          : [256, 1],
    'critic_net_arch'             : [400, 300],
    'OU_MEAN'                     : 0.00,
    'OU_SIGMA'                    : 0.09,
    'OU_THETA'                    : 0.015,
    'top_view_size'               : 200,

    'batch_size'                  : 125,
    'lr'                          : 1e-3,
    'final_lr'                    : 1e-5,
    'n_steps'                     : 2000,
    'gamma'                       : 0.98,
    'tau'                         : 0.99,
    'n_updates'                   : 32,
    'num_ctx'                     : 128,
    'actor_lr'                    : 1e-3,
    'critic_lr'                   : 1e-2,
    'weight_decay'                : 1e-2,
    'collision_threshold'         : 20,
    'debug'                       : debug,
    'max_vyaw'                    : 1.5,
    'policy_delay'                : 5,
    'seed'                        : 281,
    'target_speed'                : 8.0,
    'lr_schedule_preprocesing'    : [
                                        {
                                            'name' : 'ExponentialLRSchedule',
                                            'class' : torch.optim.lr_scheduler.ExponentialLR,
                                            'kwargs' : {
                                                'gamma' : 0.99,
                                                'last_epoch' : - 1,
                                                'verbose' : False
                                            }
                                        }, {
                                            'name' : 'ReduceLROnPlateauSchedule',
                                            'class' : torch.optim.lr_scheduler.ReduceLROnPlateau,
                                            'kwargs' : {
                                                'mode' : 'min',
                                                'factor' : 0.5,
                                                'patience' : 10,
                                                'threshold' : 1e-5,
                                            }
                                        }
                                    ],
    'preprocessing'               : {
                                        'num_epochs'      : 1000
                                    },
    'lstm_steps'                  : 2,
    'autoencoder_arch'            : [1, 1, 1, 1],
    'add_ref_scales'              : False,
    'kld_weight'                  : 5e-4,
    'n_eval_episodes'             : 5,
}

params_quadruped = {
    'num_legs'                    : 4,
    'INIT_HEIGHT'                 : 0.05,
    'INIT_JOINT_POS'              : np.array([0.0, 0.0, 1.0, 0.0, 0.0, -1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 1.0], dtype = np.float32),
    'update_action_every'         : 1.0,
    'end_eff'                     : [5, 9, 13, 17],
    'degree'                      : 15,
    'alpha'                       : 0.6,
    'lambda'                      : 1,
    'beta'                        : 1.0,
    'delta'                       : 0.05,
    'DEFAULT_SIZE'                : 512,
}

params_environment = {
    "available_rgb"               : [
                                        [0.1, 0.1, 0.7],
                                        [0.1, 0.7, 0.1],
                                        [0.1, 0.7, 0.7],
                                        [0.7, 0.7, 0.1],
                                        [0.7, 0.1, 0.7],
                                        [0.1, 0.4, 1.0],
                                        [0.1, 1.0, 0.4]
                                    ],
    'available_shapes'            : ['sphere'],
    'target_shape'                : 'sphere',
    'target_rgb'                  : [0.7, 0.1, 0.1]
}

params_hopf = {
    'degree'                      : 15,
    'thresholds'                  : np.array([
                                        0.0,
                                        np.pi / 6,
                                        5 * np.pi / 6,
                                        np.pi,
                                        3 * np.pi / 2,
                                        3 * np.pi / 2 + np.pi / 6,
                                        2 * np.pi
                                    ], dtype = np.float32),
}

params.update(params_quadruped)
params.update(params_environment)
params.update(params_hopf)

image_height = 298
image_width = 298
image_channels = 3
n_history_steps = 5
action_dim = 2
num_legs = 4
action_dim = 8
units_osc = action_dim #60#action_dim#60 exp 68 units_osc = 8
cpg_param_size = 16 #16 #16 v1 to v4
input_size = 212 # 212 # 132 # 132 is size for results v 3
params.update({
    'num_legs'                    : num_legs,
    'end_eff'                     : [5, 9, 13, 17],
    'motion_state_size'           : 6,#:exp69, 6,#:exp 67,68, 3 #:exp66, 4 :exp64,65,
    'robot_state_size'            : 45,#:exp69, 111,#:exp67,68, 111 for stable_baselines model #4*action_dim + 4 + 8*3,
    'dt'                          : 0.005,
    'units_output_mlp'            : [60,100, 100, cpg_param_size],
    'input_size_low_level_control': input_size,
    'units_low_level_control'     : [256, 512, 256, 81],
    'cpg_param_size'              : cpg_param_size,
    'units_osc'                   : units_osc,
    'units_combine_rddpg'         : [200, units_osc],
    'units_combine'               : [200, units_osc],
    'units_robot_state'           : [145, 250, units_osc],
    'units_motion_state'          : [150, 75],
    'units_mu'                    : [150, 75],
    'units_mean'                  : [150, 75],
    'units_omega'                 : [150, 75],
    'units_dim_omega'             : 1,
    'units_robot_state_critic'    : [200, 120],
    'units_gru_rddpg'             : 400,
    'units_q'                     : 1,
    'actor_version'               : 2,
    'hopf_version'                : 1,
    'units_motion_state_critic'   : [200, 120],
    'units_action_critic'         : [200, 120],
    'units_history'               : 24,
    'BATCH_SIZE'                  : 50,
    'BUFFER_SIZE'                 : 1000000,
    'WARMUP'                      : 20000,
    'GAMMA'                       : 0.9,
    'TEST_AFTER_N_EPISODES'       : 25,
    'TAU'                         : 0.001,
    'decay_steps'                 : int(20),
    'LRA'                         : 1e-3,
    'LRC'                         : 1e-3,
    'EXPLORE'                     : 10000,
    'train_episode_count'         : 20000000,
    'test_episode_count'          : 5,
    'max_steps'                   : 1,
    'action_dim'                  : action_dim,
    'action_scale_factor'         : 1.2217,#:exp69, 1.0,
    'scale_factor_1'              : 0.1,
    'units_action_input'          : 20,
    'rnn_steps'                   : 128,
    'units_critic_hidden'         : 20,
    'lstm_units'                  : action_dim,
    'lstm_state_dense_activation' : 'relu',
    'L0'                          : 0.01738,
    'L1'                          : 0.025677,
    'L2'                          : 0.017849,
    'L3'                          : 0.02550,
    'g'                           : -9.81,
    'thigh'                       : 0.06200,
    'base_breadth'                : 0.04540,
    'friction_constant'           : 1e-4,
    'mu'                          : 0.001,
    'm1'                          : 0.010059,
    'm2'                          : 0.026074,
    'm3'                          : 0.007661,
    'future_steps'                : 4,
    'ou_theta'                    : 0.15,
    'ou_sigma'                    : 0.2,
    'ou_mu'                       : 0.0,
    'seed'                        : 1,
    'trajectory_length'           : 20,
    'window_length'               : 6,
    'num_validation_episodes'     : 5,
    'validate_interval'           : 20000,
    'update_interval'             : 2,
    'step_version'                : 1,
    'transition_items'            : ['ob','action','reward','next_ob','done', 'info'],
    'her_ratio'                   : 0.4,
    'render_train'                : True,
    'log_interval'                : 5000,
})

class TensorSpec:
    def __init__(self, shape, dtype, name = None):
        self.shape = shape
        self.dtype = dtype
        self.name = name

observation_spec = [
    TensorSpec(
        shape = (
            params['motion_state_size'],
        ),
        dtype = np.float32,
        name = 'motion_state_inp'
    ),
    TensorSpec(
        shape = (
            params['robot_state_size'],
        ),
        dtype = np.float32,
        name = 'robot_state_inp'
    ),
    TensorSpec(
        shape = (
            params['units_osc'] * 2,
        ),
        dtype = np.float32,
        name = 'oscillator_state_inp'
    ),
]

params_spec = [
    TensorSpec(
        shape = (
            params['rnn_steps'],
            params['action_dim']
        ),
        dtype = np.float32,
        name = 'quadruped action'
    )
] * 2

action_spec = [
    TensorSpec(
        shape = (
            params['rnn_steps'],
            params['action_dim']
        ),
        dtype = np.float32,
        name = 'quadruped action'
    ),
   TensorSpec(
       shape = (
           params['rnn_steps'],
           params['units_osc'] * 2,
       ),
       dtype = np.float32,
       name = 'oscillator action'
   )
]

reward_spec = [
    TensorSpec(
        shape = (),
        dtype = np.float32,
        name = 'reward'
    )
]

history_spec = TensorSpec(
    shape = (
        2 * params['rnn_steps'] - 1,
        params['action_dim']
    ),
    dtype = np.float32
)

history_osc_spec = TensorSpec(
    shape = (
        2 * params['rnn_steps'] - 1,
        2* params['units_osc']
    ),
    dtype = np.float32
)


data_spec = []
data_spec.extend(observation_spec)
data_spec.extend(action_spec)
data_spec.extend(params_spec)
data_spec.extend(reward_spec)
data_spec.append(observation_spec)

data_spec.extend([
    TensorSpec(
        shape = (),
        dtype = np.bool,
        name = 'done'
    )
])

specs = {
    'observation_spec' : observation_spec,
    'reward_spec' : reward_spec,
    'action_spec' : action_spec,
    'data_spec' : data_spec,
    'history_spec' : history_spec,
    'history_osc_spec' : history_osc_spec
}

params.update(specs)

robot_data = {
    'leg_name_lst' : [
        'front_right_leg',
        'front_left_leg',
        'back_right_leg',
        'back_left_leg'
    ],
    'link_name_lst' :  [
        'quadruped::base_link',
        'quadruped::front_right_leg1',
        'quadruped::front_right_leg2',
        'quadruped::front_right_leg3',
        'quadruped::front_left_leg1',
        'quadruped::front_left_leg2',
        'quadruped::front_left_leg3',
        'quadruped::back_right_leg1',
        'quadruped::back_right_leg2',
        'quadruped::back_right_leg3',
        'quadruped::back_left_leg1',
        'quadruped::back_left_leg2',
        'quadruped::back_left_leg3'
    ],
    'joint_name_lst' : [
        'front_right_leg1_joint',
        'front_right_leg2_joint',
        'front_right_leg3_joint',
        'front_left_leg1_joint',
        'front_left_leg2_joint',
        'front_left_leg3_joint',
        'back_right_leg1_joint',
        'back_right_leg2_joint',
        'back_right_leg3_joint',
        'back_left_leg1_joint',
        'back_left_leg2_joint',
        'back_left_leg3_joint'
    ],
    'starting_pos' : np.array([
        -0.01, 0.01, 0.01,
        -0.01, 0.01, -0.01,
        -0.01, 0.01, -0.01,
        -0.01, 0.01, 0.01
    ], dtype = np.float32),
    'L' : 2.2*0.108,
    'W' : 2.2*0.108
}

params.update(robot_data)

Tsw = [i for i in range(80, 260, 4)]
Tsw = Tsw + Tsw + Tsw
Tst = [i*3 for i in Tsw]
Tst = Tst + Tst + Tst
theta_h = [30 for i in range(len(Tsw))] + [45 for i in range(len(Tsw))] + \
        [60 for i in range(len(Tsw))]
theta_k = [30 for i in range(len(Tsw))] + [30 for i in range(len(Tsw))] + \
        [30 for i in range(len(Tsw))]

pretraining = {
    'Tst' : Tst,
    'Tsw' : Tsw,
    'theta_h' : theta_h,
    'theta_k' : theta_k
}


num_data = 135
params.update(pretraining)
bs = 15
#num_data =135 * params['rnn_steps'] * params['max_steps']
params.update({
    'num_data' : num_data,
    'pretrain_bs': bs,
    'train_test_split' : (num_data - bs) / num_data,
    'pretrain_test_interval' : 3,
    'pretrain_epochs' : 500,
    'pretrain_ds_path': 'data/pretrain_rddpg_6'
    })


params_ars = {
    'nb_steps'                    : 1000,
    'episode_length'              : 300,
    'learning_rate'               : 0.001,
    'nb_directions'               : 12,
    'rnn_steps'                   : 1,
    'nb_best_directions'          : 7,
    'noise'                       : 0.0003,
    'seed'                        : 1,
}

params_per = {
    'alpha'                       : 0.6,
    'epsilon'                     : 1.0,
    'beta_init'                   : 0.4,
    'beta_final'                  : 1.0,
    'beta_final_at'               : params['train_episode_count'],
    'step_size'                   : params['LRA'] / 4
}

params.update(params_per)

params_env = {
    'DEFAULT_SIZE'                : 500,
    'INIT_HEIGHT'                 : 0.05,
    'dt'                          : 0.005,
    'reward_energy_coef'          : 1e-3,
    'reward_velocity_coef'        : 1,
    'update_action_every'         : 1.0,
    'INIT_JOINT_POS'              : np.array([0.0, 0.0, 1.0, 0.0, 0.0, -1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 1.0], dtype = np.float32)
}


max_steps = 2000 #data collection env
n_envs = 1
params_rl = {
    'MAX_STEPS'                   : max_steps,
    'OU_MEAN'                     : 0.0,
    'OU_SIGMA'                    : 0.08,
    'BATCH_SIZE'                  : 50, #max_steps * n_envs ,#8,
    'NET_ARCH'                    : [dict(pi=[256, 256], vf=[256, 256])],
    'POLICY_TYPE'                 : "MultiInputPolicy",
    'LEARNING_STARTS'             : 2000,
    'TRAIN_FREQ'                  : [1, 'steps'],
    'CHECK_FREQ'                  : 6000,
    'sde_sample_freq'             : 4,
    'n_epochs'                    : 100,
    'gae_lambda'                  : 0.9,
    'clip_range'                  : 0.4,
    'vf_coef'                     : 0.4,
    'LEARNING_RATE'               : 0.0001,
    'gamma'                       : 0.9,
    'tau'                         : 0.02,
    'steps'                       : int(1e7),
    'n_steps'                     : 8,
    'n_envs'                      : n_envs,
    'dt'                          : 0.005,
    'max_step_length'             : 600,
    'gait_list'                   : [
                                        'ds_crawl',
                                        'ls_crawl',
                                        'trot',
                                        'pace',
                                        'bound',
                                        'transverse_gallop',
                                        'rotary_gallop'
                                    ],
    'task_list'                   : [
                                        'straight',
                                        'rotate',
                                        'turn',
                                        'roll',
                                        'pitch',
                                        'squat'
                                    ],
    'direction_list'              : [
                                        'forward',
                                        'backward',
                                        'left',
                                        'right',
                                    ],
    'track_list'                  : [
                                        'joint_pos',
                                        'action',
                                        'velocity',
                                        'position',
                                        'true_joint_pos',
                                        'sensordata',
                                        'qpos',
                                        'qvel',
                                        'achieved_goal',
                                        'observation',
                                        'heading_ctrl',
                                        'omega',
                                        'z',
                                        'mu',
                                        'd1',
                                        'd2',
                                        'd3',
                                        'stability',
                                        'omega_o',
                                        'reward',
                                        'rewards'
                                    ],
    'ref_path'                    : os.path.join('assets', 'out', 'reference'),
    'env_name'                    : 'Quadruped',
    'osc_omega_offset'            : 0.0
}


action_dim = 8
units_osc = 60#action_dim#60 exp 68 units_osc = 8
params_pretrain = {
    'action_dim'                  : action_dim,
    'batch_size'                  : 50,
    'n_epochs'                    : 100,
    'n_steps'                     : 3500,
    'n_update_steps'              : 20,
    'n_eval_steps'                : 5,
    'n_episodes'                  : 500,
    'min_epoch_size'              : 1000,
    'motion_state_size'           : 6,#:exp69, 6,#:exp 67,68, 3 #:exp66, 4 :exp64,65,
    'robot_state_size'            : 18,#:exp69, 111,#:exp67,68, 111 for stable_baselines model #4*action_dim + 4 + 8*3,
    'dt'                          : 0.005,
    'units_output_mlp'            : [256,512,100,action_dim],
    'units_osc'                   : units_osc,
    'units_combine_rddpg'         : [200, units_osc],
    'units_combine'               : [200, units_osc],
    'units_robot_state'           : [145, 250, units_osc],
    'units_motion_state'          : [150, 75],
    'units_mu'                    : [150, 75],
    'units_mean'                  : [150, 75],
    'units_omega'                 : [150, 75],
    'units_dim_omega'             : 1,
    'units_robot_state_critic'    : [200, 120],
    'units_gru_rddpg'             : 400,
    'units_q'                     : 1,
    'actor_version'               : 2,
    'hopf_version'                : 1,
    'units_motion_state_critic'   : [200, 120],
    'units_action_critic'         : [200, 120],
    'units_history'               : 24,
    'max_action'                  : 2.0,
}

params_conv = {
    'h'                           : 250,
    'w'                           : params_pretrain['motion_state_size'],
    'stride'                      : 100,
    'window_length'               : 2500,
}

params.update(params_env)
params.update(params_rl)
params.update(params_pretrain)
params.update(params_conv)

params.update({
    'version'                     : 1,
    'offset'                      : np.array([0.25, 0.25, 0.25, 0.25], dtype = np.float32),
    'degree'                      : 15,
    'thresholds'                  : np.array([
                                        0.0,
                                        np.pi / 6,
                                        5 * np.pi / 6,
                                        np.pi,
                                        3 * np.pi / 2,
                                        3 * np.pi / 2 + np.pi / 6,
                                        2 * np.pi
                                    ], dtype = np.float32),
    'camera_name'                 : 'esp32cam',
    'memory_size'                 : 10,
    'data_gen_granularity'        : 350,
    'window_size'                 : 600,
    'scheduler_update_freq'       : 5,
    'observation_version'         : 0,
    'max_epoch_size'              : 100,
    'env_version'                 : 1,
    'coupling_strength'           : 2.5,
    'weight_net_units'            : [256, 512, 1024, 512, 256],
    'save_freq'                   : 1000,
    'render_freq'                 : 5000,
    'eval_freq'                   : 2000,
    'total_timesteps'             : int(1e6),
    'max_episode_size'            : 500,
    'props'                       : {
                                        'ls_crawl' : {
                                            'omega' : [np.pi / 6, 5 * np.pi / 6],
                                            'mu' : [0.10, 0.75]
                                        },
                                        'ds_crawl' : {
                                            'omega' : [np.pi / 6, 5 * np.pi / 6],
                                            'mu' : [0.10, 0.75]
                                        },
                                        'trot' : {
                                            'omega' : [np.pi, 3 * np.pi / 2],
                                            'mu' : [0.1, 0.5]
                                        },
                                    },
    'lambda'                      : 1,
    'beta'                        : 1.0,
    'alpha'                       : 0.5,
    'power'                       : 0.64
})

if params['observation_version'] == 0:
    params['input_size_low_level_control'] = 132
elif params['observation_version'] == 1:
    params['input_size_low_level_control'] = 212
elif params['observation_version'] == 2:
    params['input_size_low_level_control'] = 292
else:
    raise ValueError(
        'Expected one of `0`, `1` or `2`, got {}'.format(params['obervation_version'])
    )

if params['env_version'] == 0:
    params['cpg_param_size'] = 16
elif params['env_version'] == 1:
    params['cpg_param_size'] = 12
else:
    raise ValueError(
        'Expected one of `0` or `1`, got {}'.format(params['env_version'])
    )


params['gait_list'] = ['trot', 'ds_crawl', 'ls_crawl']
params['task_list'] = ['straight', 'rotate', 'turn']
params['direction_list'] = ['forward', 'backward', 'left', 'right']


params_obstacles = {
    'num_obstacles'               : 500,
    'max_height'                  : 0.03,
    'max_size'                    : 0.2,
}
params.update(params_obstacles)


params_floquet = {
    'tmax' : 100,
}

params.update(params_floquet)
