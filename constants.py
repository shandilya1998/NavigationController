import torch
import numpy as np

debug = False

params = {
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
    'dt'                          : 0.02,
    'learning_starts'             : int(1e2),
    'staging_steps'               : int(1e4),
    'imitation_steps'             : int(2e4),
    'render_freq'                 : int(4e4),
    'save_freq'                   : int(4e4),
    'eval_freq'                   : int(2e4),
    'buffer_size'                 : int(1.75e5),
    'max_episode_size'            : int(5e2),
    'max_seq_len'                 : 5,
    'seq_sample_freq'             : 5,
    'burn_in_seq_len'             : 5,
    'total_timesteps'             : int(1e6),
    'history_steps'               : 50,
    'net_arch'                    : [400, 300],
    'n_critics'                   : 2,
    'ds'                          : 0.01,
    'motor_cortex'                : [256, 128],
    'snc'                         : [256, 1],
    'af'                          : [256, 1],
    'critic_net_arch'             : [400, 300],
    'OU_MEAN'                     : 0.00,
    'OU_SIGMA'                    : 0.2,
    'OU_THETA'                    : 0.015,
    'top_view_size'               : 200,

    'batch_size'                  : 64,
    'lr'                          : 1e-3,
    'final_lr'                    : 1e-5,
    'n_steps'                     : 2000,
    'gamma'                       : 0.98,
    'tau'                         : 0.99, 
    'n_updates'                   : 32,
    'num_ctx'                     : 512,
    'actor_lr'                    : 1e-3,
    'critic_lr'                   : 1e-2,
    'weight_decay'                : 1e-2,
    'collision_threshold'         : 20,
    'debug'                       : debug,
    'max_vyaw'                    : 1.5,
    'policy_delay'                : 2,
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

params.update(params_quadruped)
params.update(params_environment)

image_height = 298
image_width = 298
image_channels = 3
n_history_steps = 5
action_dim = 2
