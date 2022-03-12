import torch
import numpy as np

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
    'learning_starts'             : 750,
    'staging_steps'               : int(6e4),
    'imitation_steps'             : int(1.2e5),
    'render_freq'                 : int(3e4),
    'save_freq'                   : int(3e4),
    'eval_freq'                   : int(7.5e3),
    'buffer_size'                 : int(2e5),
    'max_episode_size'            : int(7.5e2),
    'total_timesteps'             : int(1e6),
    'history_steps'               : 15,
    'net_arch'                    : [150, 300, 150],
    'n_critics'                   : 2,
    'ds'                          : 0.01,
    'motor_cortex'                : [256, 128],
    'snc'                         : [256, 1],
    'af'                          : [256, 1],
    'critic_net_arch'             : [400, 300],
    'OU_MEAN'                     : 0.00,
    'OU_SIGMA'                    : 0.12,
    'top_view_size'               : 50.,
    'batch_size'                  : 100,
    'lr'                          : 1e-3,
    'final_lr'                    : 1e-5,
    'n_steps'                     : 2000,
    'gamma'                       : 0.98,
    'tau'                         : 0.002, 
    'n_updates'                   : 32,
    'num_ctx'                     : 400,
    'actor_lr'                    : 1e-3,
    'critic_lr'                   : 1e-2,
    'weight_decay'                : 1e-2,
    'collision_threshold'         : 20,
    'debug'                       : False,
    'max_vyaw'                    : 1.5,
    'policy_delay'                : 2,
    'seed'                        : 245,
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
    'add_ref_scales'              : True,
    'stage1'                      : int(2e5),
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

params.update(params_quadruped)

import tensorflow as tf
import tf_agents as tfa

image_height = 320
image_width = 320
image_channels = 3
n_history_steps = 5
activation_fn_actor = tf.keras.activations.relu
activation_fn_critic = tf.keras.activations.relu
action_dim = 2
tf_params = {
    'image_height'                : image_height,
    'image_width'                 : image_width,
    'image_channels'              : image_channels,
    'action_dim'                  : action_dim,
    'visual_cortex'               : [
                                        {
                                            'class' : tf.keras.layers.Conv2D,
                                            'kwargs' : {
                                                'filters' : 128,
                                                'kernel_size' : 8,
                                                'strides' : 4,
                                                'activation' : activation_fn_actor,
                                                'input_shape' : (image_height, image_width, image_channels)
                                            }
                                        },
                                        {
                                            'class' : tf.keras.layers.Conv2D,
                                            'kwargs' : {
                                                'filters' : 256,
                                                'kernel_size' : 4,
                                                'strides' : 2,
                                                'activation' : activation_fn_actor
                                            }
                                        },
                                        {
                                            'class' : tf.keras.layers.Conv2D,
                                            'kwargs' : {
                                                'filters' : 144,
                                                'kernel_size' : 4,
                                                'strides' : 2,
                                                'activation' : activation_fn_actor
                                            }
                                        },
                                        {
                                            'class' : tf.keras.layers.Conv2D,
                                            'kwargs' : {
                                                'filters' : 32,
                                                'kernel_size' : 3,
                                                'strides' : 1,
                                                'activation' : activation_fn_actor
                                            }
                                        },
                                        {
                                            'class' : tf.keras.layers.Flatten,
                                            'kwargs' : {}
                                        },
                                        {
                                            'class' : tf.keras.layers.Dense,
                                            'kwargs' : {
                                                'units' : 512,
                                                'activation' : activation_fn_actor,
                                                'kernel_initializer' : 'glorot_uniform',
                                                'bias_initializer' : 'glorot_uniform'
                                            }
                                        }
                                    ],
    'proprioreceptive_cortex'     : [
                                        {
                                            'class' : tf.keras.layers.Dense,
                                            'kwargs' : {
                                                'units' : (4 + 4 * n_history_steps) * 4,
                                                'activation' : activation_fn_actor,
                                                'kernel_initializer' : 'glorot_uniform',
                                                'bias_initializer' : 'glorot_uniform'
                                            }
                                        },
                                        {
                                            'class' : tf.keras.layers.Dense,
                                            'kwargs' : {
                                                'units' : (4 + 4 * n_history_steps) * 3,
                                                'activation' : activation_fn_actor,
                                                'kernel_initializer' : 'glorot_uniform',
                                                'bias_initializer' : 'glorot_uniform'
                                            }
                                        }
                                    ],
    'action_preprocessing_layers' : [ 
                                        {   
                                            'class' : tf.keras.layers.Dense,
                                            'kwargs' : { 
                                                'units' : (4 + 4 * n_history_steps) * 4,
                                                'activation' : activation_fn_actor,
                                                'kernel_initializer' : 'glorot_uniform',
                                                'bias_initializer' : 'glorot_uniform'
                                            }   
                                        },  
                                        {   
                                            'class' : tf.keras.layers.Dense,
                                            'kwargs' : { 
                                                'units' : (4 + 4 * n_history_steps) * 3,
                                                'activation' : activation_fn_actor,
                                                'kernel_initializer' : 'glorot_uniform',
                                                'bias_initializer' : 'glorot_uniform'
                                            }   
                                        }   
                                    ],
    'input_fc_layer_params_actor'       : [1024, 768, 512],
    'lstm_size_actor'             : [256, 128],
    'output_fc_layer_params_actor': [64, 32],
    'activation_fn_actor'         : activation_fn_actor,
    'activation_fn_critic'        : activation_fn_critic,
    'input_fc_layer_params_critic'       : [1024, 768, 512],
    'lstm_size_critic'             : [256, 128],
    'output_fc_layer_params_critic': [64, 32],

    'optimizer_class_actor'        : tf.keras.optimizers.Adam,
    'optimizer_kwargs_actor'       : {
                                        'learning_rate' : 1e-2,
                                        'beta_1'        : 0.9,
                                        'beta_2'        : 0.999,
                                        'epsilon'       : 1e-7,
                                        'amsgrad'       : False,
                                        'name'          : 'adam_actor'
                                    },
    'optimizer_class_critic'        : tf.keras.optimizers.Adam,
    'optimizer_kwargs_critic'       : { 
                                        'learning_rate' : 1e-2,
                                        'beta_1'        : 0.9,
                                        'beta_2'        : 0.999,
                                        'epsilon'       : 1e-7,
                                        'amsgrad'       : False,
                                        'name'          : 'adam_critic'
                                    },
    'exploration_noise_std'         : 0.1,
    'target_update_tau'             : 0.01,
    'target_update_period'          : 4,
    'actor_update_period'           : 2,
    'gamma'                         : 0.98,
    'reward_scale_factor'           : 1.0,
    'target_policy_noise'           : 0.2,
    'target_policy_noise_clip'      : 0.5,
    'gradient_clipping'             : None,
    'debug'                         : False,

    'buffer_capacity'               : int(1e4),
    'train_metrics'                 : [
                                        (   
                                            tfa.metrics.tf_metrics.NumberOfEpisodes,
                                            {}  
                                        ),
                                        (
                                            tfa.metrics.tf_metrics.EnvironmentSteps,
                                            {}
                                        ),
                                        (
                                            tfa.metrics.tf_metrics.AverageEpisodeLengthMetric,
                                            {}
                                        ),
                                        (
                                            tfa.metrics.tf_metrics.AverageReturnMetric,
                                            {}
                                        ),
                                        (
                                            tfa.metrics.tf_metrics.MaxReturnMetric,
                                            {}
                                        ),
                                    ],
    'eval_metrics'                 : [ 
                                        (   
                                            tfa.metrics.tf_metrics.AverageEpisodeLengthMetric,
                                            {}  
                                        ),  
                                        (   
                                            tfa.metrics.tf_metrics.AverageReturnMetric,
                                            {}  
                                        ),  
                                        (   
                                            tfa.metrics.tf_metrics.NumberOfEpisodes,
                                            {}  
                                        ),  
                                        (   
                                            tfa.metrics.tf_metrics.EnvironmentSteps,
                                            {}  
                                        ),  
                                        (   
                                            tfa.metrics.tf_metrics.MaxReturnMetric,
                                            {}  
                                        ),  
                                    ],
    'initial_collect_episodes'      : 3,
    'collect_episodes_per_iteration': 5,
    'use_tf_functions'              : True,
    'num_parallel_calls'            : 2,
    'batch_size'                    : 128,
    'train_sequence_length'         : 10,
    'num_prefetch'                  : 3,

    'summaries_flush_secs'          : 10,
    
    'max_episode_size'              : 750,
    'obs_history_steps'             : 5,

    'num_eval_episodes'             : 5,
    'num_iterations'                : int(1e5),
    'train_steps_per_iteration'     : 200,
}
