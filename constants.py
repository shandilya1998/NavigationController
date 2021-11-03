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
    'dt'                          : 0.005,
    'learning_starts'             : 100,
    'batch_size'                  : 2,
    'render_freq'                 : 20000,
    'save_freq'                   : 10000,
    'eval_freq'                   : 10000,
    'buffer_size'                 : 100000,
    'total_timesteps'             : int(1e6),
    'ds'                          : 0.1,
    'motor_cortex'                : [[64, 256, 20], [128, 128, 2]],
}
