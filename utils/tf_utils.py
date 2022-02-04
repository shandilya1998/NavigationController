import tf_agents as tfa
import tensorflow as tf
import numpy as np
import random
import os
from typing import Any, List, Optional, Tuple, Type
import functools
from simulations.tf_maze_env import MazeEnv
from simulations.point import PointEnv
from simulations.maze_task import CustomGoalReward4Rooms
import absl
import time
from models.preprocessing_layers import VisualCortex, ProprioreceptiveCortex, ActionPreprocessing

class MultiInputActorRnnNetwork(tfa.networks.lstm_encoding_network.LSTMEncodingNetwork):
    def __init__(
        self,
        input_tensor_spec,
        action_spec,
        preprocessing_layers=None,
        preprocessing_combiner=None,
        conv_layer_params=None,
        input_fc_layer_params=(75, 40),
        lstm_size=None,
        output_fc_layer_params=(75, 40),
        activation_fn=tf.keras.activations.relu,
        rnn_construction_fn=None,
        rnn_construction_kwargs=None,
        dtype=tf.float32,
        name='MultiInputActorRnnNetwork',
    ):
        flat_action_spec = tf.nest.flatten(action_spec)
        action_layers = [
            tf.keras.layers.Dense(
                single_action_spec.shape.num_elements(),
                activation=tf.keras.activations.tanh,
                kernel_initializer=tf.keras.initializers.RandomUniform(
                    minval=-0.003, maxval=0.003),
                name='action') for single_action_spec in flat_action_spec
        ]
        self._output_tensor_spec = action_spec

        num_actions = len(flat_action_spec)

        def repeat_input(inp):
            return {'action{}'.format(i) : inp for i in range(num_actions)}

        def ensure_correct_output(inp):
            return tf.nest.pack_sequence_as(self._output_tensor_spec, inp)

        action_gen = tfa.networks.sequential.Sequential([
            tfa.networks.NestFlatten(),
            tf.keras.layers.Lambda(repeat_input),
            tfa.networks.nest_map.NestMap(
                {'action{}'.format(i) : action_layers[i] for i in range(num_actions)}
            ),
            tfa.networks.nest_map.NestFlatten(),
            tf.keras.layers.Lambda(ensure_correct_output)
        ])

        super(MultiInputActorRnnNetwork, self).__init__(
            input_tensor_spec = input_tensor_spec,
            preprocessing_layers = preprocessing_layers,
            preprocessing_combiner = preprocessing_combiner,
            conv_layer_params = conv_layer_params,
            input_fc_layer_params = input_fc_layer_params,
            lstm_size = lstm_size,
            output_fc_layer_params = output_fc_layer_params,
            activation_fn = activation_fn,
            rnn_construction_fn = rnn_construction_fn,
            rnn_construction_kwargs = rnn_construction_kwargs,
            dtype = dtype,
            name = name
        )

        self._output_encoder.append(action_gen)

    """    
    def call(self, observation, step_type, network_state=(), training=False):
        num_outer_dims = tfa.utils.nest_utils.get_outer_rank(observation,
                                               self.input_tensor_spec)
        if num_outer_dims not in (1, 2):
            raise ValueError('Input observation must have a batch or batch x time outer shape.')

        has_time_dim = num_outer_dims == 2
        if not has_time_dim:
            # Add a time dimension to the inputs.
            observation = tf.nest.map_structure(lambda t: tf.expand_dims(t, 1), observation)
            step_type = tf.nest.map_structure(lambda t: tf.expand_dims(t, 1), step_type)

        state, _ = self._input_encoder(
            observation, step_type=step_type, network_state=(), training=training
        )

        network_kwargs = {}
        if isinstance(self._lstm_network, tfa.keras_layers.dynamic_unroll_layer.DynamicUnroll):
            network_kwargs['reset_mask'] = tf.equal(step_type,
                                                    tfa.trajectories.time_step.StepType.FIRST,
                                                    name='mask')

        # Unroll over the time sequence.
        output = self._lstm_network(
            inputs=state,
            initial_state=network_state,
            training=training,
            **network_kwargs)

        if isinstance(self._lstm_network, tfa.keras_layers.dynamic_unroll_layer.DynamicUnroll):
            state, network_state = output
        else:
            state = output[0]
            network_state = tf.nest.pack_sequence_as(
                self._lstm_network.cell.state_size, tf.nest.flatten(output[1:]))

        for layer in self._output_encoder:
            state = layer(state, training=training)
       
        print('state', state.shape)
        if not has_time_dim:
            # Remove time dimension from the state.
            state = tf.squeeze(state, [1])
        print('state', state.shape)

        actions = []
        for layer, spec in zip(self._action_layers, self._flat_action_spec):
            action = layer(state, training=training)
            action = tfa.utils.common.scale_to_spec(action, spec)
            #action = batch_squash.unflatten(action)  # [B x T, ...] -> [B, T, ...]
            actions.append(action)

        output_actions = tf.nest.pack_sequence_as(self._output_tensor_spec, actions)
        return output_actions, network_state
    """

class MultiInputCriticRnnNetwork(tfa.networks.lstm_encoding_network.LSTMEncodingNetwork):
    def __init__(
        self,
        input_tensor_spec,
        preprocessing_layers=None,
        preprocessing_combiner=None,
        conv_layer_params=None,
        input_fc_layer_params=(75, 40),
        lstm_size=None,
        output_fc_layer_params=(75, 40),
        activation_fn=tf.keras.activations.relu,
        rnn_construction_fn=None,
        rnn_construction_kwargs=None,
        dtype=tf.float32,
        name='MultiInputActorRnnNetwork',
    ):
        #self._flatten_inputs = tfa.networks.NestFlatten()
        q_layer = tf.keras.layers.Dense(
                1,
                activation = None,
                kernel_initializer = tf.keras.initializers.RandomUniform(
                    minval=-0.003, maxval=0.003),
                name='value') 
        # Need to provide a separate preprocessing_combiner that takes into account input structure 
        super(MultiInputCriticRnnNetwork, self).__init__(
            input_tensor_spec = input_tensor_spec,
            preprocessing_layers = preprocessing_layers,
            preprocessing_combiner = preprocessing_combiner,
            conv_layer_params = conv_layer_params,
            input_fc_layer_params = input_fc_layer_params,
            lstm_size = lstm_size,
            output_fc_layer_params = output_fc_layer_params,
            activation_fn = activation_fn,
            rnn_construction_fn = rnn_construction_fn,
            rnn_construction_kwargs = rnn_construction_kwargs,
            dtype = dtype,
            name = name
        )
        self._output_encoder.append(q_layer)

    """
    def call(self, inputs, step_type, network_state=(), training=False):
        # `inputs` must be `(observation, action)`
        #inputs = self._flatten_inputs(inputs)
        num_outer_dims = tfa.utils.nest_utils.get_outer_rank(inputs,
                                               self.input_tensor_spec)
        if num_outer_dims not in (1, 2):
            raise ValueError('Input must have a batch or batch x time outer shape.')

        has_time_dim = num_outer_dims == 2
        if not has_time_dim:
            # Add a time dimension to the inputs.
            inputs = tf.nest.map_structure(lambda t: tf.expand_dims(t, 1), inputs)
            step_type = tf.nest.map_structure(lambda t: tf.expand_dims(t, 1), step_type)

        state, _ = self._input_encoder(
            inputs, step_type=step_type, network_state=(), training=training)

        network_kwargs = {}
        if isinstance(self._lstm_network, tfa.keras_layers.dynamic_unroll_layer.DynamicUnroll):
            network_kwargs['reset_mask'] = tf.equal(step_type,
                                                    tfa.trajectories.time_step.StepType.FIRST,
                                                    name='mask')

        # Unroll over the time sequence.
        output = self._lstm_network(
            inputs=state,
            initial_state=network_state,
            training=training,
            **network_kwargs)

        if isinstance(self._lstm_network, tfa.keras_layers.dynamic_unroll_layer.DynamicUnroll):
            state, network_state = output
        else:
            state = output[0]
            network_state = tf.nest.pack_sequence_as(
                self._lstm_network.cell.state_size, tf.nest.flatten(output[1:]))

        for layer in self._output_encoder:
            state = layer(state, training=training)
        output = self._q_layer(state)
        q_value = tf.reshape(output, [-1])
        q_value = batch_squash.unflatten(q_value)  # [B x T, ...] -> [B, T, ...]
        if not has_time_dim:
            q_value = tf.squeeze(q_value, axis=1)

        return q_value, network_state
    """

def create_actor_network(
    params,
    env
):
    preprocessing_layers_actor = tuple([
        VisualCortex(),
        ProprioreceptiveCortex()
    ])
    preprocessing_combiner_actor = tf.keras.layers.Concatenate(axis=-1)
    actor = MultiInputActorRnnNetwork(
        input_tensor_spec = env.time_step_spec().observation,
        action_spec = env.action_spec(),
        preprocessing_layers = preprocessing_layers_actor,
        preprocessing_combiner = preprocessing_combiner_actor,
        conv_layer_params = None,
        input_fc_layer_params = params['input_fc_layer_params_actor'],
        lstm_size = params['lstm_size_actor'],
        output_fc_layer_params = params['output_fc_layer_params_actor'],
        activation_fn = params['activation_fn_actor'],
        rnn_construction_fn=None,
        rnn_construction_kwargs=None,
        dtype=tf.float32,
        name='MultiInputActorRnnNetwork',
    )
    return actor

def create_critic_network(
    params,
    env
):
    if isinstance(params['action_dim'], int):
        assert params['action_dim'] == env.action_spec().shape[-1]
    elif isinstance(params['action_dim'], list):
        for dim, spec in zip(params['action_dim'], env.action_spec()):
            # Support only for vector action currently
            assert dim == spec.shape[-1]
    preprocessing_layers_critic = tuple([
        tuple([
            VisualCortex(),
            ProprioreceptiveCortex()
        ]),
        ActionPreprocessing()
    ])
    preprocessing_layers_critic = tuple(preprocessing_layers_critic)
    preprocessing_combiner_critic = tf.keras.layers.Concatenate(axis=-1)
    input_tensor_spec = (
        env.time_step_spec().observation,
        env.action_spec()
    )
    critic = MultiInputCriticRnnNetwork(
        input_tensor_spec = input_tensor_spec,
        preprocessing_layers = preprocessing_layers_critic,
        preprocessing_combiner = preprocessing_combiner_critic,
        conv_layer_params = None,
        input_fc_layer_params = params['input_fc_layer_params_critic'],
        lstm_size = params['lstm_size_critic'],
        output_fc_layer_params = params['output_fc_layer_params_critic'],
        activation_fn = params['activation_fn_critic'],
        rnn_construction_fn=None,
        rnn_construction_kwargs=None,
        name='MultiInputCriticRnnNetwork',
    )
    return critic

def create_rtd3_agent(
    params,
    env,
):
    actor = create_actor_network(params, env)
    critic = create_critic_network(params, env)
    agent = tfa.agents.Td3Agent(
        time_step_spec = env.time_step_spec(),
        action_spec = env.action_spec(),
        actor_network = actor,
        critic_network = critic,
        actor_optimizer = params['optimizer_class_actor'](
            **params['optimizer_kwargs_actor']
        ),
        critic_optimizer = params['optimizer_class_critic'](
            **params['optimizer_kwargs_critic']
        ),
        exploration_noise_std = params['exploration_noise_std'],
        target_update_tau = params['target_update_tau'],
        target_update_period = params['target_update_period'],
        actor_update_period = params['actor_update_period'],
        gamma = params['gamma'],
        reward_scale_factor = params['reward_scale_factor'],
        target_policy_noise = params['target_policy_noise'],
        target_policy_noise_clip = params['target_policy_noise_clip'],
        debug_summaries = params['debug'],
        summarize_grads_and_vars = params['debug'],
        name = 'RTd3Agent'
    )
    if params['use_tf_functions']:
        agent.train = tfa.utils.common.function(agent.train)
    return agent

def create_replay_buffer(
    params,
    env,
    agent
):
    replay_buffer = tfa.replay_buffers.tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec = agent.collect_data_spec,
        batch_size = env.batch_size,
        max_length = params['buffer_capacity'],
    )
    return replay_buffer

def create_train_metrics(
    params
):
    metrics = [
        metric(**kwargs) for metric, kwargs in params['train_metrics']
    ]
    return metrics

def create_eval_metrics(
    params
):
    metrics = [
        metric(**kwargs) for metric, kwargs in params['eval_metrics']
    ]
    return metrics

def create_drivers(
    params,
    env,
    agent,
    replay_buffer
):
    train_metrics = create_train_metrics(params)
    initial_collect_driver = tfa.drivers.dynamic_episode_driver.DynamicEpisodeDriver(
        env = env,
        policy = agent.collect_policy,
        observers = [replay_buffer.add_batch] + train_metrics,
        num_episodes = params['initial_collect_episodes']
    )
    collect_driver = tfa.drivers.dynamic_episode_driver.DynamicEpisodeDriver(
        env = env,
        policy = agent.collect_policy,
        observers = [replay_buffer.add_batch] + train_metrics,
        num_episodes = params['collect_episodes_per_iteration']
    )
    if params['use_tf_functions']:
      initial_collect_driver.run = tfa.utils.common.function(initial_collect_driver.run)
      collect_driver.run = tfa.utils.common.function(collect_driver.run)
    return initial_collect_driver, collect_driver, train_metrics

def create_train_step(
    params,
    agent,
    replay_buffer
):
    dataset = replay_buffer.as_dataset(
        num_parallel_calls = params['num_parallel_calls'],
        sample_batch_size = params['batch_size'],
        num_steps = params['train_sequence_length'] + 1
    ).prefetch(params['num_prefetch'])
    iterator = iter(dataset)
    def train_step():
        experience, _ = next(iterator)
        return agent.train(experience)
    if params['use_tf_functions']:
        train_step = tfa.utils.common.function(train_step)
    return train_step

def train_rtd3(
    params,
    log_dir
):

    # Log Config
    train_dir = os.path.join(log_dir, 'train')
    eval_dir = os.path.join(log_dir, 'eval')
    train_summary_writer = tf.compat.v2.summary.create_file_writer(
      train_dir, flush_millis = params['summaries_flush_secs'] * 1000
    )
    train_summary_writer.set_as_default()

    eval_summary_writer = tf.compat.v2.summary.create_file_writer(
      eval_dir, flush_millis = params['summaries_flush_secs'] * 1000
    )
    eval_metrics = create_eval_metrics(params)

    # Variable to store surrent step
    global_step = tf.compat.v1.train.get_or_create_global_step()

    # Environment Creation
    env = tfa.environments.tf_py_environment.TFPyEnvironment(
        MazeEnv(
            model_cls = PointEnv,
            maze_task = CustomGoalReward4Rooms,
            max_episode_size = params['max_episode_size'],
            n_steps = params['obs_history_steps']
        )
    )

    """
    step = 0
    while step < 100:
        print('step {}'.format(step))
        ob = env.step(np.expand_dims(env._envs[0].get_action(), 0))
        step += 1
    """

    eval_env = tfa.environments.tf_py_environment.TFPyEnvironment(
        MazeEnv(
            model_cls = PointEnv,
            maze_task = CustomGoalReward4Rooms,
            max_episode_size = params['max_episode_size'],
            n_steps = params['obs_history_steps']
        )   
    )

    

    # Create and Initialize Agent
    agent = create_rtd3_agent(params, env)
    agent.initialize()

    # Create Replay Buffer and Data Collection Driver
    replay_buffer = create_replay_buffer(params, env, agent)
    collect_driver, initial_collect_driver, train_metrics = create_drivers(
        params, env, agent, replay_buffer
    )


    # Debug Code
    num_episodes = 4
    maximum_iterations = 100
    time_step = env.reset()
    policy_state = collect_driver.policy.get_initial_state(collect_driver.env.batch_size)
    batch_dims = tfa.utils.nest_utils.get_outer_shape(
        time_step,
        env.time_step_spec())
    counter = tf.zeros(batch_dims, tf.int32)
    #print('here')
    [_, time_step, policy_state] = tf.nest.map_structure(
        tf.stop_gradient,
        tf.while_loop(
            cond=collect_driver._loop_condition_fn(num_episodes),
            body=collect_driver._loop_body_fn(),
            loop_vars=[counter, time_step, policy_state],
            parallel_iterations=1,
            maximum_iterations=maximum_iterations,
            name='driver_loop'
        )
    )
    #print('here 2')

    # Populating Replay Buffer with data from random policy
    absl.logging.info(
        'Initializing replay buffer by collecting experience for %d episodes '
        'with a random policy.', params['initial_collect_episodes'])
    print('here 3')
    initial_collect_driver.run()
    
    # Evaluation Step
    print('here')
    results = tfa.eval.metric_utils.eager_compute(
        metrics = eval_metrics,
        environment = eval_env,
        policy = agent.policy,
        num_episodes = params['num_eval_episodes'],
        train_step = global_step,
        summary_writer = eval_summary_writer,
        summary_prefix = 'Metrics'
    )
    print('here 2')
    tfa.eval.metric_utils.log_metrics(eval_metrics)

    time_step = None
    policy_state = agent.collect_policy.get_initial_state(env.batch_size)
    timed_at_step = global_step.numpy()
    time_acc = 0
    
    train_step = create_train_step(params, agent, replay_buffer)

    # Training Loops
    for _ in range(params['num_iterations']):
        start_time = time.time()
        time_step, policy_state = collect_driver.run(
            time_step = time_step,
            policy_state = policy_state,
        )
        for _ in range(params['train_steps_per_iteration']):
            train_loss = train_step()

        time_acc += time.time() - start_time
        if global_step.numpy() % log_interval == 0:
            absl.logging.info(
                'step = %d, loss = %f',
                global_step.numpy(),
                train_loss.loss
            )
            steps_per_sec = (global_step.numpy() - timed_at_step) / time_acc
            absl.logging.info('%.3f steps/sec', steps_per_sec)
            tf.compat.v2.summary.scalar(
                name = 'global_steps_per_sec',
                data = steps_per_sec,
                step = global_step
            )
            timed_at_step = global_step.numpy()
            time_acc = 0

        for train_metric in train_metrics:
            train_metric.tf_summaries(
                train_step = global_step,
                step_metrics = train_metrics[:2]
            )

        if global_step.numpy() % eval_interval == 0:
            results = tfa.eval.metric_utils.eager_compute(
                metrics = eval_metrics,
                environment = eval_env,
                policy = agent.policy,
                num_episodes = params['num_eval_episodes'],
                train_step = global_step,
                summary_writer = eval_summary_writer,
                summary_prefix = 'Metrics'
            )   
            tfa.eval.metric_utils.log_metrics(eval_metrics)
    return True
