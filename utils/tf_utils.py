import tf_agents as tfa
import tensorflow as tf
import numpy as np
import random
import os
from typing import Any, List, Optional, Tuple, Type
import functools
from constants import tf_params as params


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
        flat_action_spec = tf.nest.flatten(output_tensor_spec)
        action_layers = [
            tf.keras.layers.Dense(
                single_action_spec.shape.num_elements(),
                activation=tf.keras.activations.tanh,
                kernel_initializer=tf.keras.initializers.RandomUniform(
                    minval=-0.003, maxval=0.003),
                name='action') for single_action_spec in flat_action_spec
        ]
        self._flat_action_spec = flat_action_spec
        self._action_layers = action_layers
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
            observation, step_type=step_type, network_state=(), training=training)

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
 
        actions = []
        for layer, spec in zip(self._action_layers, self._flat_action_spec):
            action = layer(state, training=training)
            action = common.scale_to_spec(action, spec)
            action = batch_squash.unflatten(action)  # [B x T, ...] -> [B, T, ...]
            if not has_time_dim:
                action = tf.squeeze(action, axis=1)
            actions.append(action)

        output_actions = tf.nest.pack_sequence_as(self._output_tensor_spec, actions)
        return output_actions, network_state

class MultiInputCriticRnnNetwork(tfa.networks.lstm_encoding_network.LSTMEncodingNetwork):
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
        q_layer = tf.keras.layers.Dense(
                1,
                activation = None,
                kernel_initializer = tf.keras.initializers.RandomUniform(
                    minval=-0.003, maxval=0.003)
                name='value') 
        self._q_layer = q_layer
        # Need to provide a separate preprocessing_combiner that takes into account input structure 
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

    def call(self, inputs, step_type, network_state=(), training=False):
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
