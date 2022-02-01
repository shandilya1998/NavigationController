from utils.tf_utils import train_rtd3
import tensorflow as tf
import tf_agents as tfa
import os
from constants import tf_params as params
import absl

tf.compat.v1.enable_v2_behavior()
absl.logging.set_verbosity(absl.logging.INFO)
log_dir = 'assets/out/models/exp23'
done = train_rtd3(params, log_dir)
if done:
    absl.logging.info('Done')
