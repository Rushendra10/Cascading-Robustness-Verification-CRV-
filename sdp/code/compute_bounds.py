"""Code with dual formulation for certification problem."""
from __future__ import absolute_import
from __future__ import division
import tensorflow as tf

def compute_bounds(test_input, epsilon, input_minval, input_maxval, nn_params):
    input_minval = tf.convert_to_tensor(input_minval, dtype=tf.float32)
    input_maxval = tf.convert_to_tensor(input_maxval, dtype=tf.float32)
    epsilon = tf.convert_to_tensor(epsilon, dtype=tf.float32)
    test_input = tf.convert_to_tensor(test_input, dtype=tf.float32)
    lower = []
    upper = []

    lower.append(tf.maximum(test_input - epsilon, input_minval))
    upper.append(tf.minimum(test_input + epsilon, input_maxval))

    for i in range(0, nn_params.num_hidden_layers):
        current_lower = 0.5*(nn_params.forward_pass(lower[i] + upper[i], i) + nn_params.forward_pass(lower[i] - upper[i], i, is_abs=True)) + nn_params.biases[i]
        current_upper = 0.5*(nn_params.forward_pass(lower[i] + upper[i], i) + nn_params.forward_pass(upper[i] - lower[i], i, is_abs=True)) + nn_params.biases[i]
        lower.append(tf.nn.relu(current_lower))
        upper.append(tf.nn.relu(current_upper))
    return lower, upper