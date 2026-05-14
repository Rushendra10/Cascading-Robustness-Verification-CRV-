"""Code for running the certification problem."""

from __future__ import absolute_import, division, print_function

import json
import os
import time

import numpy as np
import scipy.io as sio
import tensorflow as tf

import compute_bounds
import matlab_interface
import neural_net_params
import read_weights

flags = tf.compat.v1.flags
FLAGS = flags.FLAGS
flags.DEFINE_string(
    "checkpoint", None, "Path of checkpoint with trained model to verify"
)
flags.DEFINE_string("model_json", None, "Path of json file with model description")
# change
# flags.DEFINE_string('test_input', None, 'Path of numpy file with test input to certify')
# change
# flags.DEFINE_integer('true_class', 8, 'True class of the test input')
flags.DEFINE_float("input_minval", 0, "Minimum value of valid input")
flags.DEFINE_float("input_maxval", 1, "Maximum value of valid input")
flags.DEFINE_float("epsilon", 0.1, "Size of perturbation")
flags.DEFINE_integer(
    "adv_class", -1, "target class of adversarial example; test all classes if -1"
)
flags.DEFINE_integer("num_classes", 10, "total number of classes to verify against")
# Working folder to save the .m files that the matlab function reads
flags.DEFINE_string("matlab_folder", None, "Folder to save matlab things")
flags.DEFINE_integer("input_dimension", 784, "Folder to save matlab things")

# changed:ask
if os.path.exists(os.path.join(FLAGS.matlab_folder, "SDP_optimum.mat")):
    os.remove(os.path.join(FLAGS.matlab_folder, "SDP_optimum.mat"))
    print("removed sdp")


filepath = os.path.join("logs", "output.txt")
if not os.path.exists(filepath):
    print("File not found. Creating new file...")
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with open(filepath, "w") as f:
        f.write("")
        print("File created")


def write_to_text_file(text):
    with open(filepath, "a") as f:
        f.write(str(text))
        f.write("\n")


def main(_):
    
    sdp_inputs_folderpath = '<REPLACE WITH THE ABSOLUTE PATH TO /sdp_inputs>'

    true_labels = os.listdir(sdp_inputs_folderpath)
    for i in true_labels:
        true_class = i
        test_files = os.listdir(os.path.join(sdp_inputs_folderpath, i))
        for j in test_files:
            test_input_path = os.path.join(sdp_inputs_folderpath, i, j)
            test_input = test_input_path
            # # change
            test_input = np.load(test_input)

            num_rows = 28
            num_columns = 28
            num_channels = 1

            # change
            print("Running certification for input file", test_input_path)
            net_weights, net_biases, net_layer_types = read_weights.read_weights(
                FLAGS.checkpoint,
                FLAGS.model_json,
                [num_rows, num_columns, num_channels],
            )

            nn_params = neural_net_params.NeuralNetParams(
                net_weights, net_biases, net_layer_types
            )

            test_input = np.reshape(test_input, [-1, 1])
            start_class = 0
            end_class = FLAGS.num_classes
            for adv_class in range(start_class, end_class):
                write_to_text_file(str("true_class:" + true_class))
                write_to_text_file(str("test_input_path:" + test_input_path))
                write_to_text_file(str("adv_class:" + str(adv_class)))
                if os.path.exists(os.path.join(FLAGS.matlab_folder, "SDP_optimum.mat")):
                    os.remove(os.path.join(FLAGS.matlab_folder, "SDP_optimum.mat"))
                    print("removing optimum file in iteration" + str(adv_class))
                if adv_class == int(true_class):
                    write_to_text_file("-1")
                    write_to_text_file("-1")
                    write_to_text_file("--------------------")
                    continue

                config = tf.ConfigProto(device_count={"GPU": 0})

                with tf.Session(config=config) as sess:
                    sess.run(tf.global_variables_initializer())
                    if not os.path.exists(FLAGS.matlab_folder):
                        os.mkdir(FLAGS.matlab_folder)
                    matlab_object = matlab_interface.MatlabInterface(
                        FLAGS.matlab_folder
                    )

                    matlab_object.save_weights(nn_params, sess)
                    opt_params = {}
                    opt_params["test_input"] = test_input
                    opt_params["epsilon"] = FLAGS.epsilon
                    # change
                    opt_params["true_class"] = int(true_class)
                    opt_params["adv_class"] = adv_class
                    # change
                    opt_params["final_linear"] = sess.run(
                        nn_params.final_weights[adv_class, :]
                        - nn_params.final_weights[int(true_class), :]
                    )
                    opt_params["final_constant"] = sess.run(
                        nn_params.final_bias[adv_class]
                        - nn_params.final_bias[int(true_class)]
                    )
                    lower, upper = compute_bounds.compute_bounds(
                        test_input,
                        FLAGS.epsilon,
                        FLAGS.input_minval,
                        FLAGS.input_maxval,
                        nn_params,
                    )
                    opt_params["lower"] = [sess.run(l) for l in lower]
                    opt_params["upper"] = [sess.run(u) for u in upper]
                    matlab_object.save_opt_params(opt_params)
                    # run_string = f"matlab -nosplash -nodesktop -r \"matlab_sdp('{FLAGS.matlab_folder}');\""
                    run_string = f"matlab -nosplash -nodesktop -r \"matlab_sdp('{FLAGS.matlab_folder}'); exit;\""

                    
                    print(run_string)
                    os.system(run_string)

                    while True:
                        if os.path.exists(
                            os.path.join(FLAGS.matlab_folder, "SDP_optimum.mat")
                        ):
                            time.sleep(1)
                            break

                    opt_val = sio.loadmat(
                        os.path.join(FLAGS.matlab_folder, "SDP_optimum.mat")
                    )
                    opt_val = opt_val["val"]
                    if opt_val < 0:
                        print(
                            "Input example is robust to perturbation to adv class "
                            + str(adv_class)
                        )
                    else:
                        print(
                            "Input example cannot be certified as robust to perturbation to adv class "
                            + str(adv_class)
                        )
                        # exit()
                print("Input example succesfully verified")
                write_to_text_file("--------------------")


if __name__ == "__main__":
    tf.app.run(main)
