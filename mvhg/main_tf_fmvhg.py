import os
import sys
import numpy as np
import logging

import tensorflow as tf
import matplotlib.pyplot as plt

import mvhg.tf_fmvhg as tf_fmvhg

tf.compat.v1.disable_eager_execution()


if __name__ == "__main__":
    dir_results = os.path.join(".", "results")
    log_level = logging.INFO
    logging.basicConfig(level=log_level)
    class_names = ["violet", "yellow", "green"]
    colors = ["blueviolet", "yellow", "green"]
    num_classes = 3
    num_samples = 1
    create_plot = True
    m = tf.Variable(tf.constant([10, 10, 10], dtype=tf.float32))
    n = tf.Variable(tf.constant([10], dtype=tf.float32))
    w = tf.math.log(tf.Variable(tf.constant([1.0, 1.0, 1.0], dtype=tf.float32)))
    tau = tf.Variable(tf.constant([1.0]))
    w = tf.repeat(tf.expand_dims(w, 0), repeats=num_samples, axis=0)
    n_repeats = 1

    for h in range(n_repeats):
        init = tf.compat.v1.global_variables_initializer()
        sess = tf.compat.v1.Session()
        with tf.compat.v1.Session() as sess:
            sess.run(init)
            # y, x, _, log_p = mvhg(w)
            y, x, y_mask = tf_fmvhg.pmf_noncentral_fmvhg(
                m, n, w, tau, num_classes, add_gumbel_noise=True, hard=True
            )
            p, log_p = tf_fmvhg.get_probability(x, y, m, n, w, num_classes)
            log_p = print_t(log_p)
