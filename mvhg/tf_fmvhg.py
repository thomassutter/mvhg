import os
import sys
import numpy as np
import logging

import tensorflow as tf
import matplotlib.pyplot as plt

from mvhg.tf_heaviside import heaviside

tf.compat.v1.disable_eager_execution()


def sample_gumbel(shape, eps=1e-20):
    U = tf.random.uniform(shape, minval=0, maxval=1)
    return -tf.math.log(-tf.math.log(U + eps) + eps)


def gumbel_softmax(logits, temp, hard=True, add_noise=True):
    gs_sample = logits
    if add_noise:
        gs_sample = gs_sample + sample_gumbel(tf.shape(logits))
    y = tf.nn.softmax(gs_sample / temp)
    y_hard = tf.cast(tf.equal(y, tf.reduce_max(y, 2, keepdims=True)), y.dtype)
    y_hard = tf.stop_gradient(y_hard - y) + y
    return y, y_hard


def get_logits(n, m, m2, log_w1, log_w2, hside_check, eps=1e-20):
    n_samples = tf.shape(log_w1)[0]
    x_all = tf.range(m + 1)
    x_all = tf.repeat(tf.expand_dims(x_all, 0), repeats=n_samples, axis=0)
    x_all = tf.expand_dims(x_all, 1)
    log_x1_fact = tf.math.lgamma(x_all + 1.0)
    log_m1_x1_fact = tf.math.lgamma(m - x_all + 1.0)
    log_x2_fact = tf.math.lgamma(tf.nn.relu(n - x_all) + 1.0)
    log_m2_x2_fact = tf.math.lgamma(m2 - tf.nn.relu(n - x_all) + 1.0)
    log_m_x = log_x1_fact + log_x2_fact + log_m1_x1_fact + log_m2_x2_fact
    x1_log_w1 = x_all * log_w1
    x2_log_w2 = (n - x_all) * log_w2
    log_p_shifted = x1_log_w1 + x2_log_w2 - log_m_x
    log_p_shifted_sup = log_p_shifted + tf.math.log(hside_check + eps)
    return log_p_shifted_sup


def calc_group_w_m(m_group, log_w_group):
    m_G = tf.math.reduce_sum(m_group)
    lse_arg = log_w_group + tf.math.log(m_group) - tf.math.log(m_G)
    log_w_G = tf.reduce_logsumexp(lse_arg, axis=2)
    return log_w_G, m_G


def get_hside_check(n, x_i, m_r):
    check_x_i = n - x_i
    check_x_rest = m_r - tf.nn.relu(n - x_i)
    hside_check_x_i = heaviside(check_x_i)
    hside_check_x_rest = heaviside(check_x_rest)
    hside_check = hside_check_x_i * hside_check_x_rest
    return hside_check


def pmf_noncentral_fmvhg(
    m_all, n, log_w_all, temperature, n_c, add_gumbel_noise=True, hard=True
):
    n = tf.expand_dims(n, 1)
    n_samples = tf.shape(log_w_all)[0]

    rnd_order = np.random.permutation(n_c)
    inv_order = np.zeros_like(rnd_order)
    inv_order[rnd_order] = np.arange(rnd_order.size)
    n_out = tf.zeros((n_samples, 1, 1), dtype=tf.float32)
    m_out = tf.zeros((1), dtype=tf.float32)
    log_w_all = tf.expand_dims(log_w_all, 1)
    log_w_all_perm = tf.gather(log_w_all, rnd_order, axis=2)
    m_all_perm = [m_all[i] for i in rnd_order]
    y_all = []
    y_mask_all = []
    x_all = []
    x_all_soft = []
    for i in range(0, n_c):
        n_new = n - n_out
        m_i = m_all_perm[i]
        log_w_i = tf.expand_dims(log_w_all_perm[:, :, i], 1)
        x_i = tf.range(m_i + 1)
        x_i = tf.repeat(
            tf.expand_dims(tf.expand_dims(x_i, 0), 1), repeats=n_samples, axis=0
        )
        if i + 1 < n_c:
            m_rest = tf.math.reduce_sum(m_all_perm) - m_out - m_i
        else:
            m_rest = tf.zeros((1), dtype=tf.float32)
        N_new = m_i + m_rest
        m_rest_ind = m_all_perm[i + 1 :]
        if i + 1 < n_c:
            log_w_rest_ind = log_w_all_perm[:, :, i + 1 :]
            log_w_rest, m_rest = calc_group_w_m(m_rest_ind, log_w_rest_ind)
            log_w_rest = tf.expand_dims(log_w_rest, 2)
        else:
            w = tf.zeros((n_samples, 1, 1), dtype=tf.float32)

        x_rest = tf.nn.relu(n_new - x_i)
        hside_check = get_hside_check(n_new, x_i, m_rest)
        if i + 1 < n_c:
            logits_p_x_i = get_logits(
                n_new, m_i, m_rest, log_w_i, log_w_rest, hside_check
            )
            p_x_i = tf.nn.softmax(logits_p_x_i)
            _, y_i = gumbel_softmax(
                logits_p_x_i, temperature, hard=hard, add_noise=add_gumbel_noise
            )
            y_i_soft, _ = gumbel_softmax(
                logits_p_x_i, temperature, hard=hard, add_noise=False
            )
        else:
            y_i = tf.dtypes.cast(hside_check, tf.float32)
        y_all.append(y_i)
        ones = tf.ones(
            (
                tf.dtypes.cast(n_samples, tf.float32),
                tf.dtypes.cast(m_i + 1, tf.int32),
                tf.dtypes.cast(m_i + 1, tf.int32),
            )
        )
        lt = tf.linalg.LinearOperatorLowerTriangular(ones).to_dense()
        mask_filled = tf.linalg.matmul(y_i, lt)
        y_mask_all.append(mask_filled)
        x_i_all = tf.range(m_i + 1)
        x_i_soft = tf.math.reduce_sum(y_i_soft * x_i_all, axis=-1, keepdims=True)
        x_i = tf.math.reduce_sum(y_i * x_i_all, axis=-1, keepdims=True)
        x_all_soft.append(x_i_soft)
        x_all.append(x_i)
        n_out += x_i
        m_out += m_i
    log_w_all_perm = tf.squeeze(log_w_all_perm, 1)
    n = tf.squeeze(n, 1)
    log_p = get_probability(x_all, y_all, m_all_perm, n, log_w_all_perm, n_c)
    y_all = [y_all[i] for i in inv_order]
    x_all = [x_all[i] for i in inv_order]
    y_mask_all = [y_mask_all[i] for i in inv_order]
    return y_all, x_all, x_all_soft, y_mask_all, log_p


def get_log_p_x_i(logits_p_x_i, y):
    log_p_x_i = logits_p_x_i - tf.reduce_logsumexp(logits_p_x_i, axis=-1, keepdims=True)
    y = tf.squeeze(y, 1)
    y = tf.expand_dims(y, 2)
    log_p_x_sel = log_p_x_i @ y
    log_p_x_sel = tf.squeeze(log_p_x_sel, 1)
    return log_p_x_sel


def get_probability(X_all, y_all, m_all, n, log_w_all, n_c):
    n = tf.expand_dims(n, 1)
    n_samples = tf.shape(log_w_all)[0]
    n_out = tf.zeros((n_samples, 1, 1), dtype=tf.float32)
    m_out = tf.zeros((1), dtype=tf.float32)
    log_w_all = tf.expand_dims(log_w_all, 1)
    log_p_x_all = []
    for i in range(0, n_c - 1):
        x_sel = tf.cast(tf.squeeze(X_all[i]), tf.int32)
        x_sel = tf.expand_dims(x_sel, 1)
        ind_sel = tf.expand_dims(tf.range(n_samples), 1)
        x_sel = tf.concat([ind_sel, x_sel], axis=1)

        n_new = n - n_out
        m_i = m_all[i]
        log_w_i = tf.expand_dims(log_w_all[:, :, i], 1)
        x_i = tf.range(m_i + 1)
        x_i = tf.repeat(
            tf.expand_dims(tf.expand_dims(x_i, 0), 1), repeats=n_samples, axis=0
        )
        if i + 1 < n_c:
            m_rest = tf.math.reduce_sum(m_all) - m_out - m_i
        else:
            m_rest = tf.zeros((1), dtype=tf.float32)

        N_new = m_i + m_rest
        m_rest_ind = m_all[i + 1 :]
        if i + 1 < n_c:
            log_w_rest_ind = log_w_all[:, :, i + 1 :]
            log_w_rest, m_rest = calc_group_w_m(m_rest_ind, log_w_rest_ind)
        else:
            w = tf.zeros((n_samples, 1, 1), dtype=tf.float32)

        x_rest = tf.nn.relu(n_new - x_i)
        hside_check = get_hside_check(n_new, x_i, m_rest)
        log_w_rest = tf.expand_dims(log_w_rest, 1)
        logits_p_x_i = get_logits(n_new, m_i, m_rest, log_w_i, log_w_rest, hside_check)
        y_i_sel = y_all[i]
        log_p_x_sel = get_log_p_x_i(logits_p_x_i, y_i_sel)
        log_p_x_all.append(log_p_x_sel)
        n_out += X_all[i]
        m_out += m_i
    log_p_x_all = tf.concat(log_p_x_all, axis=-1)
    log_p = tf.math.reduce_sum(log_p_x_all, axis=-1)
    return log_p
