import os
import sys
import numpy as np
import random

import torch
import torch.nn.functional as F

from mvhg.pt_heaviside import sigmoid


class MVHG(torch.nn.Module):
    def __init__(self, device="cuda", eps=1e-20):
        super().__init__()
        self.device = device
        self.eps = eps
        self.relu = torch.nn.ReLU()
        self.sm = torch.nn.Softmax(dim=-1)
        self.hside_approx = sigmoid(slope=1.0)
        self.p_hside = 100

    def sample_gumbel(self, shape):
        U = torch.rand(shape).to(self.device)
        return -torch.log(-torch.log(U + self.eps) + self.eps)

    def gumbel_softmax_sample(self, logits, temperature, gumbel_noise):
        gs = self.sample_gumbel(logits.size())
        if gumbel_noise:
            logits = logits + gs
        return F.softmax(logits / temperature, dim=-1)

    def gumbel_softmax(self, logits, temperature, gumbel_noise, hard):
        """
        input: [*, n_class]
        return: [*, n_class] an one-hot vector
        """
        y = self.gumbel_softmax_sample(logits, temperature, gumbel_noise)
        if hard:
            shape = y.size()
            _, ind = y.max(dim=-1)
            y_hard = torch.zeros_like(y).view(-1, shape[-1])
            y_hard.scatter_(1, ind.view(-1, 1), 1)
            y_hard = y_hard.view(*shape)
            return (y_hard - y).detach() + y
        else:
            return y
        return (y_hard - y).detach() + y

    def get_logits_fisher(self, n, m1, m2, log_w1, log_w2, hside_check):
        n_samples = log_w1.shape[0]
        x_all = torch.arange(m1 + 1, dtype=torch.float32).to(self.device)
        x_all = x_all.unsqueeze(0).repeat(n_samples, 1, 1)
        log_x1_fact = torch.lgamma(x_all + 1.0)
        log_m1_x1_fact = torch.lgamma(m1 - x_all + 1.0)
        log_x2_fact = torch.lgamma(self.relu(n - x_all) + 1.0)
        log_m2_x2_fact = torch.lgamma(self.relu(m2 - self.relu(n - x_all)) + 1.0)
        log_m_x = log_x1_fact + log_x2_fact + log_m1_x1_fact + log_m2_x2_fact
        x1_log_w1 = x_all * log_w1
        x2_log_w2 = (n - x_all) * log_w2
        log_p_shifted = x1_log_w1 + x2_log_w2 - log_m_x
        log_p_shifted_sup2 = log_p_shifted + self.p_hside * torch.log(
            hside_check + self.eps
        )
        return log_p_shifted_sup2

    def calc_group_w_m(self, m_group, log_w_group):
        """
        Calculates the merged w and m for a group G of classes

        returns: log(w_G), m_G
        """
        m_group = m_group.float()
        m_G = torch.sum(m_group)
        lse_arg = log_w_group + m_group.log() - m_G.log()
        log_w_G = torch.logsumexp(lse_arg, dim=2)
        return log_w_G, m_G

    def get_w_m(self, idx, m_all, log_w_all):
        n_c = m_all.shape[0]
        n_samples = log_w_all.shape[0]
        m_i = m_all[idx]
        log_w_i = log_w_all[:, :, idx].unsqueeze(1)
        m_rest_ind = m_all[idx + 1 :]
        if idx + 1 < n_c:
            log_w_rest_ind = log_w_all[:, :, idx + 1 :]
            log_w_rest, m_rest = self.calc_group_w_m(m_rest_ind, log_w_rest_ind)
        else:
            log_w_rest = torch.zeros((n_samples, 1, 1)).to(self.device)
            m_rest = torch.zeros((n_samples, 1, 1)).to(self.device)
        return [log_w_i, log_w_rest.unsqueeze(-1)], [m_i, m_rest]

    def get_log_p_x_from_logits(self, logits):
        log_p = logits - torch.logsumexp(logits, dim=2, keepdims=True)
        return log_p

    def get_y_mask(self, y):
        n_samples = y.shape[0]
        n_dim = y.shape[2]
        ones = torch.ones((n_samples, n_dim, n_dim), device=self.device)
        lt = torch.tril(ones)
        y_mask = torch.matmul(y, lt)
        return y_mask

    def get_x(self, y_mask):
        x = torch.sum(y_mask, dim=2).unsqueeze(2) - 1.0
        return x

    def get_hside_check(self, n, x_i, m_r):
        check_x_i = n - x_i
        check_x_rest = m_r - self.relu(n - x_i)
        hside_check_x_i = self.hside_approx(check_x_i)
        hside_check_x_rest = self.hside_approx(check_x_rest)
        hside_check = hside_check_x_i * hside_check_x_rest
        return hside_check

    def get_log_p_x_i(self, logits_p_x, y):
        n_samples = logits_p_x.shape[0]
        log_p_x = self.get_log_p_x_from_logits(logits_p_x)
        y = y.transpose(1, 2)
        log_p_x_i = log_p_x @ y
        log_p_x_i = log_p_x_i.squeeze(1)
        return log_p_x_i

    def forward(self, m_all, n, log_w_all, temperature, add_noise=True, hard=True):
        n_c = m_all.shape[0]
        n = n.unsqueeze(1)
        n_samples = log_w_all.shape[0]

        rnd_order = np.random.permutation(n_c)
        inv_order = np.zeros_like(rnd_order)
        inv_order[rnd_order] = np.arange(rnd_order.size)
        n_out = torch.zeros((n_samples, 1, 1), device=self.device)
        log_w_all = log_w_all.unsqueeze(1)
        log_w_all_perm = log_w_all[:, :, rnd_order]
        m_all_perm = m_all[rnd_order]
        y_all = []
        y_mask_all = []
        x_all = []
        log_p_X_all = []
        for i in range(0, n_c):
            n_new = self.relu(n - n_out)
            log_ws, ms = self.get_w_m(i, m_all_perm, log_w_all_perm)
            log_w_i, log_w_rest = log_ws
            m_i, m_rest = ms
            x_i = torch.arange(m_i + 1).to(self.device)
            x_i = x_i.unsqueeze(0).unsqueeze(1).repeat(n_samples, 1, 1)
            hside_check = self.get_hside_check(n_new, x_i, m_rest)
            if i + 1 < n_c:
                logits_p_x_i = self.get_logits_fisher(
                    n_new,
                    m_i,
                    m_rest,
                    log_w_i,
                    log_w_rest,
                    hside_check,
                )
                y_i = self.gumbel_softmax(logits_p_x_i, temperature, add_noise, hard)
            else:
                y_i = hside_check
            y_mask_filled = self.get_y_mask(y_i)
            x_i = self.get_x(y_mask_filled)
            # calculate log(p(x_i))
            if i + 1 < n_c:
                log_p_x_i = self.get_log_p_x_i(logits_p_x_i, y_i)
                log_p_X_all.append(log_p_x_i)
            n_out += x_i
            y_all.append(y_i)
            x_all.append(x_i)
            y_mask_all.append(y_mask_filled)
        log_p_X = torch.sum(torch.cat(log_p_X_all, dim=1), dim=1)
        y_all = [y_all[i] for i in inv_order]
        x_all = [x_all[i] for i in inv_order]
        y_mask_all = [y_mask_all[i] for i in inv_order]
        return y_all, x_all, y_mask_all, log_p_X

    def get_log_probs(self, m_all, n, x_all, y_all, log_w_all):
        """
        calculate the log probability of a random sample

        """
        n_c = m_all.shape[0]
        n = n.unsqueeze(1)
        n_samples = log_w_all.shape[0]
        n_out = torch.zeros((n_samples, 1, 1), device=self.device)
        log_w_all = log_w_all.unsqueeze(1)
        log_p_X_all = []
        for i in range(0, n_c):
            n_new = self.relu(n - n_out)
            log_ws, ms = self.get_w_m(i, m_all, log_w_all)
            log_w_i, log_w_rest = log_ws
            m_i, m_rest = ms
            x_i_all = torch.arange(m_i + 1).to(self.device)
            x_i_all = x_i_all.unsqueeze(0).unsqueeze(1).repeat(n_samples, 1, 1)
            hside_check = self.get_hside_check(n_new, x_i_all, m_rest)
            if i + 1 < n_c:
                logits_p_x_i = self.get_logits_fisher(
                    n_new,
                    m_i,
                    m_rest,
                    log_w_i,
                    log_w_rest,
                    hside_check,
                )
                x_i_sel = x_all[i]
                y_i_sel = y_all[i]
                log_p_x_i = self.get_log_p_x_i(logits_p_x_i, y_i_sel)
                log_p_X_all.append(log_p_x_i)
            n_out += x_i_sel
        log_p_X = torch.sum(torch.cat(log_p_X_all, dim=1), dim=1)
        return log_p_X
