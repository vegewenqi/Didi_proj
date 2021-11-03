#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2020/11/30
# @Author  : Yang Guan (Tsinghua Univ.)
# @FileName: load_policy.py
# =====================================
import argparse
import json

import tensorflow as tf
import numpy as np

from env_build.endtoend import CrossroadEnd2endMix
from env_build.utils.policy import AttentionPolicy4Toyota
from env_build.utils.preprocessor import Preprocessor


class LoadPolicy(object):
    def __init__(self, exp_dir, iter):
        model_dir = exp_dir + '/models'
        parser = argparse.ArgumentParser()
        params = json.loads(open(exp_dir + '/config.json').read())
        for key, val in params.items():
            parser.add_argument("-" + key, default=val)
        self.args = parser.parse_args()
        env = CrossroadEnd2endMix(future_point_num=self.args.num_rollout_list_for_policy_update[0])
        self.policy = AttentionPolicy4Toyota(self.args)
        self.policy.load_weights(model_dir, iter)
        self.preprocessor = Preprocessor(self.args.obs_scale, self.args.reward_scale, self.args.reward_shift,
                                         args=self.args, gamma=self.args.gamma)
        init_obs, all_info = env.reset()
        mask = all_info['mask']
        self.run_batch(init_obs[np.newaxis, :], mask[np.newaxis, :])
        self.obj_value_batch(init_obs[np.newaxis, :], mask[np.newaxis, :])

    @tf.function
    def run_batch(self, obses, masks):
        processed_obses = self.preprocessor.process_obs(obses)
        states = self._get_states(processed_obses, masks)
        actions = self.policy.compute_mode(states)
        return actions

    @tf.function
    def obj_value_batch(self, obses, masks):
        processed_obses = self.preprocessor.process_obs(obses)
        states = self._get_states(processed_obses, masks)
        values = self.policy.compute_obj_v(states)
        return values

    def _get_states(self, mb_obs, mb_mask):
        mb_obs_others, mb_attn_weights = self.policy.compute_attn(mb_obs[:, self.args.other_start_dim:], mb_mask)
        mb_state = tf.concat((mb_obs[:, :self.args.other_start_dim], mb_obs_others), axis=1)
        return mb_state

