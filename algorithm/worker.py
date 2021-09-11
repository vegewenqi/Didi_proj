#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2020/9/1
# @Author  : Yang Guan (Tsinghua Univ.)
# @FileName: worker.py
# =====================================

import logging

from env_build.endtoend import CrossroadEnd2endMixPiFix
import numpy as np
import tensorflow as tf

from preprocessor import Preprocessor
from utils.misc import judge_is_nan, args2envkwargs

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# logger.setLevel(logging.INFO)


class OffPolicyWorker(object):
    tf.config.experimental.set_visible_devices([], 'GPU')
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)
    """just for sample"""

    def __init__(self, policy_cls, env_id, args, worker_id):
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        self.worker_id = worker_id
        self.args = args
        self.env = CrossroadEnd2endMixPiFix(**args2envkwargs(args))
        self.policy_with_value = policy_cls(self.args)
        self.batch_size = self.args.batch_size
        self.obs = self.env.reset()
        self.done = False
        self.preprocessor = Preprocessor((self.args.obs_dim, ), self.args.obs_preprocess_type, self.args.reward_preprocess_type,
                                          self.args.reward_scale, self.args.reward_shift, args=self.args, gamma=self.args.gamma)

        self.explore_sigma = self.args.explore_sigma
        self.iteration = 0
        self.num_sample = 0
        self.sample_times = 0
        self.stats = {}
        logger.info('Worker initialized')

    def get_stats(self):
        self.stats.update(dict(worker_id=self.worker_id,
                               num_sample=self.num_sample,
                               # ppc_params=self.get_ppc_params()
                               )
                          )
        return self.stats

    def save_weights(self, save_dir, iteration):
        self.policy_with_value.save_weights(save_dir, iteration)

    def load_weights(self, load_dir, iteration):
        self.policy_with_value.load_weights(load_dir, iteration)

    def get_weights(self):
        return self.policy_with_value.get_weights()

    def set_weights(self, weights):
        return self.policy_with_value.set_weights(weights)

    def apply_gradients(self, iteration, grads):
        self.iteration = iteration
        self.policy_with_value.apply_gradients(tf.constant(iteration, dtype=tf.int32), grads)

    def get_ppc_params(self):
        return self.preprocessor.get_params()

    def set_ppc_params(self, params):
        self.preprocessor.set_params(params)

    def save_ppc_params(self, save_dir):
        self.preprocessor.save_params(save_dir)

    def load_ppc_params(self, load_dir):
        self.preprocessor.load_params(load_dir)

    def sample(self):
        batch_data = []
        sample_count = 0
        for _ in range(self.batch_size):
            # extract infos for each kind of participants
            start = 0; end = self.args.state_ego_dim + self.args.state_track_dim
            obs_ego = self.obs[start:end]
            start = end; end = start + self.args.state_bike_dim
            obs_bike = self.obs[start:end]
            start = end; end = start + self.args.state_person_dim
            obs_person = self.obs[start:end]
            start = end; end = start + self.args.state_veh_dim
            obs_veh = self.obs[start:end]

            obs_bike = np.reshape(obs_bike, [-1, self.args.per_bike_dim])
            obs_person = np.reshape(obs_person, [-1, self.args.per_person_dim])
            obs_veh = np.reshape(obs_veh, [-1, self.args.per_veh_dim])
            processed_obs_ego, processed_obs_bike, processed_obs_person, processed_obs_veh \
                = self.preprocessor.process_obs_PI(obs_ego, obs_bike, obs_person, obs_veh)
            processed_obs_other = np.concatenate([processed_obs_bike, processed_obs_person, processed_obs_veh])
            PI_obs_other = tf.reduce_sum(self.policy_with_value.compute_PI(processed_obs_other), axis=0)
            processed_obs = np.concatenate((processed_obs_ego, PI_obs_other.numpy()), axis=0)
            judge_is_nan([processed_obs])
            action, logp = self.policy_with_value.compute_action(processed_obs[np.newaxis, :])
            if self.explore_sigma is not None:
                action += np.random.normal(0, self.explore_sigma, np.shape(action))
            try:
                judge_is_nan([action])
            except ValueError:
                print('processed_obs', processed_obs)
                print('preprocessor_params', self.preprocessor.get_params())
                print('policy_weights', self.policy_with_value.policy.trainable_weights)
                action, logp = self.policy_with_value.compute_action(processed_obs[np.newaxis, :])
                judge_is_nan([action])
                raise ValueError
            obs_tp1, reward, self.done, info = self.env.step(action.numpy()[0])
            sample_count += 1
            self.done = 1 if sample_count > self.args.max_step else self.done
            start = 0; end = self.args.state_ego_dim + self.args.state_track_dim
            obs_ego_next = obs_tp1[start:end]
            start = end; end = start + self.args.state_bike_dim
            obs_bike = obs_tp1[start:end]
            start = end; end = start + self.args.state_person_dim
            obs_person = obs_tp1[start:end]
            start = end; end = start + self.args.state_veh_dim
            obs_veh = obs_tp1[start:end]

            obs_bike_next = np.reshape(obs_bike, [-1, self.args.per_bike_dim])
            obs_person_next = np.reshape(obs_person, [-1, self.args.per_person_dim])
            obs_veh_next = np.reshape(obs_veh, [-1, self.args.per_veh_dim])

            batch_data.append((obs_ego_next, obs_bike_next, obs_person_next, obs_veh_next, self.done, info['ref_index']))
            if self.done:
                self.obs = self.env.reset()
                sample_count = 0
            else:
                self.obs = obs_tp1.copy()

        if self.worker_id == 1 and self.sample_times % self.args.worker_log_interval == 0:
            logger.info('Worker_info: {}'.format(self.get_stats()))

        self.num_sample += len(batch_data)
        self.sample_times += 1
        return batch_data

    def sample_with_count(self):
        batch_data = self.sample()
        return batch_data, len(batch_data)
