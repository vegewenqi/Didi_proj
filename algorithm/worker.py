#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2020/9/1
# @Author  : Yang Guan (Tsinghua Univ.)
# @FileName: worker.py
# =====================================

import logging

from env_build.endtoend import CrossroadEnd2endMix
from env_build.endtoend_env_utils import *
import numpy as np
import tensorflow as tf
import traci

from algorithm.preprocessor import Preprocessor
from algorithm.utils.misc import judge_is_nan, args2envkwargs

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# logger.setLevel(logging.INFO)


class OffPolicyWorkerWithAttention(object):
    tf.config.experimental.set_visible_devices([], 'GPU')
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)
    """just for sample"""

    def __init__(self, policy_cls, env_id, args, worker_id):
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        self.worker_id = worker_id
        self.args = args
        self.env = CrossroadEnd2endMix(**args2envkwargs(args))
        self.policy_with_value = policy_cls(self.args)
        self.batch_size = self.args.batch_size
        self.obs, self.info = self.env.reset()
        self.done = False
        self.preprocessor = Preprocessor(self.args.obs_scale, self.args.reward_scale, self.args.reward_shift,
                                         args=self.args, gamma=self.args.gamma)

        self.explore_sigma = self.args.explore_sigma
        self.iteration = 0
        self.num_sample = 0
        self.sample_times = 0
        self.stats = {}
        logger.info('Worker initialized')
    
    def get_stats(self):
        self.stats.update(dict(worker_id=self.worker_id,
                               num_sample=self.num_sample,
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

    def _get_state(self, obs, mask):
        obs_other, _ = self.policy_with_value.compute_attn(obs[self.args.other_start_dim:][np.newaxis, :],
                                                           mask[np.newaxis, :])
        obs_other = obs_other.numpy()[0]
        state = np.concatenate((obs[:self.args.other_start_dim], obs_other), axis=0)
        return state

    def sample(self):
        batch_data = []
        sample_count = 0
        for _ in range(self.batch_size):
            processed_obs = self.preprocessor.process_obs(self.obs)
            mask = self.info['mask']
            state = self._get_state(processed_obs, mask)

            action, logp = self.policy_with_value.compute_action(state[np.newaxis, :])
            if self.explore_sigma is not None:
                action += np.random.normal(0, self.explore_sigma, np.shape(action))
            try:
                judge_is_nan([action])
            except ValueError:
                print('processed_obs', processed_obs)
                # print('preprocessor_params', self.preprocessor.get_params())
                print('policy_weights', self.policy_with_value.policy.trainable_weights)
                action, logp = self.policy_with_value.compute_action(processed_obs[np.newaxis, :])
                judge_is_nan([action])
                raise ValueError
            obs_tp1, reward, self.done, info = self.env.step(action.numpy()[0])
            sample_count += 1
            self.done = 1 if sample_count > self.args.max_step else self.done

            batch_data.append((self.obs.copy(), action.numpy()[0], reward, obs_tp1.copy(),
                               self.done, self.info['future_n_point'], self.info['mask']))
            if self.done:
                self.obs, self.info = self.env.reset()
                sample_count = 0
            else:
                self.obs = obs_tp1.copy()
                self.info = info.copy()

        if self.worker_id == 1 and self.sample_times % self.args.worker_log_interval == 0:
            logger.info('Worker_info: {}'.format(self.get_stats()))

        self.num_sample += len(batch_data)
        self.sample_times += 1
        return batch_data

    def sample_with_count(self):
        batch_data = self.sample()
        return batch_data, len(batch_data)

    # def sample_with_count(self):
    #     try:
    #         batch_data = self.sample()
    #     except traci.exceptions.FatalTraCIError as e:
    #         self.restart_env()
    #         try:
    #             batch_data = self.sample()
    #         except Exception as ee:
    #             raise ValueError("The first sample after restart error")
    #     except AttributeError as e:
    #         batch_data = []
    #     except Exception as e:
    #         raise ValueError("Unknown Error")
    #     return batch_data, len(batch_data)
    #
    # def restart_env(self):
    #     try:
    #         self.env.traffic.close()
    #     except Exception as e:
    #         print(e)
    #         logger.info("Try to close env")
    #     logger.warning(f"worker: {self.worker_id} restart environment")
    #     self.env = CrossroadEnd2endMix(**args2envkwargs(self.args))
    #     self.obs, self.info = self.env.reset()
    #     self.done = False