#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2020/9/1
# @Author  : Yang Guan (Tsinghua Univ.)
# @FileName: evaluator.py
# =====================================

import logging
import os

from env_build.endtoend import CrossroadEnd2endMix
import numpy as np
import tensorflow as tf

from preprocessor import Preprocessor
from utils.misc import TimerStat, args2envkwargs

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class Evaluator(object):
    import tensorflow as tf
    tf.config.experimental.set_visible_devices([], 'GPU')
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)

    def __init__(self, policy_cls, env_id, args):
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        self.args = args
        self.env = CrossroadEnd2endMix(**args2envkwargs(args))
        self.policy_with_value = policy_cls(self.args)
        self.iteration = 0
        if self.args.mode == 'training':
            self.log_dir = self.args.log_dir + '/evaluator'
        else:
            self.log_dir = self.args.test_log_dir
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.preprocessor = Preprocessor((self.args.obs_dim, ), self.args.obs_preprocess_type, self.args.reward_preprocess_type,
                                         self.args.reward_scale, self.args.reward_shift, args=self.args, gamma=self.args.gamma)

        self.writer = self.tf.summary.create_file_writer(self.log_dir)
        self.stats = {}
        self.eval_timer = TimerStat()
        self.eval_times = 0

    def get_stats(self):
        self.stats.update(dict(eval_time=self.eval_timer.mean))
        return self.stats

    def load_weights(self, load_dir, iteration):
        self.policy_with_value.load_weights(load_dir, iteration)

    def load_ppc_params(self, load_dir):
        self.preprocessor.load_params(load_dir)

    def evaluate_saved_model(self, model_load_dir, ppc_params_load_dir, iteration):
        self.load_weights(model_load_dir, iteration)
        self.load_ppc_params(ppc_params_load_dir)

    def run_an_episode(self, steps=None, render=True):
        reward_list = []
        reward_info_dict_list = []
        done = 0
        obs = self.env.reset()
        if render: self.env.render()
        if steps is not None:
            for _ in range(steps):
                # extract infos for each kind of participants
                start = 0;
                end = self.args.state_ego_dim + self.args.state_track_dim
                obs_ego = obs[start:end]
                start = end;
                end = start + self.args.state_bike_dim
                obs_bike = obs[start:end]
                start = end;
                end = start + self.args.state_person_dim
                obs_person = obs[start:end]
                start = end;
                end = start + self.args.state_veh_dim
                obs_veh = obs[start:end]
                obs_bike = np.reshape(obs_bike, [-1, self.args.per_bike_dim])
                obs_person = np.reshape(obs_person, [-1, self.args.per_person_dim])
                obs_veh = np.reshape(obs_veh, [-1, self.args.per_veh_dim])
                processed_obs_ego, processed_obs_bike, processed_obs_person, processed_obs_veh \
                    = self.preprocessor.tf_process_obses_PI(obs_ego, obs_bike, obs_person, obs_veh)
                processed_obs_other = tf.concat([processed_obs_bike, processed_obs_person, processed_obs_veh], axis=0)

                PI_obs_other = tf.reduce_sum(self.policy_with_value.compute_PI(processed_obs_other), axis=0)
                processed_obs = np.concatenate((processed_obs_ego, PI_obs_other.numpy()), axis=0)

                action = self.policy_with_value.compute_mode(processed_obs[np.newaxis, :])
                obs, reward, done, info = self.env.step(action.numpy()[0])
                reward_info_dict_list.append(info['reward_info'])
                if render: self.env.render()
                reward_list.append(reward)
        else:
            while not done:
                start = 0; end = self.args.state_ego_dim + self.args.state_track_dim
                obs_ego = obs[start:end]
                start = end; end = start + self.args.state_bike_dim
                obs_bike = obs[start:end]
                start = end; end = start + self.args.state_person_dim
                obs_person = obs[start:end]
                start = end; end = start + self.args.state_veh_dim
                obs_veh = obs[start:end]
                obs_bike = np.reshape(obs_bike, [-1, self.args.per_bike_dim])
                obs_person = np.reshape(obs_person, [-1, self.args.per_person_dim])
                obs_veh = np.reshape(obs_veh, [-1, self.args.per_veh_dim])
                processed_obs_ego, processed_obs_bike, processed_obs_person, processed_obs_veh \
                    = self.preprocessor.tf_process_obses_PI(obs_ego, obs_bike, obs_person, obs_veh)
                processed_obs_other = tf.concat([processed_obs_bike, processed_obs_person, processed_obs_veh], axis=0)

                PI_obs_other = tf.reduce_sum(self.policy_with_value.compute_PI(processed_obs_other), axis=0)
                processed_obs = np.concatenate((processed_obs_ego, PI_obs_other.numpy()), axis=0)

                action = self.policy_with_value.compute_mode(processed_obs[np.newaxis, :])
                obs, reward, done, info = self.env.step(action.numpy()[0])
                reward_info_dict_list.append(info['reward_info'])
                if render: self.env.render()
                reward_list.append(reward)
        episode_return = sum(reward_list)
        episode_len = len(reward_list)
        info_dict = dict()
        for key in reward_info_dict_list[0].keys():
            info_key = list(map(lambda x: x[key], reward_info_dict_list))
            mean_key = sum(info_key) / len(info_key)
            info_dict.update({key: mean_key})
        info_dict.update(dict(episode_return=episode_return,
                              episode_len=episode_len))
        return info_dict

    def run_n_episode(self, n):
        list_of_info_dict = []
        for _ in range(n):
            logger.info('logging {}-th episode'.format(_))
            info_dict = self.run_an_episode(self.args.fixed_steps, self.args.eval_render)
            list_of_info_dict.append(info_dict.copy())
        n_info_dict = dict()
        for key in list_of_info_dict[0].keys():
            info_key = list(map(lambda x: x[key], list_of_info_dict))
            mean_key = sum(info_key) / len(info_key)
            n_info_dict.update({key: mean_key})
        return n_info_dict

    def set_weights(self, weights):
        self.policy_with_value.set_weights(weights)

    def set_ppc_params(self, params):
        self.preprocessor.set_params(params)

    def run_evaluation(self, iteration):
        with self.eval_timer:
            self.iteration = iteration
            n_info_dict = self.run_n_episode(self.args.num_eval_episode)
            with self.writer.as_default():
                for key, val in n_info_dict.items():
                    self.tf.summary.scalar("evaluation/{}".format(key), val, step=self.iteration)
                for key, val in self.get_stats().items():
                    self.tf.summary.scalar("evaluation/{}".format(key), val, step=self.iteration)
                self.writer.flush()
        if self.eval_times % self.args.eval_log_interval == 0:
            logger.info('Evaluator_info: {}, {}'.format(self.get_stats(),n_info_dict))
        self.eval_times += 1


class EvaluatorWithAttention(object):
    import tensorflow as tf
    tf.config.experimental.set_visible_devices([], 'GPU')
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)

    def __init__(self, policy_cls, env_id, args):
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        self.args = args
        self.env = CrossroadEnd2endMix(**args2envkwargs(args))
        self.policy_with_value = policy_cls(self.args)
        self.iteration = 0
        if self.args.mode == 'training':
            self.log_dir = self.args.log_dir + '/evaluator'
        else:
            self.log_dir = self.args.test_log_dir
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.preprocessor = Preprocessor((self.args.obs_dim, ), self.args.obs_preprocess_type, self.args.reward_preprocess_type,
                                         self.args.reward_scale, self.args.reward_shift, args=self.args, gamma=self.args.gamma)

        self.writer = self.tf.summary.create_file_writer(self.log_dir)
        self.stats = {}
        self.eval_timer = TimerStat()
        self.eval_times = 0


    def get_stats(self):
        self.stats.update(dict(eval_time=self.eval_timer.mean))
        return self.stats

    def load_weights(self, load_dir, iteration):
        self.policy_with_value.load_weights(load_dir, iteration)

    def load_ppc_params(self, load_dir):
        self.preprocessor.load_params(load_dir)

    def evaluate_saved_model(self, model_load_dir, ppc_params_load_dir, iteration):
        self.load_weights(model_load_dir, iteration)
        self.load_ppc_params(ppc_params_load_dir)

    def run_an_episode(self, steps=None, render=True):
        reward_list = []
        reward_info_dict_list = []
        done = 0
        obs, info = self.env.reset()
        if render: self.env.render()
        if steps is not None:
            for _ in range(steps):
                # extract infos for each kind of participants
                start = 0; 
                end = self.args.state_ego_dim + self.args.state_track_dim + self.args.state_light_dim + self.args.state_task_dim
                obs_ego = obs[start:end]
                start = end
                end = start + self.args.Attn_in_total_dim
                obs_others = obs[start:end]

                processed_obs_ego, processed_obs_others \
                    = self.preprocessor.tf_process_obses_attention(obs_ego, obs_others)
                mask = tf.cast(self.info['mask'], dtype=tf.float32)
                attention_obs_others = self.policy_with_value.compute_Attn(processed_obs_others, mask)
                processed_obs = np.concatenate((processed_obs_ego, tf.squeeze(attention_obs_others, axis=0).numpy()), axis=0)

                action = self.policy_with_value.compute_mode(processed_obs[np.newaxis, :])
                obs, reward, done, info = self.env.step(action.numpy()[0])
                reward_info_dict_list.append(info['reward_info'])
                if render: self.env.render()
                reward_list.append(reward)
        else:
            while not done:
                start = 0; 
                end = self.args.state_ego_dim + self.args.state_track_dim + self.args.state_light_dim + self.args.state_task_dim
                obs_ego = obs[start:end]
                start = end
                end = start + self.args.Attn_in_total_dim
                obs_others = obs[start:end]

                processed_obs_ego, processed_obs_others \
                    = self.preprocessor.tf_process_obses_attention(obs_ego, obs_others)
                mask = tf.cast(self.info['mask'], dtype=tf.float32)
                attention_obs_others = self.policy_with_value.compute_Attn(processed_obs_others, mask)
                processed_obs = np.concatenate((processed_obs_ego, tf.squeeze(attention_obs_others, axis=0).numpy()), axis=0)

                action = self.policy_with_value.compute_mode(processed_obs[np.newaxis, :])
                obs, reward, done, info = self.env.step(action.numpy()[0])
                reward_info_dict_list.append(info['reward_info'])
                if render: self.env.render()
                reward_list.append(reward)

        episode_return = sum(reward_list)
        episode_len = len(reward_list)
        info_dict = dict()
        for key in reward_info_dict_list[0].keys():
            info_key = list(map(lambda x: x[key], reward_info_dict_list))
            mean_key = sum(info_key) / len(info_key)
            info_dict.update({key: mean_key})
        info_dict.update(dict(episode_return=episode_return,
                              episode_len=episode_len))
        return info_dict

    def run_n_episode(self, n):
        list_of_info_dict = []
        for _ in range(n):
            logger.info('logging {}-th episode'.format(_))
            info_dict = self.run_an_episode(self.args.fixed_steps, self.args.eval_render)
            list_of_info_dict.append(info_dict.copy())
        n_info_dict = dict()
        for key in list_of_info_dict[0].keys():
            info_key = list(map(lambda x: x[key], list_of_info_dict))
            mean_key = sum(info_key) / len(info_key)
            n_info_dict.update({key: mean_key})
        return n_info_dict

    def set_weights(self, weights):
        self.policy_with_value.set_weights(weights)

    def set_ppc_params(self, params):
        self.preprocessor.set_params(params)

    def run_evaluation(self, iteration):
        with self.eval_timer:
            self.iteration = iteration
            n_info_dict = self.run_n_episode(self.args.num_eval_episode)
            with self.writer.as_default():
                for key, val in n_info_dict.items():
                    self.tf.summary.scalar("evaluation/{}".format(key), val, step=self.iteration)
                for key, val in self.get_stats().items():
                    self.tf.summary.scalar("evaluation/{}".format(key), val, step=self.iteration)
                self.writer.flush()
        if self.eval_times % self.args.eval_log_interval == 0:
            logger.info('Evaluator_info: {}, {}'.format(self.get_stats(),n_info_dict))
        self.eval_times += 1


def test_trained_model(model_dir, ppc_params_dir, iteration):
    from train_script import built_mixedpg_parser
    from policy import PolicyWithQs
    args = built_mixedpg_parser()
    evaluator = Evaluator(PolicyWithQs, args.env_id, args)
    evaluator.load_weights(model_dir, iteration)
    evaluator.load_ppc_params(ppc_params_dir)
    return evaluator.metrics(1000, render=False, reset=False)

def test_evaluator():
    import ray
    ray.init()
    import time
    from train_script import built_parser
    from policy import Policy4Toyota
    args = built_parser('AMPC')
    # evaluator = Evaluator(Policy4Toyota, args.env_id, args)
    # evaluator.run_evaluation(3)
    evaluator = ray.remote(num_cpus=1)(Evaluator).remote(Policy4Toyota, args.env_id, args)
    evaluator.run_evaluation.remote(3)
    time.sleep(10000)


if __name__ == '__main__':
    test_evaluator()
