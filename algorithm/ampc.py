#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2020/9/1
# @Author  : Yang Guan (Tsinghua Univ.)
# @FileName: ampc.py
# =====================================

import logging

import numpy as np
from env_build.dynamics_and_models import EnvironmentModel

from preprocessor import Preprocessor
from utils.misc import TimerStat, args2envkwargs

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class AMPCLearner(object):
    import tensorflow as tf
    tf.config.experimental.set_visible_devices([], 'GPU')
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)

    def __init__(self, policy_cls, args):
        self.args = args
        self.policy_with_value = policy_cls(self.args)
        self.batch_data = {}
        self.all_data = {}
        self.M = self.args.M
        self.num_rollout_list_for_policy_update = self.args.num_rollout_list_for_policy_update

        self.model = EnvironmentModel(**args2envkwargs(args))
        self.preprocessor = Preprocessor((self.args.obs_dim, ), self.args.obs_preprocess_type, self.args.reward_preprocess_type,
                                         self.args.reward_scale, self.args.reward_shift, args=self.args,
                                         gamma=self.args.gamma)
        self.grad_timer = TimerStat()
        self.stats = {}
        self.info_for_buffer = {}

    def get_stats(self):
        return self.stats

    def get_states(self, processed_obses_ego, processed_obses_bike, processed_obses_person, processed_obses_veh, grad):
        PI_obses_bike = self.policy_with_value.compute_PI(processed_obses_bike)
        PI_obses_person = self.policy_with_value.compute_PI(processed_obses_person)
        PI_obses_veh = self.policy_with_value.compute_PI(processed_obses_veh)

        PI_obses_bike_sum, PI_obses_person_sum, PI_obses_veh_sum = [], [], []

        for i in range(len(processed_obses_ego)):
            PI_obses_bike_sum.append(self.tf.math.reduce_sum(PI_obses_bike[i * self.args.max_bike_num: (i+1) * self.args.max_bike_num, :],
                                                        keepdims=True, axis=0))
            PI_obses_person_sum.append(self.tf.math.reduce_sum(PI_obses_person[i * self.args.max_person_num: (i+1) * self.args.max_person_num, :],
                                                          keepdims=True, axis=0))
            PI_obses_veh_sum.append(self.tf.math.reduce_sum(PI_obses_veh[i * self.args.max_veh_num: (i+1) * self.args.max_veh_num, :],
                                                       keepdims=True, axis=0))
        PI_obses_bike_sum = self.tf.concat(PI_obses_bike_sum, axis=0)
        PI_obses_person_sum = self.tf.concat(PI_obses_person_sum, axis=0)
        PI_obses_veh_sum = self.tf.concat(PI_obses_veh_sum, axis=0)
        if not grad:
            PI_obses_bike_sum = self.tf.stop_gradient(PI_obses_bike_sum)
            PI_obses_person_sum = self.tf.stop_gradient(PI_obses_person_sum)
            PI_obses_veh_sum = self.tf.stop_gradient(PI_obses_veh_sum)
        if self.args.per_bike_dim == self.args.per_person_dim == self.args.per_veh_dim:
            PI_obses_other_sum = PI_obses_bike_sum + PI_obses_person_sum + PI_obses_veh_sum
        else:
            PI_obses_other_sum = self.tf.concat([PI_obses_bike_sum, PI_obses_person_sum, PI_obses_veh_sum],axis=1)
        processed_obses = self.tf.concat((processed_obses_ego, PI_obses_other_sum), axis=1)
        return processed_obses

    def get_info_for_buffer(self):
        return self.info_for_buffer

    def get_batch_data(self, batch_data, rb, indexes):
        self.batch_data = {'batch_obs_ego': batch_data[0].astype(np.float32),
                           'batch_obs_bike': batch_data[1].astype(np.float32),
                           'batch_obs_person': batch_data[2].astype(np.float32),
                           'batch_obs_veh': batch_data[3].astype(np.float32),
                           'batch_dones': batch_data[4].astype(np.float32),
                           'batch_ref_index': batch_data[5].astype(np.int32)
                           }

    def get_weights(self):
        return self.policy_with_value.get_weights()

    def set_weights(self, weights):
        return self.policy_with_value.set_weights(weights)

    def set_ppc_params(self, params):
        self.preprocessor.set_params(params)

    def punish_factor_schedule(self, ite):
        init_pf = self.args.init_punish_factor
        interval = self.args.pf_enlarge_interval
        amplifier = self.args.pf_amplifier
        pf = init_pf * self.tf.pow(amplifier, self.tf.cast(ite//interval, self.tf.float32))
        return pf

    def model_rollout_for_update(self, start_obses_ego, start_obses_bike, start_obses_person, start_obses_veh, ite, mb_ref_index):
        start_obses_ego = self.tf.tile(start_obses_ego, [self.M, 1])
        start_obses_bike = self.tf.tile(start_obses_bike, [self.M, 1])
        start_obses_person = self.tf.tile(start_obses_person, [self.M, 1])
        start_obses_veh = self.tf.tile(start_obses_veh, [self.M, 1])

        self.model.reset(start_obses_ego, start_obses_bike, start_obses_person, start_obses_veh, mb_ref_index)

        rewards_sum = self.tf.zeros((start_obses_ego.shape[0],))
        punish_terms_for_training_sum = self.tf.zeros((start_obses_ego.shape[0],))
        real_punish_terms_sum = self.tf.zeros((start_obses_ego.shape[0],))
        veh2veh4real_sum = self.tf.zeros((start_obses_ego.shape[0],))
        veh2road4real_sum = self.tf.zeros((start_obses_ego.shape[0],))
        veh2bike4real_sum = self.tf.zeros((start_obses_ego.shape[0],))
        veh2person4real_sum = self.tf.zeros((start_obses_ego.shape[0],))
        entropy_sum = self.tf.zeros((start_obses_ego.shape[0],))

        pf = self.punish_factor_schedule(ite)
        obses_ego, obses_bike, obses_person, obses_veh = start_obses_ego, start_obses_bike, start_obses_person, start_obses_veh
        processed_obses_ego, processed_obses_bike, processed_obses_person, processed_obses_veh \
            = self.preprocessor.tf_process_obses_PI(obses_ego, obses_bike, obses_person, obses_veh)
        # no supplement vehicle currently
        processed_obses = self.get_states(processed_obses_ego, processed_obses_bike, processed_obses_person, processed_obses_veh, grad=True)
        obj_v_pred = self.policy_with_value.compute_obj_v(processed_obses)

        for _ in range(self.num_rollout_list_for_policy_update[0]):
            processed_obses_ego, processed_obses_bike, processed_obses_person, processed_obses_veh \
                = self.preprocessor.tf_process_obses_PI(obses_ego, obses_bike, obses_person, obses_veh)
            processed_obses = self.get_states(processed_obses_ego, processed_obses_bike, processed_obses_person,
                                              processed_obses_veh, grad=False)
            actions, logps = self.policy_with_value.compute_action(processed_obses)
            obses_ego, obses_bike, obses_person, obses_veh, rewards, punish_terms_for_training, real_punish_term, \
                veh2veh4real, veh2road4real, veh2bike4real, veh2person4real = self.model.rollout_out(actions)
            rewards_sum += self.preprocessor.tf_process_rewards(rewards)
            punish_terms_for_training_sum += self.args.reward_scale * punish_terms_for_training
            real_punish_terms_sum += self.args.reward_scale * real_punish_term
            veh2veh4real_sum += self.args.reward_scale * veh2veh4real
            veh2road4real_sum += self.args.reward_scale * veh2road4real
            veh2bike4real_sum += self.args.reward_scale * veh2bike4real
            veh2person4real_sum += self.args.reward_scale * veh2person4real
            entropy_sum += -logps

        # obj v loss
        obj_v_loss = self.tf.reduce_mean(self.tf.square(obj_v_pred - self.tf.stop_gradient(-rewards_sum)))
        # con_v_loss = self.tf.reduce_mean(self.tf.square(con_v_pred - self.tf.stop_gradient(real_punish_terms_sum)))

        # pg loss
        obj_loss = -self.tf.reduce_mean(rewards_sum)
        punish_term_for_training = self.tf.reduce_mean(punish_terms_for_training_sum)
        punish_loss = self.tf.stop_gradient(pf) * punish_term_for_training
        pg_loss = obj_loss + punish_loss

        real_punish_term = self.tf.reduce_mean(real_punish_terms_sum)
        veh2veh4real = self.tf.reduce_mean(veh2veh4real_sum)
        veh2road4real = self.tf.reduce_mean(veh2road4real_sum)
        veh2bike4real = self.tf.reduce_mean(veh2bike4real_sum)
        veh2person4real = self.tf.reduce_mean(veh2person4real_sum)

        policy_entropy = self.tf.reduce_mean(entropy_sum)

        return obj_v_loss, obj_loss, punish_term_for_training, punish_loss, pg_loss,\
               real_punish_term, veh2veh4real, veh2road4real, veh2bike4real, veh2person4real, pf, policy_entropy

    @tf.function
    def forward_and_backward(self, mb_obs_ego, mb_obs_bike, mb_obs_person, mb_obs_veh, ite, mb_ref_index):
        with self.tf.GradientTape(persistent=True) as tape:
            obj_v_loss, obj_loss, punish_term_for_training, punish_loss, pg_loss, \
            real_punish_term, veh2veh4real, veh2road4real, veh2bike4real, veh2person4real, pf, policy_entropy\
                = self.model_rollout_for_update(mb_obs_ego, mb_obs_bike, mb_obs_person, mb_obs_veh, ite, mb_ref_index)

        with self.tf.name_scope('policy_gradient') as scope:
            pg_grad = tape.gradient(pg_loss, self.policy_with_value.policy.trainable_weights)
        with self.tf.name_scope('obj_v_gradient') as scope:
            obj_v_grad = tape.gradient(obj_v_loss, self.policy_with_value.obj_v.trainable_weights)
            PI_net_grad = tape.gradient(obj_v_loss, self.policy_with_value.PI_net.trainable_weights)

        # with self.tf.name_scope('con_v_gradient') as scope:
        #     con_v_grad = tape.gradient(con_v_loss, self.policy_with_value.con_v.trainable_weights)

        return pg_grad, obj_v_grad, PI_net_grad, obj_v_loss, obj_loss, \
               punish_term_for_training, punish_loss, pg_loss,\
               real_punish_term, veh2veh4real, veh2road4real, veh2bike4real, veh2person4real, pf, policy_entropy

    def export_graph(self, writer):
        mb_obs = self.batch_data['batch_obs']
        self.tf.summary.trace_on(graph=True, profiler=False)
        self.forward_and_backward(mb_obs, self.tf.convert_to_tensor(0, self.tf.int32),
                                  self.tf.zeros((len(mb_obs),), dtype=self.tf.int32))
        with writer.as_default():
            self.tf.summary.trace_export(name="policy_forward_and_backward", step=0)

    def compute_gradient(self, samples, rb, indexs, iteration):
        self.get_batch_data(samples, rb, indexs)
        mb_obs_ego = self.tf.constant(self.batch_data['batch_obs_ego'])
        mb_obs_bike = self.tf.constant(self.batch_data['batch_obs_bike'])
        mb_obs_person = self.tf.constant(self.batch_data['batch_obs_person'])
        mb_obs_veh = self.tf.constant(self.batch_data['batch_obs_veh'])
        iteration = self.tf.convert_to_tensor(iteration, self.tf.int32)
        mb_ref_index = self.tf.constant(self.batch_data['batch_ref_index'], self.tf.int32)

        with self.grad_timer:
            pg_grad, obj_v_grad, PI_net_grad, obj_v_loss, obj_loss, \
            punish_term_for_training, punish_loss, pg_loss, \
            real_punish_term, veh2veh4real, veh2road4real, veh2bike4real, veh2person4real, pf, policy_entropy =\
                self.forward_and_backward(mb_obs_ego, mb_obs_bike, mb_obs_person, mb_obs_veh, iteration, mb_ref_index)

            pg_grad, pg_grad_norm = self.tf.clip_by_global_norm(pg_grad, self.args.gradient_clip_norm)
            obj_v_grad, obj_v_grad_norm = self.tf.clip_by_global_norm(obj_v_grad, self.args.gradient_clip_norm)
            PI_net_grad, PI_net_grad_norm = self.tf.clip_by_global_norm(PI_net_grad, self.args.gradient_clip_norm)
            # con_v_grad, con_v_grad_norm = self.tf.clip_by_global_norm(con_v_grad, self.args.gradient_clip_norm)

        self.stats.update(dict(
            iteration=iteration,
            grad_time=self.grad_timer.mean,
            obj_loss=obj_loss.numpy(),
            punish_term_for_training=punish_term_for_training.numpy(),
            real_punish_term=real_punish_term.numpy(),
            veh2veh4real=veh2veh4real.numpy(),
            veh2road4real=veh2road4real.numpy(),
            veh2bike4real=veh2bike4real.numpy(),
            veh2person4real=veh2person4real.numpy(),
            punish_loss=punish_loss.numpy(),
            pg_loss=pg_loss.numpy(),
            obj_v_loss=obj_v_loss.numpy(),
            # con_v_loss=con_v_loss.numpy(),
            punish_factor=pf.numpy(),
            pg_grads_norm=pg_grad_norm.numpy(),
            obj_v_grad_norm=obj_v_grad_norm.numpy(),
            PI_net_grad_norm=PI_net_grad_norm.numpy(),
            policy_entropy=policy_entropy.numpy(),
            # con_v_grad_norm=con_v_grad_norm.numpy()
        ))

        grads = obj_v_grad + pg_grad + PI_net_grad
        return list(map(lambda x: x.numpy(), grads))


class AMPCLearnerWithAttention(object):
    import tensorflow as tf
    tf.config.experimental.set_visible_devices([], 'GPU')
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)

    def __init__(self, policy_cls, args):
        self.args = args
        self.policy_with_value = policy_cls(self.args)
        self.batch_data = {}
        self.all_data = {}
        self.M = self.args.M
        self.num_rollout_list_for_policy_update = self.args.num_rollout_list_for_policy_update

        self.model = EnvironmentModel(**args2envkwargs(args))
        self.preprocessor = Preprocessor((self.args.obs_dim, ), self.args.obs_preprocess_type, self.args.reward_preprocess_type,
                                         self.args.reward_scale, self.args.reward_shift, args=self.args,
                                         gamma=self.args.gamma)
        self.grad_timer = TimerStat()
        self.stats = {}
        self.info_for_buffer = {}

    def get_stats(self):
        return self.stats
    
    def get_states(self, processed_obses_ego, processed_obses_others, mask, grad):
        obses_others = self.policy_with_value.compute_Attn(processed_obses_others, mask)
        if not grad:
            obses_others = self.tf.stop_gradient(obses_others)
        processed_obses = self.tf.concat((processed_obses_ego, obses_others), axis=1)
        return processed_obses

    def get_info_for_buffer(self):
        return self.info_for_buffer

    def get_batch_data(self, batch_data, rb, indexes):
        self.batch_data = {'batch_obs_ego': batch_data[0].astype(np.float32),
                           'batch_obs_others': batch_data[1].astype(np.float32),
                           'batch_dones': batch_data[2].astype(np.float32),
                           'batch_ref_index': batch_data[3].astype(np.int32),
                           'batch_mask': batch_data[4].astype(np.float32),
                           }
    def get_weights(self):
        return self.policy_with_value.get_weights()

    def set_weights(self, weights):
        return self.policy_with_value.set_weights(weights)

    def set_ppc_params(self, params):
        self.preprocessor.set_params(params)

    def punish_factor_schedule(self, ite):
        init_pf = self.args.init_punish_factor
        interval = self.args.pf_enlarge_interval
        amplifier = self.args.pf_amplifier
        pf = init_pf * self.tf.pow(amplifier, self.tf.cast(ite//interval, self.tf.float32))
        return pf
    
    def model_rollout_for_update(self, start_obses_ego, start_obses_others, ite, mb_ref_index, mb_mask):
        start_obses_ego = self.tf.tile(start_obses_ego, [self.M, 1])
        start_obses_others = self.tf.tile(start_obses_others, [self.M, 1])

        self.model.reset(start_obses_ego, start_obses_others, mb_ref_index)

        rewards_sum = self.tf.zeros((start_obses_ego.shape[0],))
        punish_terms_for_training_sum = self.tf.zeros((start_obses_ego.shape[0],))
        real_punish_terms_sum = self.tf.zeros((start_obses_ego.shape[0],))
        veh2veh4real_sum = self.tf.zeros((start_obses_ego.shape[0],))
        veh2road4real_sum = self.tf.zeros((start_obses_ego.shape[0],))
        veh2bike4real_sum = self.tf.zeros((start_obses_ego.shape[0],))
        veh2person4real_sum = self.tf.zeros((start_obses_ego.shape[0],))
        entropy_sum = self.tf.zeros((start_obses_ego.shape[0],))

        pf = self.punish_factor_schedule(ite)
        obses_ego, obses_others = start_obses_ego, start_obses_others
        processed_obses_ego, processed_obses_others \
            = self.preprocessor.tf_process_obses_attention(obses_ego, obses_others)
        # no supplement vehicle currently
        processed_obses = self.get_states(processed_obses_ego, processed_obses_others, mb_mask, grad=False)
        obj_v_pred = self.policy_with_value.compute_obj_v(processed_obses)

        for _ in range(self.num_rollout_list_for_policy_update[0]):
            processed_obses_ego, processed_obses_others \
                = self.preprocessor.tf_process_obses_attention(obses_ego, obses_others)
            processed_obses = self.get_states(processed_obses_ego, processed_obses_others, mb_mask, grad=True)
            actions, logps = self.policy_with_value.compute_action(processed_obses)
            obses_ego, obses_others, rewards, punish_terms_for_training, real_punish_term, \
                veh2veh4real, veh2road4real, veh2bike4real, veh2person4real = self.model.rollout_out(actions)
            
            rewards_sum += self.preprocessor.tf_process_rewards(rewards)
            punish_terms_for_training_sum += self.args.reward_scale * punish_terms_for_training
            real_punish_terms_sum += self.args.reward_scale * real_punish_term
            veh2veh4real_sum += self.args.reward_scale * veh2veh4real
            veh2road4real_sum += self.args.reward_scale * veh2road4real
            veh2bike4real_sum += self.args.reward_scale * veh2bike4real
            veh2person4real_sum += self.args.reward_scale * veh2person4real
            entropy_sum += -logps

        # obj v loss
        obj_v_loss = self.tf.reduce_mean(self.tf.square(obj_v_pred - self.tf.stop_gradient(-rewards_sum)))
        # con_v_loss = self.tf.reduce_mean(self.tf.square(con_v_pred - self.tf.stop_gradient(real_punish_terms_sum)))

        # pg loss
        obj_loss = -self.tf.reduce_mean(rewards_sum)
        punish_term_for_training = self.tf.reduce_mean(punish_terms_for_training_sum)
        punish_loss = self.tf.stop_gradient(pf) * punish_term_for_training
        pg_loss = obj_loss + punish_loss

        real_punish_term = self.tf.reduce_mean(real_punish_terms_sum)
        veh2veh4real = self.tf.reduce_mean(veh2veh4real_sum)
        veh2road4real = self.tf.reduce_mean(veh2road4real_sum)
        veh2bike4real = self.tf.reduce_mean(veh2bike4real_sum)
        veh2person4real = self.tf.reduce_mean(veh2person4real_sum)

        policy_entropy = self.tf.reduce_mean(entropy_sum)

        return obj_v_loss, obj_loss, punish_term_for_training, punish_loss, pg_loss,\
               real_punish_term, veh2veh4real, veh2road4real, veh2bike4real, veh2person4real, pf, policy_entropy

    @tf.function
    def forward_and_backward(self, mb_obs_ego, mb_obs_others, ite, mb_ref_index, mb_mask):
        with self.tf.GradientTape(persistent=True) as tape:
            obj_v_loss, obj_loss, punish_term_for_training, punish_loss, pg_loss, \
            real_punish_term, veh2veh4real, veh2road4real, veh2bike4real, veh2person4real, pf, policy_entropy\
                = self.model_rollout_for_update(mb_obs_ego, mb_obs_others, ite, mb_ref_index, mb_mask)

        with self.tf.name_scope('policy_gradient') as scope:
            pg_grad = tape.gradient(pg_loss, self.policy_with_value.policy.trainable_weights)
            Attn_net_grad = tape.gradient(pg_loss, self.policy_with_value.Attn_net.trainable_weights)
        with self.tf.name_scope('obj_v_gradient') as scope:
            obj_v_grad = tape.gradient(obj_v_loss, self.policy_with_value.obj_v.trainable_weights)

        # with self.tf.name_scope('con_v_gradient') as scope:
        #     con_v_grad = tape.gradient(con_v_loss, self.policy_with_value.con_v.trainable_weights)

        return pg_grad, obj_v_grad, Attn_net_grad, obj_v_loss, obj_loss, \
               punish_term_for_training, punish_loss, pg_loss,\
               real_punish_term, veh2veh4real, veh2road4real, veh2bike4real, veh2person4real, pf, policy_entropy

    def export_graph(self, writer):
        mb_obs = self.batch_data['batch_obs']
        self.tf.summary.trace_on(graph=True, profiler=False)
        self.forward_and_backward(mb_obs, self.tf.convert_to_tensor(0, self.tf.int32),
                                  self.tf.zeros((len(mb_obs),), dtype=self.tf.int32))
        with writer.as_default():
            self.tf.summary.trace_export(name="policy_forward_and_backward", step=0)

    def compute_gradient(self, samples, rb, indexs, iteration):
        self.get_batch_data(samples, rb, indexs)
        mb_obs_ego = self.tf.constant(self.batch_data['batch_obs_ego'])
        mb_obs_others = self.tf.constant(self.batch_data['batch_obs_others'])
        iteration = self.tf.convert_to_tensor(iteration, self.tf.int32)
        mb_ref_index = self.tf.constant(self.batch_data['batch_ref_index'], self.tf.int32)
        mb_mask = self.tf.constant(self.tf.cast(self.batch_data['batch_mask'], self.tf.float32))

        with self.grad_timer:
            pg_grad, obj_v_grad, Attn_net_grad, obj_v_loss, obj_loss, \
            punish_term_for_training, punish_loss, pg_loss, \
            real_punish_term, veh2veh4real, veh2road4real, veh2bike4real, veh2person4real, pf, policy_entropy =\
                self.forward_and_backward(mb_obs_ego, mb_obs_others, iteration, mb_ref_index, mb_mask)

            pg_grad, pg_grad_norm = self.tf.clip_by_global_norm(pg_grad, self.args.gradient_clip_norm)
            obj_v_grad, obj_v_grad_norm = self.tf.clip_by_global_norm(obj_v_grad, self.args.gradient_clip_norm)
            Attn_net_grad, Attn_net_grad_norm = self.tf.clip_by_global_norm(Attn_net_grad, self.args.gradient_clip_norm)
            # con_v_grad, con_v_grad_norm = self.tf.clip_by_global_norm(con_v_grad, self.args.gradient_clip_norm)

        self.stats.update(dict(
            iteration=iteration,
            grad_time=self.grad_timer.mean,
            obj_loss=obj_loss.numpy(),
            punish_term_for_training=punish_term_for_training.numpy(),
            real_punish_term=real_punish_term.numpy(),
            veh2veh4real=veh2veh4real.numpy(),
            veh2road4real=veh2road4real.numpy(),
            veh2bike4real=veh2bike4real.numpy(),
            veh2person4real=veh2person4real.numpy(),
            punish_loss=punish_loss.numpy(),
            pg_loss=pg_loss.numpy(),
            obj_v_loss=obj_v_loss.numpy(),
            # con_v_loss=con_v_loss.numpy(),
            punish_factor=pf.numpy(),
            pg_grads_norm=pg_grad_norm.numpy(),
            obj_v_grad_norm=obj_v_grad_norm.numpy(),
            PI_net_grad_norm=Attn_net_grad_norm.numpy(),
            policy_entropy=policy_entropy.numpy(),
            # con_v_grad_norm=con_v_grad_norm.numpy()
        ))

        grads = obj_v_grad + pg_grad + Attn_net_grad
        return list(map(lambda x: x.numpy(), grads))

if __name__ == '__main__':
    pass
