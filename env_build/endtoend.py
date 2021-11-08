#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2020/11/08
# @Author  : Yang Guan (Tsinghua Univ.)
# @FileName: endtoend.py
# =====================================

import warnings
from collections import OrderedDict
from math import cos, sin, pi, sqrt
import random
from random import choice

import gym
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D
from matplotlib.collections import PatchCollection
import numpy as np
from gym.utils import seeding

from env_build.dynamics_and_models import VehicleDynamics, ReferencePath, EnvironmentModel
from env_build.endtoend_env_utils import *
from env_build.traffic import Traffic

warnings.filterwarnings("ignore")


def convert_observation_to_space(observation):
    if isinstance(observation, dict):
        space = gym.spaces.Dict(OrderedDict([
            (key, convert_observation_to_space(value))
            for key, value in observation.items()
        ]))
    elif isinstance(observation, np.ndarray):
        low = np.full(observation.shape, -float('inf'))
        high = np.full(observation.shape, float('inf'))
        space = gym.spaces.Box(low, high, dtype=np.float32)
    else:
        raise NotImplementedError(type(observation), observation)

    return space


class CrossroadEnd2endMix(gym.Env):
    def __init__(self,
                 mode='training',
                 multi_display=False,
                 state_mode='fix',  # 'dyna'
                 future_point_num=25,
                 traffic_mode='auto',  # 'auto'
                 **kwargs):
        self.mode = mode
        self.traffic_mode = traffic_mode
        self.dynamics = VehicleDynamics()
        self.interested_other = None
        self.detected_vehicles = None
        self.all_other = None
        self.ego_dynamics = None
        self.state_mode = state_mode
        self.init_state = {}
        self.ego_l, self.ego_w = Para.L, Para.W
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

        self.seed()
        self.light_phase = None
        self.light_encoding = None
        self.task_encoding = None
        self.step_length = 100  # ms

        self.step_time = self.step_length / 1000.0
        self.obs = None
        self.action = None

        self.done_type = 'not_done_yet'
        self.reward_info = None
        self.ego_info_dim = Para.EGO_ENCODING_DIM
        self.track_info_dim = Para.TRACK_ENCODING_DIM
        self.light_info_dim = Para.LIGHT_ENCODING_DIM
        self.task_info_dim = Para.TASK_ENCODING_DIM
        self.ref_info_dim = Para.REF_ENCODING_DIM
        self.his_act_info_dim = Para.HIS_ACT_ENCODING_DIM
        self.per_other_info_dim = Para.PER_OTHER_INFO_DIM
        self.other_start_dim = sum([self.ego_info_dim, self.track_info_dim, self.light_info_dim,
                                    self.task_info_dim, self.ref_info_dim, self.his_act_info_dim])
        self.veh_num = Para.MAX_VEH_NUM
        self.bike_num = Para.MAX_BIKE_NUM
        self.person_num = Para.MAX_PERSON_NUM
        self.other_number = sum([self.veh_num, self.bike_num, self.person_num])

        self.veh_mode_dict = None
        self.bicycle_mode_dict = None
        self.person_mode_dict = None
        self.training_task = None
        self.env_model = None
        self.ref_path = None
        self.future_n_point = None
        self.future_point_num = future_point_num

        self.vector_noise = True
        if self.vector_noise:
            self.rng = np.random.default_rng(12345)

        self.action_store = ActionStore(maxlen=2)

        if not multi_display:
            self.traffic = Traffic(self.step_length,
                                   mode=self.mode,
                                   init_n_ego_dict=self.init_state,
                                   traffic_mode=traffic_mode)
            self.reset()
            action = self.action_space.sample()
            observation, _reward, done, _info = self.step(action)
            self._set_observation_space(observation)
            plt.ion()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self, **kwargs):  # kwargs include three keys
        self.light_phase = self.traffic.init_light()
        if self.traffic_mode == 'auto':
            self.training_task = choice(['left', 'straight', 'right'])
        else:
            self.training_task = self.traffic.case_task()
        self.task_encoding = TASK_ENCODING[self.training_task]
        self.light_encoding = LIGHT_ENCODING[self.light_phase]
        self.ref_path = ReferencePath(self.training_task, LIGHT_PHASE_TO_GREEN_OR_RED[self.light_phase])
        self.veh_mode_dict = VEHICLE_MODE_DICT[self.training_task]
        self.bicycle_mode_dict = BIKE_MODE_DICT[self.training_task]
        self.person_mode_dict = PERSON_MODE_DICT[self.training_task]
        self.env_model = EnvironmentModel()
        self.action_store.reset()
        self.init_state = self._reset_init_state()
        self.traffic.init_traffic(self.init_state, self.training_task)
        self.traffic.sim_step()
        ego_dynamics = self._get_ego_dynamics([self.init_state['ego']['v_x'],
                                               self.init_state['ego']['v_y'],
                                               self.init_state['ego']['r'],
                                               self.init_state['ego']['x'],
                                               self.init_state['ego']['y'],
                                               self.init_state['ego']['phi']],
                                              [0,
                                               0,
                                               self.dynamics.vehicle_params['miu'],
                                               self.dynamics.vehicle_params['miu']]
                                              )
        self._get_all_info(ego_dynamics)
        self.obs, other_mask_vector, self.future_n_point = self._get_obs()
        self.action = None
        self.reward_info = None
        self.done_type = 'not_done_yet'
        all_info = dict(future_n_point=self.future_n_point, mask=other_mask_vector)
        return self.obs, all_info

    def close(self):
        del self.traffic

    def step(self, action):
        self.action_store.put(action)
        self.action = self._action_transformation_for_end2end(action)
        reward, self.reward_info = self._compute_reward(self.obs, self.action, action)
        next_ego_state, next_ego_params = self._get_next_ego_state(self.action)
        ego_dynamics = self._get_ego_dynamics(next_ego_state, next_ego_params)
        self.traffic.set_own_car(dict(ego=ego_dynamics))
        self.traffic.sim_step()
        all_info = self._get_all_info(ego_dynamics)
        self.obs, other_mask_vector, self.future_n_point = self._get_obs()
        self.done_type, done = self._judge_done()
        self.reward_info.update({'final_rew': reward})
        all_info.update({'reward_info': self.reward_info, 'future_n_point': self.future_n_point, 'mask': other_mask_vector})
        return self.obs, reward, done, all_info

    def _set_observation_space(self, observation):
        self.observation_space = convert_observation_to_space(observation)
        return self.observation_space

    def _get_ego_dynamics(self, next_ego_state, next_ego_params):
        out = dict(v_x=next_ego_state[0],
                   v_y=next_ego_state[1],
                   r=next_ego_state[2],
                   x=next_ego_state[3],
                   y=next_ego_state[4],
                   phi=next_ego_state[5],
                   l=self.ego_l,
                   w=self.ego_w,
                   alpha_f=next_ego_params[0],
                   alpha_r=next_ego_params[1],
                   miu_f=next_ego_params[2],
                   miu_r=next_ego_params[3], )
        miu_f, miu_r = out['miu_f'], out['miu_r']
        F_zf, F_zr = self.dynamics.vehicle_params['F_zf'], self.dynamics.vehicle_params['F_zr']
        C_f, C_r = self.dynamics.vehicle_params['C_f'], self.dynamics.vehicle_params['C_r']
        alpha_f_bound, alpha_r_bound = 3 * miu_f * F_zf / C_f, 3 * miu_r * F_zr / C_r
        r_bound = miu_r * self.dynamics.vehicle_params['g'] / (abs(out['v_x']) + 1e-8)

        l, w, x, y, phi = out['l'], out['w'], out['x'], out['y'], out['phi']

        def cal_corner_point_of_ego_car():
            x0, y0, a0 = rotate_and_shift_coordination(l / 2, w / 2, 0, -x, -y, -phi)
            x1, y1, a1 = rotate_and_shift_coordination(l / 2, -w / 2, 0, -x, -y, -phi)
            x2, y2, a2 = rotate_and_shift_coordination(-l / 2, w / 2, 0, -x, -y, -phi)
            x3, y3, a3 = rotate_and_shift_coordination(-l / 2, -w / 2, 0, -x, -y, -phi)
            return (x0, y0), (x1, y1), (x2, y2), (x3, y3)

        corner_point = cal_corner_point_of_ego_car()
        out.update(dict(alpha_f_bound=alpha_f_bound,
                        alpha_r_bound=alpha_r_bound,
                        r_bound=r_bound,
                        corner_point=corner_point))

        return out

    def _get_all_info(self, ego_dynamics):  # used to update info, must be called every timestep before _get_obs
        # to fetch info
        self.all_other = self.traffic.n_ego_vehicles['ego']  # coordination 2
        self.ego_dynamics = ego_dynamics  # coordination 2
        self.light_phase = self.traffic.v_light

        # all_vehicles
        # dict(x=x, y=y, v=v, phi=a, l=length,
        #      w=width, route=route)

        all_info = dict(all_other=self.all_other,
                        ego_dynamics=self.ego_dynamics,
                        v_light=self.light_phase)
        return all_info

    def _judge_done(self):
        """
        :return:
         1: bad done: collision
         2: bad done: break_road_constrain
         3: good done: task succeed
         4: not done
        """
        if self.traffic.collision_flag:
            return 'collision', 1
        if self._break_road_constrain():
            return 'break_road_constrain', 1
        elif self._deviate_too_much():
            return 'deviate_too_much', 1
        elif self._break_stability():
            return 'break_stability', 1
        elif self._break_red_light():   # todo
            return 'break_red_light', 1
        elif self._is_achieve_goal():
            return 'good_done', 1
        else:
            return 'not_done_yet', 0

    def _deviate_too_much(self):
        delta_longi, delta_lateral, delta_phi, delta_v = self.obs[self.ego_info_dim:self.ego_info_dim + self.track_info_dim]
        return True if abs(delta_lateral) > 15 else False

    def _break_road_constrain(self):
        results = list(map(lambda x: judge_feasible(*x, self.training_task), self.ego_dynamics['corner_point']))
        return not all(results)

    def _break_stability(self):
        alpha_f, alpha_r, miu_f, miu_r = self.ego_dynamics['alpha_f'], self.ego_dynamics['alpha_r'], \
                                         self.ego_dynamics['miu_f'], self.ego_dynamics['miu_r']
        alpha_f_bound, alpha_r_bound = self.ego_dynamics['alpha_f_bound'], self.ego_dynamics['alpha_r_bound']
        r_bound = self.ego_dynamics['r_bound']
        # if -alpha_f_bound < alpha_f < alpha_f_bound \
        #         and -alpha_r_bound < alpha_r < alpha_r_bound and \
        #         -r_bound < self.ego_dynamics['r'] < r_bound:
        if -r_bound < self.ego_dynamics['r'] < r_bound:
            return False
        else:
            return True

    def _break_red_light(self):
        return True if self.light_phase > 2 and self.ego_dynamics[
            'y'] > -Para.CROSSROAD_SIZE_LON / 2 and self.training_task != 'right' else False

    def _is_achieve_goal(self):
        x = self.ego_dynamics['x']
        y = self.ego_dynamics['y']
        if self.training_task == 'left':
            return True if x < -Para.CROSSROAD_SIZE_LAT / 2 - 10 and Para.OFFSET_L + Para.GREEN_BELT_LAT < y < Para.OFFSET_L + Para.GREEN_BELT_LAT + Para.LANE_WIDTH_1 * 3 else False
        elif self.training_task == 'right':
            return True if x > Para.CROSSROAD_SIZE_LAT / 2 + 10 and Para.OFFSET_R - Para.LANE_WIDTH_1 * 3 < y < Para.OFFSET_R else False
        else:
            assert self.training_task == 'straight'
            return True if y > Para.CROSSROAD_SIZE_LON / 2 + 10 and Para.OFFSET_U + Para.GREEN_BELT_LON < x < Para.OFFSET_U + Para.GREEN_BELT_LON + Para.LANE_WIDTH_3 * 2 else False

    def _action_transformation_for_end2end(self, action):  # [-1, 1]
        action = np.clip(action, -1.05, 1.05)
        steer_norm, a_x_norm = action[0], action[1]
        scaled_steer = 0.4 * steer_norm
        scaled_a_x = 2.25 * a_x_norm - 0.75  # [-3, 1.5]
        # if self.light_phase != 0 and self.ego_dynamics['y'] < -25 and self.training_task != 'right':
        #     scaled_steer = 0.
        #     scaled_a_x = -3.
        scaled_action = np.array([scaled_steer, scaled_a_x], dtype=np.float32)
        return scaled_action

    def _get_next_ego_state(self, trans_action):
        current_v_x = self.ego_dynamics['v_x']
        current_v_y = self.ego_dynamics['v_y']
        current_r = self.ego_dynamics['r']
        current_x = self.ego_dynamics['x']
        current_y = self.ego_dynamics['y']
        current_phi = self.ego_dynamics['phi']
        steer, a_x = trans_action
        state = np.array([[current_v_x, current_v_y, current_r, current_x, current_y, current_phi]], dtype=np.float32)
        action = np.array([[steer, a_x]], dtype=np.float32)
        next_ego_state, next_ego_params = self.dynamics.prediction(state, action, 10)
        next_ego_state, next_ego_params = next_ego_state.numpy()[0], next_ego_params.numpy()[0]
        next_ego_state[0] = next_ego_state[0] if next_ego_state[0] >= 0 else 0.
        next_ego_state[-1] = deal_with_phi(next_ego_state[-1])
        return next_ego_state, next_ego_params

    def _get_obs(self, exit_='D'):
        # comment ego info, add noise to ego_vector
        # compute track error with noised ego_vector instead of label
        other_vector, other_mask_vector = self._construct_other_vector_short(exit_)
        ego_vector = self._construct_ego_vector_short()
        if self.vector_noise:
            other_vector = self._add_noise_to_vector(other_vector, 'other')
            ego_vector = self._add_noise_to_vector(ego_vector, 'ego')

        track_vector = self.ref_path.tracking_error_vector_vectorized(ego_vector[3], ego_vector[4], ego_vector[5], ego_vector[0]) # 3 for x; 4 foy y
        future_n_point = self.ref_path.get_future_n_point(ego_vector[3], ego_vector[4], self.future_point_num)
        self.light_encoding = LIGHT_ENCODING[self.light_phase]
        vector = np.concatenate((ego_vector, track_vector, self.light_encoding, self.task_encoding,
                                 self.ref_path.ref_encoding, self.action_store[0], self.action_store[1], other_vector), axis=0)
        vector = vector.astype(np.float32)
        vector = self._convert_to_rela(vector)

        return vector, other_mask_vector, future_n_point

    def _add_noise_to_vector(self, vector, vec_type=None):
        '''
        Enabled by the 'vector_noise' variable in this class
        Add noise to the vector of objects, whose order is (x, y, v, phi, l, w) for other and (v_x, v_y, r, x, y, phi) for ego

        Noise is i.i.d for each element in the vector, i.e. the covariance matrix is diagonal
        Different types of objs lead to different mean and var, which are defined in the 'Para' class in e2e_utils.py

        :params
            vector: np.array(6,)
            vec_type: str in ['ego', 'other']
        :return
            noise_vec: np.array(6,)
        '''
        assert self.vector_noise
        assert vec_type in ['ego', 'other']
        if vec_type == 'ego':
            return vector + self.rng.multivariate_normal(Para.EGO_MEAN, Para.EGO_VAR)
        elif vec_type == 'other':
            return vector + self.rng.multivariate_normal(Para.OTHERS_MEAN, Para.OTHERS_VAR)

    def _convert_to_rela(self, obs_abso):
        obs_ego, obs_track, obs_light, obs_task, obs_ref, obs_his_ac, obs_other = self._split_all(obs_abso)
        obs_other_reshape = self._reshape_other(obs_other)
        ego_x, ego_y = obs_ego[3], obs_ego[4]
        ego = np.array(([ego_x, ego_y] + [0.] * (self.per_other_info_dim - 2)), dtype=np.float32)
        ego = ego[np.newaxis, :]
        rela = obs_other_reshape - ego
        rela_obs_other = self._reshape_other(rela, reverse=True)
        return np.concatenate([obs_ego, obs_track, obs_light, obs_task, obs_ref, obs_his_ac, rela_obs_other], axis=0)

    def _convert_to_abso(self, obs_rela):
        obs_ego, obs_track, obs_light, obs_task, obs_ref, obs_his_ac, obs_other = self._split_all(obs_rela)
        obs_other_reshape = self._reshape_other(obs_other)
        ego_x, ego_y = obs_ego[3], obs_ego[4]
        ego = np.array(([ego_x, ego_y] + [0.] * (self.per_other_info_dim - 2)), dtype=np.float32)
        ego = ego[np.newaxis, :]
        abso = obs_other_reshape + ego
        abso_obs_other = self._reshape_other(abso, reverse=True)
        return np.concatenate([obs_ego, obs_track, obs_light, obs_task, obs_ref, obs_his_ac, abso_obs_other])

    def _split_all(self, obs):
        obs_ego = obs[:self.ego_info_dim]
        obs_track = obs[self.ego_info_dim:
                        self.ego_info_dim + self.track_info_dim]
        obs_light = obs[self.ego_info_dim + self.track_info_dim:
                        self.ego_info_dim + self.track_info_dim + self.light_info_dim]
        obs_task = obs[self.ego_info_dim + self.track_info_dim + self.light_info_dim:
                       self.ego_info_dim + self.track_info_dim + self.light_info_dim + self.task_info_dim]
        obs_ref = obs[self.ego_info_dim + self.track_info_dim + self.light_info_dim + self.task_info_dim:
                      self.ego_info_dim + self.track_info_dim + self.light_info_dim + self.task_info_dim + self.ref_info_dim]
        obs_his_ac = obs[self.ego_info_dim + self.track_info_dim + self.light_info_dim + self.task_info_dim + self.ref_info_dim:
                         self.other_start_dim]
        obs_other = obs[self.other_start_dim:]

        return obs_ego, obs_track, obs_light, obs_task, obs_ref, obs_his_ac, obs_other

    def _split_other(self, obs_other):
        obs_bike = obs_other[:self.bike_num * self.per_other_info_dim]
        obs_person = obs_other[self.bike_num * self.per_other_info_dim:
                               (self.bike_num + self.person_num) * self.per_other_info_dim]
        obs_veh = obs_other[(self.bike_num + self.person_num) * self.per_other_info_dim:]
        return obs_bike, obs_person, obs_veh

    def _reshape_other(self, obs_other, reverse=False):
        if reverse:
            return np.reshape(obs_other, (self.other_number * self.per_other_info_dim,))
        else:
            return np.reshape(obs_other, (self.other_number, self.per_other_info_dim))

    def _construct_ego_vector_short(self):
        ego_v_x = self.ego_dynamics['v_x']
        ego_v_y = self.ego_dynamics['v_y']
        ego_r = self.ego_dynamics['r']
        ego_x = self.ego_dynamics['x']
        ego_y = self.ego_dynamics['y']
        ego_phi = self.ego_dynamics['phi']
        ego_feature = [ego_v_x, ego_v_y, ego_r, ego_x, ego_y, ego_phi]
        return np.array(ego_feature, dtype=np.float32)

    def _construct_other_vector_short(self, exit_='D'):
        ego_x = self.ego_dynamics['x']
        ego_y = self.ego_dynamics['y']
        other_vector = []
        other_mask_vector = []

        name_settings = dict(D=dict(do='1o', di='1i', ro='2o', ri='2i', uo='3o', ui='3i', lo='4o', li='4i'),
                             R=dict(do='2o', di='2i', ro='3o', ri='3i', uo='4o', ui='4i', lo='1o', li='1i'),
                             U=dict(do='3o', di='3i', ro='4o', ri='4i', uo='1o', ui='1i', lo='2o', li='2i'),
                             L=dict(do='4o', di='4i', ro='1o', ri='1i', uo='2o', ui='2i', lo='3o', li='3i'))

        name_setting = name_settings[exit_]

        def filter_interested_other(vs, task):
            dl, du, dr, rd, rl, ru, ur, ud, ul, lu, lr, ld = [], [], [], [], [], [], [], [], [], [], [], []
            du_b, dr_b, rl_b, ru_b, ud_b, ul_b, lr_b, ld_b = [], [], [], [], [], [], [], []
            i1_0, o1_0, i2_0, o2_0, i3_0, o3_0, i4_0, o4_0, c0, c1, c2, c3, c_w0, c_w1, c_w2, c_w3 = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []

            # slice or fill to some number
            def slice_or_fill(sorted_list, fill_value, num):
                if len(sorted_list) >= num:
                    return sorted_list[:num]
                else:
                    while len(sorted_list) < num:
                        sorted_list.append(fill_value)
                    return sorted_list

            def cal_turn_rad(v):
                if not(-Para.CROSSROAD_SIZE_LAT/2 < v['x'] < Para.CROSSROAD_SIZE_LAT/2 and -Para.CROSSROAD_SIZE_LON/2 < v['y'] < Para.CROSSROAD_SIZE_LON/2):
                    turn_rad = 0.
                else:
                    start = v['route'][0]
                    end = v['route'][1]
                    if (start == name_setting['do'] and end == name_setting['ui']) or (start == name_setting['ro'] and end == name_setting['li'])\
                        or (start == name_setting['uo'] and end == name_setting['di']) or (start == name_setting['lo'] and end == name_setting['ri']):
                        turn_rad = 0.
                    elif (start == name_setting['do'] and end == name_setting['ri']) or (start == name_setting['ro'] and end == name_setting['ui'])\
                        or (start == name_setting['uo'] and end == name_setting['li']) or (start == name_setting['lo'] and end == name_setting['di']):
                        turn_rad = -1/(Para.CROSSROAD_SIZE_LON / 2 - 3.9 * Para.LANE_WIDTH_1)
                    elif start == name_setting['do'] and end == name_setting['li']:   # 'dl'
                        turn_rad = 1/sqrt((v['x']-(-Para.CROSSROAD_SIZE_LAT/2))**2 + (v['y']-(-Para.CROSSROAD_SIZE_LON/2))**2)
                    elif start == name_setting['ro'] and end == name_setting['di']:   # 'rd'
                        turn_rad = 1/sqrt((v['x']-(Para.CROSSROAD_SIZE_LAT/2))**2 + (v['y']-(-Para.CROSSROAD_SIZE_LON/2))**2)
                    elif start == name_setting['uo'] and end == name_setting['ri']:   # 'ur'
                        turn_rad = 1/sqrt((v['x']-(Para.CROSSROAD_SIZE_LAT/2))**2 + (v['y']-(Para.CROSSROAD_SIZE_LON/2))**2)
                    elif start == name_setting['lo'] and end == name_setting['ui']:   # 'lu'
                        turn_rad = 1/sqrt((v['x']-(-Para.CROSSROAD_SIZE_LAT/2))**2 + (v['y']-(Para.CROSSROAD_SIZE_LON/2))**2)
                    else:
                        turn_rad = 0.
                return turn_rad

            for v in vs:
                if v['type'] in ['bicycle_1', 'bicycle_2', 'bicycle_3']:
                    v.update(partici_type=[1., 0., 0.], turn_rad=0.0, exist=True)
                    route_list = v['route']
                    start = route_list[0]
                    end = route_list[1]

                    if start == name_setting['do'] and end == name_setting['ui']:
                        du_b.append(v)
                    elif start == name_setting['do'] and end == name_setting['ri']:
                        dr_b.append(v)

                    elif start == name_setting['ro'] and end == name_setting['li']:
                        rl_b.append(v)
                    elif start == name_setting['ro'] and end == name_setting['ui']:
                        ru_b.append(v)

                    elif start == name_setting['uo'] and end == name_setting['di']:
                        ud_b.append(v)
                    elif start == name_setting['uo'] and end == name_setting['li']:
                        ul_b.append(v)

                    elif start == name_setting['lo'] and end == name_setting['ri']:
                        lr_b.append(v)
                    elif start == name_setting['lo'] and end == name_setting['di']:
                        ld_b.append(v)

                elif v['type'] == 'DEFAULT_PEDTYPE':
                    v.update(partici_type=[0., 1., 0.], turn_rad=0.0, exist=True)
                    if (Para.OFFSET_U - (Para.LANE_NUMBER_LON_IN-1) * Para.LANE_WIDTH_2 - Para.LANE_WIDTH_3 - Para.BIKE_LANE_WIDTH <= v['x'] <= Para.OFFSET_U + Para.GREEN_BELT_LON + Para.LANE_NUMBER_LON_OUT * Para.LANE_WIDTH_1 + Para.BIKE_LANE_WIDTH) and \
                            (Para.CROSSROAD_SIZE_LON / 2 <= v['y'] <= Para.CROSSROAD_SIZE_LON / 2 + Para.WALK_WIDTH):
                        c0.append(v)
                    elif (Para.CROSSROAD_SIZE_LAT / 2 - Para.WALK_WIDTH <= v['x'] <= Para.CROSSROAD_SIZE_LAT / 2) and \
                         (Para.OFFSET_R - Para.LANE_WIDTH_1 * Para.LANE_NUMBER_LAT_OUT - Para.BIKE_LANE_WIDTH <= v[
                             'y'] <= Para.OFFSET_R + Para.GREEN_BELT_LAT + Para.LANE_WIDTH_1 * Para.LANE_NUMBER_LAT_IN + Para.BIKE_LANE_WIDTH):
                        c1.append(v)
                    elif (Para.OFFSET_D - Para.GREEN_BELT_LON - Para.LANE_NUMBER_LON_OUT * Para.LANE_WIDTH_1 - Para.BIKE_LANE_WIDTH <= v['x'] <= Para.OFFSET_D + (Para.LANE_NUMBER_LON_IN-1) * Para.LANE_WIDTH_2 + Para.LANE_WIDTH_3+ Para.BIKE_LANE_WIDTH) and \
                            (-Para.CROSSROAD_SIZE_LON / 2 <= v['y'] <= -Para.CROSSROAD_SIZE_LON / 2 + Para.WALK_WIDTH):
                        c2.append(v)
                    elif (-Para.CROSSROAD_SIZE_LAT / 2 <= v['x'] <= -Para.CROSSROAD_SIZE_LAT / 2 + Para.WALK_WIDTH) and \
                         (Para.OFFSET_L - Para.LANE_WIDTH_1 * Para.LANE_NUMBER_LAT_IN - Para.BIKE_LANE_WIDTH <= v['y'] <=
                          Para.OFFSET_L + Para.GREEN_BELT_LAT + Para.LANE_WIDTH_1 * Para.LANE_NUMBER_LAT_OUT + Para.BIKE_LANE_WIDTH):
                        c3.append(v)
                else:
                    v.update(partici_type=[0., 0., 1.], turn_rad=cal_turn_rad(v), exist=True)
                    route_list = v['route']
                    start = route_list[0]
                    end = route_list[1]
                    if start == name_setting['do'] and end == name_setting['li']:
                        dl.append(v)
                    elif start == name_setting['do'] and end == name_setting['ui']:
                        v.update(turn_rad=0.)
                        du.append(v)
                    elif start == name_setting['do'] and end == name_setting['ri']:
                        dr.append(v)

                    elif start == name_setting['ro'] and end == name_setting['di']:
                        rd.append(v)
                    elif start == name_setting['ro'] and end == name_setting['li']:
                        v.update(turn_rad=0.)
                        rl.append(v)
                    elif start == name_setting['ro'] and end == name_setting['ui']:
                        ru.append(v)

                    elif start == name_setting['uo'] and end == name_setting['ri']:
                        ur.append(v)
                    elif start == name_setting['uo'] and end == name_setting['di']:
                        v.update(turn_rad=0.)
                        ud.append(v)
                    elif start == name_setting['uo'] and end == name_setting['li']:
                        ul.append(v)

                    elif start == name_setting['lo'] and end == name_setting['ui']:
                        lu.append(v)
                    elif start == name_setting['lo'] and end == name_setting['ri']:
                        v.update(turn_rad=0.)
                        lr.append(v)
                    elif start == name_setting['lo'] and end == name_setting['di']:
                        ld.append(v)

            # fetch bicycle in range
            if task == 'straight':
                du_b = list(
                    filter(lambda v: ego_y - 2 < v['y'] < Para.CROSSROAD_SIZE_LON / 2 and v['x'] < ego_x + 8, du_b))
            elif task == 'right':
                du_b = list(filter(lambda v: ego_y - 2 < v['y'] < 0 and v['x'] < ego_x + 8, du_b))
            ud_b = list(filter(lambda v: max(ego_y - 2, -Para.CROSSROAD_SIZE_LON / 2) < v[
                'y'] < Para.CROSSROAD_SIZE_LON / 2 and ego_x > v['x'], ud_b))  # interest of left
            lr_b = list(filter(lambda v: 0 < v['x'] < Para.CROSSROAD_SIZE_LAT / 2 + 10, lr_b))  # interest of right

            # sort
            du_b = sorted(du_b, key=lambda v: v['y'])
            ud_b = sorted(ud_b, key=lambda v: v['y'])
            lr_b = sorted(lr_b, key=lambda v: -v['x'])

            mode2fillvalue_b = dict(
                du_b=dict(type="bicycle_1",
                          x=Para.OFFSET_D + Para.LANE_WIDTH_2 + Para.LANE_WIDTH_3 + Para.LANE_WIDTH_3 + Para.BIKE_LANE_WIDTH / 2,
                          y=-(Para.CROSSROAD_SIZE_LON / 2 + 30), v=0,
                          phi=90, w=0.48, l=2, route=('1o', '3i'), partici_type=[1., 0., 0.], turn_rad=0., exist=False),

                ud_b=dict(type="bicycle_1", x=Para.OFFSET_U - (
                            Para.LANE_WIDTH_2 + Para.LANE_WIDTH_3 + Para.LANE_WIDTH_3 + Para.BIKE_LANE_WIDTH / 2),
                          y=(Para.CROSSROAD_SIZE_LON / 2 + 20), v=0,
                          phi=-90, w=0.48, l=2, route=('3o', '1i'), partici_type=[1., 0., 0.], turn_rad=0.,
                          exist=False),

                lr_b=dict(type="bicycle_1", x=-(Para.CROSSROAD_SIZE_LAT / 2 + 30),
                          y=-(-Para.OFFSET_L + Para.LANE_WIDTH_1 * Para.LANE_NUMBER_LAT_IN + Para.BIKE_LANE_WIDTH / 2),
                          v=0, phi=0, w=0.48, l=2, route=('4o', '2i'), partici_type=[1., 0., 0.], turn_rad=0.,
                          exist=False))

            tmp_b = []
            if self.state_mode == 'fix':
                for mode, num in BIKE_MODE_DICT[task].items():
                    tmp_b_mode = slice_or_fill(eval(mode), mode2fillvalue_b[mode], num)
                    tmp_b.extend(tmp_b_mode)
            elif self.state_mode == 'dyna':
                for mode, num in BIKE_MODE_DICT[task].items():
                    tmp_b.extend(eval(mode))
                while len(tmp_b) < self.bike_num:
                    if self.training_task == 'left':
                        tmp_b.append(mode2fillvalue_b['ud_b'])
                    elif self.training_task == 'straight':
                        tmp_b.append(mode2fillvalue_b['du_b'])
                    else:
                        tmp_b.append(mode2fillvalue_b['du_b'])
                if len(tmp_b) > self.bike_num:
                    tmp_b = sorted(tmp_b, key=lambda v: (sqrt((v['y'] - ego_y) ** 2 + (v['x'] - ego_x) ** 2)))
                    tmp_b = tmp_b[:self.bike_num]

            # fetch person in range
            c0 = list(filter(lambda v: Para.OFFSET_U < v['x'] and v['y'] > ego_y - Para.L, c0))  # interest of straight
            c1 = list(filter(lambda v: v['y'] < Para.OFFSET_R + Para.GREEN_BELT_LAT and v['x'] > ego_x - Para.L, c1))  # interest of right
            c2 = list(filter(lambda v: Para.OFFSET_D - Para.GREEN_BELT_LON < v['x'] and v['y'] > ego_y - Para.L, c2))  # interest of right
            c3 = list(filter(lambda v: Para.OFFSET_L < v['y'] and v['x'] < ego_x + Para.L, c3))  # interest of left
            # sort
            c1 = sorted(c1, key=lambda v: (abs(v['y'] - ego_y), v['x']))
            c2 = sorted(c2, key=lambda v: (abs(v['x'] - ego_x), v['y']))
            c3 = sorted(c3, key=lambda v: (abs(v['y'] - ego_y), -v['x']))

            mode2fillvalue_p = dict(
                c1=dict(type='DEFAULT_PEDTYPE',
                        x=Para.OFFSET_D + Para.LANE_WIDTH_2 + Para.LANE_WIDTH_3 + Para.LANE_WIDTH_3 + Para.BIKE_LANE_WIDTH + Para.PERSON_LANE_WIDTH / 2,
                        y=-(Para.CROSSROAD_SIZE_LON / 2 + 30),
                        v=0, phi=90, w=0.525, l=0.75, road="0_c1", partici_type=[0., 1., 0.], turn_rad=0., exist=False),
                c2=dict(type='DEFAULT_PEDTYPE', x=-(Para.CROSSROAD_SIZE_LAT / 2 + 30), y=-(
                            -Para.OFFSET_L + Para.LANE_WIDTH_1 * Para.LANE_NUMBER_LAT_IN + Para.BIKE_LANE_WIDTH + Para.PERSON_LANE_WIDTH / 2),
                        v=0, phi=0, w=0.525, l=0.75, road="0_c2", partici_type=[0., 1., 0.], turn_rad=0., exist=False),
                c3=dict(type='DEFAULT_PEDTYPE', x=-(
                            Para.LANE_WIDTH_2 + Para.LANE_WIDTH_3 * 2 + Para.BIKE_LANE_WIDTH + Para.PERSON_LANE_WIDTH / 2) + Para.OFFSET_U,
                        y=(Para.CROSSROAD_SIZE_LON / 2 + 20),
                        v=0, phi=-90, w=0.525, l=0.75, road="0_c3", partici_type=[0., 1., 0.], turn_rad=0.,
                        exist=False))

            tmp_p = []
            if self.state_mode == 'fix':
                for mode, num in PERSON_MODE_DICT[task].items():
                    tmp_p_mode = slice_or_fill(eval(mode), mode2fillvalue_p[mode], num)
                    tmp_p.extend(tmp_p_mode)
            elif self.state_mode == 'dyna':
                for mode, num in PERSON_MODE_DICT[task].items():
                    tmp_p.extend(eval(mode))
                while len(tmp_p) < self.person_num:
                    if self.training_task == 'left':
                        tmp_p.append(mode2fillvalue_p['c3'])
                    elif self.training_task == 'straight':
                        tmp_p.append(mode2fillvalue_p['c2'])
                    else:
                        tmp_p.append(mode2fillvalue_p['c1'])
                if len(tmp_p) > self.person_num:
                    tmp_p = sorted(tmp_p, key=lambda v: (sqrt((v['y'] - ego_y) ** 2 + (v['x'] - ego_x) ** 2)))
                    tmp_p = tmp_p[:self.veh_num]

            # fetch veh in range
            dl = list(filter(lambda v: v['x'] > -Para.CROSSROAD_SIZE_LAT / 2 - 10 and v['y'] > ego_y - 2,
                             dl))  # interest of left straight
            du = list(filter(lambda v: ego_y - 2 < v['y'] < Para.CROSSROAD_SIZE_LON / 2 + 10 and v['x'] < ego_x + 5,
                             du))  # interest of left straight

            dr = list(
                filter(lambda v: v['x'] < Para.CROSSROAD_SIZE_LAT / 2 + 10 and v['y'] > ego_y, dr))  # interest of right

            rd = rd  # not interest in case of traffic light
            rl = rl  # not interest in case of traffic light
            ru = list(filter(
                lambda v: v['x'] < Para.CROSSROAD_SIZE_LAT / 2 + 10 and v['y'] < Para.CROSSROAD_SIZE_LON / 2 + 10,
                ru))  # interest of straight

            if task == 'straight':
                ur = list(filter(lambda v: v['x'] < ego_x + 7 and ego_y < v['y'] < Para.CROSSROAD_SIZE_LON / 2 + 10,
                                 ur))  # interest of straight
            elif task == 'right':
                ur = list(
                    filter(lambda v: v['x'] < Para.CROSSROAD_SIZE_LAT / 2 + 10 and v['y'] < Para.CROSSROAD_SIZE_LON / 2,
                           ur))  # interest of right
            ud = list(filter(lambda v: max(ego_y - 2, -Para.CROSSROAD_SIZE_LON / 2) < v[
                'y'] < Para.CROSSROAD_SIZE_LON / 2 and ego_x > v['x'], ud))  # interest of left
            ul = list(filter(
                lambda v: -Para.CROSSROAD_SIZE_LAT / 2 - 10 < v['x'] < ego_x and v['y'] < Para.CROSSROAD_SIZE_LON / 2,
                ul))  # interest of left

            lu = lu  # not interest in case of traffic light
            lr = list(filter(lambda v: -Para.CROSSROAD_SIZE_LAT / 2 - 10 < v['x'] < Para.CROSSROAD_SIZE_LAT / 2 + 10,
                             lr))  # interest of right
            ld = ld  # not interest in case of traffic light

            # sort
            dl = sorted(dl, key=lambda v: (v['y'], -v['x']))
            du = sorted(du, key=lambda v: v['y'])
            dr = sorted(dr, key=lambda v: (v['y'], v['x']))

            ru = sorted(ru, key=lambda v: (-v['x'], v['y']), reverse=True)

            if task == 'straight':
                ur = sorted(ur, key=lambda v: v['y'])
            elif task == 'right':
                ur = sorted(ur, key=lambda v: (-v['y'], v['x']), reverse=True)

            ud = sorted(ud, key=lambda v: v['y'])
            ul = sorted(ul, key=lambda v: (-v['y'], -v['x']), reverse=True)

            lr = sorted(lr, key=lambda v: -v['x'])

            mode2fillvalue = dict(
                dl=dict(type="car_1", x=Para.OFFSET_D + Para.LANE_WIDTH_2 / 2, y=-(Para.CROSSROAD_SIZE_LON / 2 + 30),
                        v=0, phi=90, w=2.5, l=5, route=('1o', '4i'), partici_type=[0., 0., 1.],
                        turn_rad=0., exist=False),
                du=dict(type="car_1", x=Para.OFFSET_D + Para.LANE_WIDTH_2 + Para.LANE_WIDTH_3 / 2,
                        y=-(Para.CROSSROAD_SIZE_LON / 2 + 30), v=0, phi=90, w=2.5, l=5, route=('1o', '3i'),
                        partici_type=[0., 0., 1.], turn_rad=0., exist=False),
                dr=dict(type="car_1", x=Para.OFFSET_D + Para.LANE_WIDTH_2 + Para.LANE_WIDTH_3 * 1.5,
                        y=-(Para.CROSSROAD_SIZE_LON / 2 + 30), v=0, phi=90, w=2.5, l=5, route=('1o', '2i'),
                        partici_type=[0., 0., 1.],
                        turn_rad=0., exist=False),
                ru=dict(type="car_1", x=(Para.CROSSROAD_SIZE_LAT / 2 + 30),
                        y=Para.LANE_WIDTH_1 * (Para.LANE_NUMBER_LAT_IN - 0.5) + Para.OFFSET_R + Para.GREEN_BELT_LAT,
                        v=0, phi=180, w=2.5, l=5, route=('2o', '3i'), partici_type=[0., 0., 1.],
                        turn_rad=0., exist=False),
                ur=dict(type="car_1", x=-(Para.LANE_WIDTH_2 / 2) + Para.OFFSET_U, y=(Para.CROSSROAD_SIZE_LON / 2 + 20),
                        v=0, phi=-90, w=2.5, l=5, route=('3o', '2i'), partici_type=[0., 0., 1.],
                        turn_rad=0., exist=False),
                ud=dict(type="car_1", x=-(Para.LANE_WIDTH_2 + Para.LANE_WIDTH_3 * 0.5) + Para.OFFSET_U,
                        y=(Para.CROSSROAD_SIZE_LON / 2 + 20), v=0, phi=-90, w=2.5, l=5, route=('3o', '1i'),
                        partici_type=[0., 0., 1.], turn_rad=0., exist=False),
                ul=dict(type="car_1", x=-(Para.LANE_WIDTH_2 + Para.LANE_WIDTH_3 * 1.5) + Para.OFFSET_U,
                        y=(Para.CROSSROAD_SIZE_LON / 2 + 20), v=0, phi=-90, w=2.5, l=5, route=('3o', '4i'),
                        partici_type=[0., 0., 1.],
                        turn_rad=0., exist=False),
                lr=dict(type="car_1", x=-(Para.CROSSROAD_SIZE_LAT / 2 + 30),
                        y=-(-Para.OFFSET_L + Para.LANE_WIDTH_1 * (Para.LANE_NUMBER_LAT_IN - 1.5)), v=0, phi=0, w=2.5,
                        l=5, route=('4o', '2i'), partici_type=[0., 0., 1.], turn_rad=0., exist=False))

            tmp_v = []
            if self.state_mode == 'fix':
                for mode, num in VEHICLE_MODE_DICT[task].items():
                    tmp_v_mode = slice_or_fill(eval(mode), mode2fillvalue[mode], num)
                    tmp_v.extend(tmp_v_mode)
            elif self.state_mode == 'dyna':
                for mode, num in VEHICLE_MODE_DICT[task].items():
                    tmp_v.extend(eval(mode))
                while len(tmp_v) < self.veh_num:
                    if self.training_task == 'left':
                        tmp_v.append(mode2fillvalue['dl'])
                    elif self.training_task == 'straight':
                        tmp_v.append(mode2fillvalue['du'])
                    else:
                        tmp_v.append(mode2fillvalue['dr'])
                if len(tmp_v) > self.veh_num:
                    tmp_v = sorted(tmp_v, key=lambda v: (sqrt((v['y'] - ego_y) ** 2 + (v['x'] - ego_x) ** 2), -v['x']))
                    tmp_v = tmp_v[:self.veh_num]
            tmp = tmp_b + tmp_p + tmp_v
            return tmp

        self.interested_other = filter_interested_other(self.all_other, self.training_task)

        for other in self.interested_other:
            other_x, other_y, other_v, other_phi, other_l, other_w, other_type, other_turn_rad, other_mask = \
                other['x'], other['y'], other['v'], other['phi'], other['l'], other['w'], other['partici_type'], other[
                    'turn_rad'], other['exist']
            other_vector.extend(
                [other_x, other_y, other_v, other_phi, other_l, other_w] + other_type + [other_turn_rad])
            other_mask_vector.append(other_mask)

        return np.array(other_vector, dtype=np.float32), np.array(other_mask_vector, dtype=np.float32)

    def _reset_init_state(self):
        if self.traffic_mode == 'auto':
            if self.training_task == 'left':
                random_index = int(np.random.random() * (900 + 500)) + 700
            elif self.training_task == 'straight':
                random_index = int(np.random.random() * (1200 + 500)) + 700
            else:
                random_index = int(np.random.random() * (420 + 500)) + 700
        else:
            random_index = MODE2INDEX[self.traffic_mode] + int(np.random.random() * 100)

        x, y, phi, exp_v = self.ref_path.idx2point(random_index)
        v = exp_v * np.random.random()
        routeID = TASK2ROUTEID[self.training_task]
        return dict(ego=dict(v_x=v,
                             v_y=0,
                             r=0,
                             x=x,
                             y=y,
                             phi=phi,
                             l=self.ego_l,
                             w=self.ego_w,
                             routeID=routeID,
                             ))

    def _compute_reward(self, obs, action, untransformed_action):
        obses, actions, untransformed_actions = obs[np.newaxis, :], action[np.newaxis, :], untransformed_action[np.newaxis, :]
        reward, _, _, _, _, _, _, reward_dict = self.env_model.compute_rewards(obses, actions, untransformed_actions)
        for k, v in reward_dict.items():
            reward_dict[k] = v.numpy()[0]
        return reward.numpy()[0], reward_dict

    def render(self, mode='human', weights=None):
        if mode == 'human':
            # plot basic map
            extension = 40
            dotted_line_style = '--'
            solid_line_style = '-'

            plt.cla()
            ax = plt.axes([-0.05, -0.05, 1.1, 1.1])
            ax.axis("equal")
            patches = []
            # ax.add_patch(plt.Rectangle((-square_length / 2 - extension, -square_length / 2 - extension),
            #                            square_length + 2 * extension, square_length + 2 * extension, edgecolor='black',
            #                            facecolor='none', linewidth=2))

            # ----------arrow--------------
            # plt.arrow(lane_width / 2, -square_length / 2 - 10, 0, 3, color='darkviolet')
            # plt.arrow(lane_width / 2, -square_length / 2 - 10 + 3, -0.5, 1.0, color='darkviolet', head_width=0.7)
            # plt.arrow(lane_width * 1.5, -square_length / 2 - 10, 0, 4, color='darkviolet', head_width=0.7)
            # plt.arrow(lane_width * 2.5, -square_length / 2 - 10, 0, 3, color='darkviolet')
            # plt.arrow(lane_width * 2.5, -square_length / 2 - 10 + 3, 0.5, 1.0, color='darkviolet', head_width=0.7)
            # ----------green belt--------------
            patches.append(plt.Rectangle((-Para.CROSSROAD_SIZE_LAT / 2 - extension, Para.OFFSET_L),
                                       extension, Para.GREEN_BELT_LAT, edgecolor='black', facecolor='green',
                                       linewidth=1))
            patches.append(plt.Rectangle((-Para.CROSSROAD_SIZE_LAT / 2 - extension, Para.OFFSET_L),
                                       extension, Para.GREEN_BELT_LAT, edgecolor='black', facecolor='green',
                                       linewidth=1))
            patches.append(plt.Rectangle((Para.CROSSROAD_SIZE_LAT / 2, Para.OFFSET_R),
                                       extension, Para.GREEN_BELT_LAT, edgecolor='black', facecolor='green',
                                       linewidth=1))
            patches.append(plt.Rectangle((Para.OFFSET_U, Para.CROSSROAD_SIZE_LON / 2),
                                       Para.GREEN_BELT_LON, extension, edgecolor='black', facecolor='green',
                                       linewidth=1))
            patches.append(plt.Rectangle((Para.OFFSET_D - Para.GREEN_BELT_LON, -Para.CROSSROAD_SIZE_LON / 2 - extension),
                                       Para.GREEN_BELT_LON, extension, edgecolor='black', facecolor='green',
                                       linewidth=1))

            # Left out lane
            for i in range(1, Para.LANE_NUMBER_LAT_OUT + 2):
                lane_width_flag = [Para.LANE_WIDTH_1, Para.LANE_WIDTH_1, Para.LANE_WIDTH_1,
                                   Para.PERSON_LANE_WIDTH + Para.BIKE_LANE_WIDTH]
                base = Para.OFFSET_L + Para.GREEN_BELT_LAT
                linestyle = dotted_line_style if i < Para.LANE_NUMBER_LAT_OUT else solid_line_style
                linewidth = 1 if i < Para.LANE_NUMBER_LAT_OUT else 1
                plt.plot([-Para.CROSSROAD_SIZE_LAT / 2 - extension, -Para.CROSSROAD_SIZE_LAT / 2],
                         [base + sum(lane_width_flag[:i]), base + sum(lane_width_flag[:i])],
                         linestyle=linestyle, color='black', linewidth=linewidth)
            # Left in lane
            for i in range(1, Para.LANE_NUMBER_LAT_IN + 2):
                lane_width_flag = [Para.LANE_WIDTH_1, Para.LANE_WIDTH_1, Para.LANE_WIDTH_1, Para.LANE_WIDTH_1,
                                   Para.PERSON_LANE_WIDTH + Para.BIKE_LANE_WIDTH]
                base = Para.OFFSET_L
                linestyle = dotted_line_style if i < Para.LANE_NUMBER_LAT_IN else solid_line_style
                linewidth = 1 if i < Para.LANE_NUMBER_LAT_IN else 1
                plt.plot([-Para.CROSSROAD_SIZE_LAT / 2 - extension, -Para.CROSSROAD_SIZE_LAT / 2],
                         [base - sum(lane_width_flag[:i]), base - sum(lane_width_flag[:i])],
                         linestyle=linestyle, color='black', linewidth=linewidth)

            # Right out lane
            for i in range(1, Para.LANE_NUMBER_LAT_OUT + 2):
                lane_width_flag = [Para.LANE_WIDTH_1, Para.LANE_WIDTH_1, Para.LANE_WIDTH_1,
                                   Para.PERSON_LANE_WIDTH + Para.BIKE_LANE_WIDTH]
                base = Para.OFFSET_R
                linestyle = dotted_line_style if i < Para.LANE_NUMBER_LAT_OUT else solid_line_style
                linewidth = 1 if i < Para.LANE_NUMBER_LAT_OUT else 1
                plt.plot([Para.CROSSROAD_SIZE_LAT / 2, Para.CROSSROAD_SIZE_LAT / 2 + extension],
                         [base - sum(lane_width_flag[:i]), base - sum(lane_width_flag[:i])],
                         linestyle=linestyle, color='black', linewidth=linewidth)

            # Right in lane
            for i in range(1, Para.LANE_NUMBER_LAT_IN + 2):
                lane_width_flag = [Para.LANE_WIDTH_1, Para.LANE_WIDTH_1, Para.LANE_WIDTH_1, Para.LANE_WIDTH_1,
                                   Para.PERSON_LANE_WIDTH + Para.BIKE_LANE_WIDTH]
                base = Para.OFFSET_R + Para.GREEN_BELT_LAT
                linestyle = dotted_line_style if i < Para.LANE_NUMBER_LAT_IN else solid_line_style
                linewidth = 1 if i < Para.LANE_NUMBER_LAT_IN else 1
                plt.plot([Para.CROSSROAD_SIZE_LAT / 2, Para.CROSSROAD_SIZE_LAT / 2 + extension],
                         [base + sum(lane_width_flag[:i]), base + sum(lane_width_flag[:i])],
                         linestyle=linestyle, color='black', linewidth=linewidth)

            # Up in lane
            for i in range(1, Para.LANE_NUMBER_LON_IN + 2):
                lane_width_flag = [Para.LANE_WIDTH_2, Para.LANE_WIDTH_3, Para.LANE_WIDTH_3,
                                   Para.PERSON_LANE_WIDTH + Para.BIKE_LANE_WIDTH]
                base = Para.OFFSET_U
                linestyle = dotted_line_style if i < Para.LANE_NUMBER_LON_IN else solid_line_style
                linewidth = 1 if i < Para.LANE_NUMBER_LON_IN else 1
                plt.plot([base - sum(lane_width_flag[:i]), base - sum(lane_width_flag[:i])],
                         [Para.CROSSROAD_SIZE_LON / 2 + extension, Para.CROSSROAD_SIZE_LON / 2],
                         linestyle=linestyle, color='black', linewidth=linewidth)

            # Up out lane
            for i in range(1, Para.LANE_NUMBER_LON_OUT + 2):
                lane_width_flag = [Para.LANE_WIDTH_3, Para.LANE_WIDTH_3, Para.PERSON_LANE_WIDTH + Para.BIKE_LANE_WIDTH]
                base = Para.OFFSET_U + Para.GREEN_BELT_LON
                linestyle = dotted_line_style if i < Para.LANE_NUMBER_LON_OUT else solid_line_style
                linewidth = 1 if i < Para.LANE_NUMBER_LON_OUT else 1
                plt.plot([base + sum(lane_width_flag[:i]), base + sum(lane_width_flag[:i])],
                         [Para.CROSSROAD_SIZE_LON / 2 + extension, Para.CROSSROAD_SIZE_LON / 2],
                         linestyle=linestyle, color='black', linewidth=linewidth)

            # Down in lane
            for i in range(1, Para.LANE_NUMBER_LON_IN + 2):
                lane_width_flag = [Para.LANE_WIDTH_2, Para.LANE_WIDTH_3, Para.LANE_WIDTH_3,
                                   Para.PERSON_LANE_WIDTH + Para.BIKE_LANE_WIDTH]
                base = Para.OFFSET_D
                linestyle = dotted_line_style if i < Para.LANE_NUMBER_LON_IN else solid_line_style
                linewidth = 1 if i < Para.LANE_NUMBER_LON_IN else 1
                plt.plot([base + sum(lane_width_flag[:i]), base + sum(lane_width_flag[:i])],
                         [-Para.CROSSROAD_SIZE_LON / 2 - extension, -Para.CROSSROAD_SIZE_LON / 2],
                         linestyle=linestyle, color='black', linewidth=linewidth)

            # Down out lane
            for i in range(1, Para.LANE_NUMBER_LON_OUT + 2):
                lane_width_flag = [Para.LANE_WIDTH_3, Para.LANE_WIDTH_3, Para.PERSON_LANE_WIDTH + Para.BIKE_LANE_WIDTH]
                base = Para.OFFSET_D - Para.GREEN_BELT_LON
                linestyle = dotted_line_style if i < Para.LANE_NUMBER_LON_OUT else solid_line_style
                linewidth = 1 if i < Para.LANE_NUMBER_LON_OUT else 1
                plt.plot([base - sum(lane_width_flag[:i]), base - sum(lane_width_flag[:i])],
                         [-Para.CROSSROAD_SIZE_LON / 2 - extension, -Para.CROSSROAD_SIZE_LON / 2],
                         linestyle=linestyle, color='black', linewidth=linewidth)

            # Oblique
            plt.plot([-Para.CROSSROAD_SIZE_LAT / 2, Para.OFFSET_U - (
                        Para.LANE_NUMBER_LON_IN - 1) * Para.LANE_WIDTH_3 - Para.LANE_WIDTH_2 - Para.BIKE_LANE_WIDTH - Para.PERSON_LANE_WIDTH],
                     [
                         Para.OFFSET_L + Para.GREEN_BELT_LAT + Para.LANE_NUMBER_LAT_OUT * Para.LANE_WIDTH_1 + Para.BIKE_LANE_WIDTH + Para.PERSON_LANE_WIDTH,
                         Para.CROSSROAD_SIZE_LON / 2],
                     color='black', linewidth=1)
            plt.plot([-Para.CROSSROAD_SIZE_LAT / 2,
                      Para.OFFSET_D - Para.GREEN_BELT_LON - Para.LANE_NUMBER_LON_OUT * Para.LANE_WIDTH_3 - Para.BIKE_LANE_WIDTH - Para.PERSON_LANE_WIDTH],
                     [
                         Para.OFFSET_L - Para.LANE_NUMBER_LAT_IN * Para.LANE_WIDTH_1 - Para.BIKE_LANE_WIDTH - Para.PERSON_LANE_WIDTH,
                         -Para.CROSSROAD_SIZE_LON / 2],
                     color='black', linewidth=1)
            plt.plot([Para.CROSSROAD_SIZE_LAT / 2,
                      Para.OFFSET_D + (
                                  Para.LANE_NUMBER_LON_IN - 1) * Para.LANE_WIDTH_3 + Para.LANE_WIDTH_2 + Para.BIKE_LANE_WIDTH + Para.PERSON_LANE_WIDTH],
                     [
                         Para.OFFSET_R - Para.LANE_NUMBER_LAT_OUT * Para.LANE_WIDTH_1 - Para.BIKE_LANE_WIDTH - Para.PERSON_LANE_WIDTH,
                         -Para.CROSSROAD_SIZE_LON / 2],
                     color='black', linewidth=1)
            plt.plot([Para.CROSSROAD_SIZE_LAT / 2,
                      Para.OFFSET_U + Para.GREEN_BELT_LON + Para.LANE_NUMBER_LON_OUT * Para.LANE_WIDTH_3 + Para.BIKE_LANE_WIDTH + Para.PERSON_LANE_WIDTH],
                     [
                         Para.OFFSET_R + Para.GREEN_BELT_LAT + Para.LANE_NUMBER_LAT_IN * Para.LANE_WIDTH_1 + Para.BIKE_LANE_WIDTH + Para.PERSON_LANE_WIDTH,
                         Para.CROSSROAD_SIZE_LON / 2],
                     color='black', linewidth=1)

            # stop line
            lane_width_flag = [Para.LANE_WIDTH_2, Para.LANE_WIDTH_3, Para.LANE_WIDTH_3,
                               Para.PERSON_LANE_WIDTH + Para.BIKE_LANE_WIDTH]  # Down
            plt.plot([Para.OFFSET_D, Para.OFFSET_D + sum(lane_width_flag[:Para.LANE_NUMBER_LON_IN])],
                     [-Para.CROSSROAD_SIZE_LON / 2, -Para.CROSSROAD_SIZE_LON / 2], color='black')
            lane_width_flag = [Para.LANE_WIDTH_2, Para.LANE_WIDTH_3, Para.LANE_WIDTH_3,
                               Para.PERSON_LANE_WIDTH + Para.BIKE_LANE_WIDTH]  # Up
            plt.plot([-sum(lane_width_flag[:Para.LANE_NUMBER_LON_IN]) + Para.OFFSET_U, Para.OFFSET_U],
                     [Para.CROSSROAD_SIZE_LON / 2, Para.CROSSROAD_SIZE_LON / 2], color='black')
            lane_width_flag = [Para.LANE_WIDTH_1, Para.LANE_WIDTH_1, Para.LANE_WIDTH_1, Para.LANE_WIDTH_1,
                               Para.PERSON_LANE_WIDTH + Para.BIKE_LANE_WIDTH]
            plt.plot([-Para.CROSSROAD_SIZE_LAT / 2, -Para.CROSSROAD_SIZE_LAT / 2],
                     [Para.OFFSET_L, Para.OFFSET_L - sum(lane_width_flag[:Para.LANE_NUMBER_LAT_IN])],
                     color='black')  # left
            lane_width_flag = [Para.LANE_WIDTH_1, Para.LANE_WIDTH_1, Para.LANE_WIDTH_1, Para.LANE_WIDTH_1,
                               Para.PERSON_LANE_WIDTH + Para.BIKE_LANE_WIDTH]
            plt.plot([Para.CROSSROAD_SIZE_LAT / 2, Para.CROSSROAD_SIZE_LAT / 2], [Para.OFFSET_R + Para.GREEN_BELT_LAT,
                                                                                  Para.OFFSET_R + Para.GREEN_BELT_LAT + sum(
                                                                                      lane_width_flag[
                                                                                      :Para.LANE_NUMBER_LAT_IN])],
                     color='black')

            # traffic light
            v_light = self.light_phase
            light_line_width = 2
            if v_light == 0 or v_light == 1:
                v_color_1, v_color_2, h_color_1, h_color_2 = 'green', 'green', 'red', 'red'
            elif v_light == 2:
                v_color_1, v_color_2, h_color_1, h_color_2 = 'orange', 'orange', 'red', 'red'
            elif v_light == 3:
                v_color_1, v_color_2, h_color_1, h_color_2 = 'red', 'red', 'red', 'green'
            elif v_light == 4:
                v_color_1, v_color_2, h_color_1, h_color_2 = 'red', 'red', 'red', 'orange'
            elif v_light == 5:
                v_color_1, v_color_2, h_color_1, h_color_2 = 'red', 'red', 'green', 'red'
            else:
                v_color_1, v_color_2, h_color_1, h_color_2 = 'red', 'red', 'orange', 'red'

            lane_width_flag = [Para.LANE_WIDTH_2, Para.LANE_WIDTH_3, Para.LANE_WIDTH_3,
                               Para.PERSON_LANE_WIDTH + Para.BIKE_LANE_WIDTH]  # Down
            plt.plot([Para.OFFSET_D, Para.OFFSET_D + sum(lane_width_flag[:1])],
                     [-Para.CROSSROAD_SIZE_LON / 2, -Para.CROSSROAD_SIZE_LON / 2],
                     color=v_color_1, linewidth=light_line_width)
            plt.plot([Para.OFFSET_D + sum(lane_width_flag[:1]), Para.OFFSET_D + sum(lane_width_flag[:2])],
                     [-Para.CROSSROAD_SIZE_LON / 2, -Para.CROSSROAD_SIZE_LON / 2],
                     color=v_color_2, linewidth=light_line_width)
            plt.plot([Para.OFFSET_D + sum(lane_width_flag[:2]), Para.OFFSET_D + sum(lane_width_flag[:3])],
                     [-Para.CROSSROAD_SIZE_LON / 2, -Para.CROSSROAD_SIZE_LON / 2],
                     color='green', linewidth=light_line_width)

            lane_width_flag = [Para.LANE_WIDTH_2, Para.LANE_WIDTH_3, Para.LANE_WIDTH_3,
                               Para.PERSON_LANE_WIDTH + Para.BIKE_LANE_WIDTH]  # Up
            plt.plot([-sum(lane_width_flag[:1]) + Para.OFFSET_U, Para.OFFSET_U],
                     [Para.CROSSROAD_SIZE_LON / 2, Para.CROSSROAD_SIZE_LON / 2],
                     color=v_color_1, linewidth=light_line_width)
            plt.plot([-sum(lane_width_flag[:2]) + Para.OFFSET_U, -sum(lane_width_flag[:1]) + Para.OFFSET_U],
                     [Para.CROSSROAD_SIZE_LON / 2, Para.CROSSROAD_SIZE_LON / 2],
                     color=v_color_2, linewidth=light_line_width)
            plt.plot([-sum(lane_width_flag[:3]) + Para.OFFSET_U, -sum(lane_width_flag[:2]) + Para.OFFSET_U],
                     [Para.CROSSROAD_SIZE_LON / 2, Para.CROSSROAD_SIZE_LON / 2],
                     color='green', linewidth=light_line_width)

            lane_width_flag = [Para.LANE_WIDTH_1, Para.LANE_WIDTH_1, Para.LANE_WIDTH_1, Para.LANE_WIDTH_1,
                               Para.PERSON_LANE_WIDTH + Para.BIKE_LANE_WIDTH]  # left
            plt.plot([-Para.CROSSROAD_SIZE_LAT / 2, -Para.CROSSROAD_SIZE_LAT / 2],
                     [Para.OFFSET_L, Para.OFFSET_L - sum(lane_width_flag[:1])],
                     color=h_color_1, linewidth=light_line_width)
            plt.plot([-Para.CROSSROAD_SIZE_LAT / 2, -Para.CROSSROAD_SIZE_LAT / 2],
                     [Para.OFFSET_L - sum(lane_width_flag[:1]), Para.OFFSET_L - sum(lane_width_flag[:3])],
                     color=h_color_2, linewidth=light_line_width)
            plt.plot([-Para.CROSSROAD_SIZE_LAT / 2, -Para.CROSSROAD_SIZE_LAT / 2],
                     [Para.OFFSET_L - sum(lane_width_flag[:3]), Para.OFFSET_L - sum(lane_width_flag[:4])],
                     color='green', linewidth=light_line_width)

            lane_width_flag = [Para.LANE_WIDTH_1, Para.LANE_WIDTH_1, Para.LANE_WIDTH_1, Para.LANE_WIDTH_1,
                               Para.PERSON_LANE_WIDTH + Para.BIKE_LANE_WIDTH]  # right
            plt.plot([Para.CROSSROAD_SIZE_LAT / 2, Para.CROSSROAD_SIZE_LAT / 2],
                     [Para.OFFSET_R + Para.GREEN_BELT_LAT,
                      Para.OFFSET_R + Para.GREEN_BELT_LAT + sum(lane_width_flag[:1])],
                     color=h_color_1, linewidth=light_line_width)
            plt.plot([Para.CROSSROAD_SIZE_LAT / 2, Para.CROSSROAD_SIZE_LAT / 2],
                     [Para.OFFSET_R + Para.GREEN_BELT_LAT + sum(lane_width_flag[:1]),
                      Para.OFFSET_R + Para.GREEN_BELT_LAT + sum(lane_width_flag[:3])],
                     color=h_color_2, linewidth=light_line_width)
            plt.plot([Para.CROSSROAD_SIZE_LAT / 2, Para.CROSSROAD_SIZE_LAT / 2],
                     [Para.OFFSET_R + Para.GREEN_BELT_LAT + sum(lane_width_flag[:3]),
                      Para.OFFSET_R + Para.GREEN_BELT_LAT + sum(lane_width_flag[:4])],
                     color='green', linewidth=light_line_width)

            # zebra crossing
            # j1, j2 = 20.5, 6.75
            # for ii in range(19):
            #     if ii <= 3:
            #         continue
            #     ax.add_patch(
            #         plt.Rectangle((-Para.CROSSROAD_SIZE_LON / 2 + j1 + ii * 1.6, -Para.CROSSROAD_SIZE_LON / 2 + 0.5),
            #                       0.8, 4,
            #                       color='lightgray', alpha=0.5))
            #     ii += 1
            # for ii in range(19):
            #     if ii <= 3:
            #         continue
            #     ax.add_patch(
            #         plt.Rectangle((-Para.CROSSROAD_SIZE_LON / 2 + j1 + ii * 1.6, Para.CROSSROAD_SIZE_LON / 2 - 0.5 - 4),
            #                       0.8, 4,
            #                       color='lightgray', alpha=0.5))
            #     ii += 1
            # for ii in range(28):
            #     if ii <= 3:
            #         continue
            #     ax.add_patch(
            #         plt.Rectangle(
            #             (-Para.CROSSROAD_SIZE_LAT / 2 + 0.5, Para.CROSSROAD_SIZE_LAT / 2 - j2 - 0.8 - ii * 1.6), 4, 0.8,
            #             color='lightgray',
            #             alpha=0.5))
            #     ii += 1
            # for ii in range(28):
            #     if ii <= 3:
            #         continue
            #     ax.add_patch(
            #         plt.Rectangle(
            #             (Para.CROSSROAD_SIZE_LAT / 2 - 0.5 - 4, Para.CROSSROAD_SIZE_LAT / 2 - j2 - 0.8 - ii * 1.6), 4,
            #             0.8,
            #             color='lightgray',
            #             alpha=0.5))
            #     ii += 1

            def is_in_plot_area(x, y, tolerance=5):
                if -Para.CROSSROAD_SIZE_LAT / 2 - extension + tolerance < x < Para.CROSSROAD_SIZE_LAT / 2 + extension - tolerance and \
                        -Para.CROSSROAD_SIZE_LON / 2 - extension + tolerance < y < Para.CROSSROAD_SIZE_LON / 2 + extension - tolerance:
                    return True
                else:
                    return False

            def draw_rotate_rec(type, x, y, a, l, w, color, linestyle='-', patch=False):
                RU_x, RU_y, _ = rotate_coordination(l / 2, w / 2, 0, -a)
                RD_x, RD_y, _ = rotate_coordination(l / 2, -w / 2, 0, -a)
                LU_x, LU_y, _ = rotate_coordination(-l / 2, w / 2, 0, -a)
                LD_x, LD_y, _ = rotate_coordination(-l / 2, -w / 2, 0, -a)
                if patch:
                    if type in ['bicycle_1', 'bicycle_2', 'bicycle_3']:
                        item_color = 'purple'
                    elif type == 'DEFAULT_PEDTYPE':
                        item_color = 'lime'
                    else:
                        item_color = 'lightgray'
                    patches.append(plt.Rectangle((x + LU_x, y + LU_y), w, l, edgecolor=item_color,facecolor=item_color,
                                               angle=-(90 - a), zorder=30))
                else:
                    patches.append(matplotlib.patches.Rectangle(np.array([-l/2+x, -w/2+y]),
                                                                width=l, height=w,
                                                                fill=False,
                                                                facecolor=None,
                                                                edgecolor=color,
                                                                linestyle=linestyle,
                                                                linewidth=1.0,
                                                                transform=Affine2D().rotate_deg_around(*(x, y),
                                                                                                       a)))

            def draw_rotate_batch_rec(x, y, a, l, w):
                for i in range(len(x)):
                    patches.append(matplotlib.patches.Rectangle(np.array([-l[i]/2+x[i], -w[i]/2+y[i]]),
                                                                width=l[i], height=w[i],
                                                                fill=False,
                                                                facecolor=None,
                                                                edgecolor='k',
                                                                linewidth=1.0,
                                                                transform=Affine2D().rotate_deg_around(*(x[i], y[i]),
                                                                                                       a[i])))


            def plot_phi_line(type, x, y, phi, color):
                if type in ['bicycle_1', 'bicycle_2', 'bicycle_3']:
                    line_length = 2
                elif type == 'DEFAULT_PEDTYPE':
                    line_length = 1
                else:
                    line_length = 5
                x_forw, y_forw = x + line_length * cos(phi * pi / 180.), \
                                 y + line_length * sin(phi * pi / 180.)
                plt.plot([x, x_forw], [y, y_forw], color=color, linewidth=0.5)

            def get_partici_type_str(partici_type):
                if partici_type[0] == 1.:
                    return 'bike'
                elif partici_type[1] == 1.:
                    return 'person'
                elif partici_type[2] == 1.:
                    return 'veh'

            # plot others
            filted_all_other = [item for item in self.all_other if is_in_plot_area(item['x'], item['y'])]
            other_xs = np.array([item['x'] for item in filted_all_other], np.float32)
            other_ys = np.array([item['y'] for item in filted_all_other], np.float32)
            other_as = np.array([item['phi'] for item in filted_all_other], np.float32)
            other_ls = np.array([item['l'] for item in filted_all_other], np.float32)
            other_ws = np.array([item['w'] for item in filted_all_other], np.float32)

            draw_rotate_batch_rec(other_xs, other_ys, other_as, other_ls, other_ws)

            # plot interested others
            if weights is not None:
                assert weights.shape == (self.other_number,), print(weights.shape)
            index_top_k_in_weights = weights.argsort()[-4:][::-1]

            # real locomotion of interested vehicles
            for i in range(len(self.interested_other)):
                item = self.interested_other[i]
                item_mask = item['exist']
                item_x = item['x']
                item_y = item['y']
                item_phi = item['phi']
                item_l = item['l']
                item_w = item['w']
                item_type = item['type']
                if is_in_plot_area(item_x, item_y):
                    plot_phi_line(item_type, item_x, item_y, item_phi, 'black')
                    draw_rotate_rec(item_type, item_x, item_y, item_phi, item_l, item_w, color='g', linestyle=':', patch=True)
                    plt.text(item_x, item_y, str(item_mask)[0])
                if i in index_top_k_in_weights:
                    plt.text(item_x, item_y, "{:.2f}".format(weights[i]), color='red', fontsize=15)

            # plot own car
            abso_obs = self._convert_to_abso(self.obs)
            obs_ego, obs_track, obs_light, obs_task, obs_ref, obs_his_ac, obs_other = self._split_all(abso_obs)
            noised_ego_v_x, noised_ego_v_y, noised_ego_r, \
                noised_ego_x, noised_ego_y, noised_ego_phi = obs_ego
            devi_longi, devi_lateral, devi_phi, devi_v = obs_track

            real_ego_x = self.ego_dynamics['x']
            real_ego_y = self.ego_dynamics['y']
            real_ego_phi = self.ego_dynamics['phi']
            real_ego_v_x = self.ego_dynamics['v_x']
            real_ego_v_y = self.ego_dynamics['v_y']
            real_ego_r = self.ego_dynamics['r']

            # render noised objects in enabled
            if self.vector_noise:
                # noised locomotion of interested vehicles
                for i in range(len(self.interested_other)):
                    item = obs_other[self.per_other_info_dim * i : self.per_other_info_dim * (i+1)]
                    item_x = item[0]
                    item_y = item[1]
                    item_phi = item[3]
                    item_l = item[4]
                    item_w = item[5]
                    item_type = get_partici_type_str(item[-3:])
                    if is_in_plot_area(item_x, item_y):
                        plot_phi_line(item_type, item_x, item_y, item_phi, 'green')
                        draw_rotate_rec(item_type, item_x, item_y, item_phi, item_l, item_w, color='green')

                # noised ego car
                plot_phi_line('self_noised_car', noised_ego_x, noised_ego_y, noised_ego_phi, 'aquamarine')
                draw_rotate_rec('self_noised_car', noised_ego_x, noised_ego_y, noised_ego_phi, self.ego_l, self.ego_w, 'aquamarine')

            plot_phi_line('self_car', real_ego_x, real_ego_y, real_ego_phi, 'red')
            draw_rotate_rec('self_car', real_ego_x, real_ego_y, real_ego_phi, self.ego_l, self.ego_w, 'red')

            ax.plot(self.ref_path.path[0], self.ref_path.path[1], color='g')
            _, point = self.ref_path._find_closest_point(noised_ego_x, noised_ego_y)
            path_x, path_y, path_phi, path_v = point[0], point[1], point[2], point[3]
            plt.plot(path_x, path_y, 'g.')
            plt.plot(self.future_n_point[0], self.future_n_point[1], 'g.')

            # from matplotlib.patches import Circle, Ellipse
            # cir_right = Circle(xy=(Para.CROSSROAD_SIZE_LAT/2, -Para.CROSSROAD_SIZE_LON/2), radius=Para.CROSSROAD_SIZE_LON / 2 - 3.9 * Para.LANE_WIDTH_1, alpha=0.5)
            # ax.add_patch(cir_right)
            # cir_right = Circle(xy=(-Para.CROSSROAD_SIZE_LAT/2, -Para.CROSSROAD_SIZE_LON/2), radius=Para.CROSSROAD_SIZE_LAT / 2 + Para.OFFSET_D + 0.5 * Para.LANE_WIDTH_2, alpha=0.5)
            # ax.add_patch(cir_right)
            # cir_right = Circle(xy=(-Para.CROSSROAD_SIZE_LAT/2, -Para.CROSSROAD_SIZE_LON/2), radius=Para.CROSSROAD_SIZE_LON / 2 + Para.OFFSET_L + Para.GREEN_BELT_LAT + 0.5 * Para.LANE_WIDTH_1, alpha=0.5)
            # ax.add_patch(cir_right)
            # cir_right = Circle(xy=(-Para.CROSSROAD_SIZE_LAT/2, -Para.CROSSROAD_SIZE_LON/2), radius=0.5*(Para.CROSSROAD_SIZE_LON / 2 + Para.OFFSET_L + Para.GREEN_BELT_LAT + 0.5 * Para.LANE_WIDTH_1)+0.5*(Para.CROSSROAD_SIZE_LAT / 2 + Para.OFFSET_D + 0.5 * Para.LANE_WIDTH_2), alpha=0.5)
            # ax.add_patch(cir_right)
            # e = Ellipse(xy=(-Para.CROSSROAD_SIZE_LAT/2, -Para.CROSSROAD_SIZE_LON/2), width=(Para.CROSSROAD_SIZE_LAT / 2 + Para.OFFSET_D + 0.5 * Para.LANE_WIDTH_2) * 2, height=(Para.CROSSROAD_SIZE_LON / 2 + Para.OFFSET_L + Para.GREEN_BELT_LAT + 0.5 * Para.LANE_WIDTH_1) * 2, angle=0.)
            # ax.add_patch(e)
            # e = Ellipse(xy=(Para.CROSSROAD_SIZE_LAT/2, -Para.CROSSROAD_SIZE_LON/2), width=(Para.CROSSROAD_SIZE_LAT / 2 + Para.OFFSET_D + 0.5 * Para.LANE_WIDTH_2) * 2, height=(Para.CROSSROAD_SIZE_LON / 2 + Para.OFFSET_L + Para.GREEN_BELT_LAT + 0.5 * Para.LANE_WIDTH_1) * 2, angle=0.)
            # ax.add_patch(e)
            # plot real time traj
            color = ['blue', 'coral', 'darkcyan', 'pink']
            for i, item in enumerate(self.ref_path.path_list['green']):
                if REF_ENCODING[i] == self.ref_path.ref_encoding:
                    plt.plot(item[0], item[1], color=color[i], alpha=1.0)
                else:
                    plt.plot(item[0], item[1], color=color[i], alpha=0.3)
                    # indexs, points = item.find_closest_point(np.array([ego_x], np.float32), np.array([ego_y], np.float32))
                    # path_x, path_y, path_phi = points[0][0], points[1][0], points[2][0]
                    # plt.plot(path_x, path_y,  color=color[i])

            # text
            text_x, text_y_start = -110, 60
            ge = iter(range(0, 1000, 4))
            plt.text(text_x, text_y_start - next(ge), 'ego_x: {:.2f}m'.format(real_ego_x))
            plt.text(text_x, text_y_start - next(ge), 'ego_y: {:.2f}m'.format(real_ego_y))
            plt.text(text_x, text_y_start - next(ge), 'path_x: {:.2f}m'.format(path_x))
            plt.text(text_x, text_y_start - next(ge), 'path_y: {:.2f}m'.format(path_y))
            plt.text(text_x, text_y_start - next(ge), 'devi_longi: {:.2f}m'.format(devi_longi))
            plt.text(text_x, text_y_start - next(ge), 'devi_lateral: {:.2f}m'.format(devi_lateral))
            plt.text(text_x, text_y_start - next(ge), 'devi_v: {:.2f}m/s'.format(devi_v))
            plt.text(text_x, text_y_start - next(ge), r'ego_phi: ${:.2f}\degree$'.format(real_ego_phi))
            plt.text(text_x, text_y_start - next(ge), r'path_phi: ${:.2f}\degree$'.format(path_phi))
            plt.text(text_x, text_y_start - next(ge), r'devi_phi: ${:.2f}\degree$'.format(devi_phi))

            plt.text(text_x, text_y_start - next(ge), 'v_x: {:.2f}m/s'.format(real_ego_v_x))
            plt.text(text_x, text_y_start - next(ge), 'exp_v: {:.2f}m/s'.format(path_v))
            plt.text(text_x, text_y_start - next(ge), 'v_y: {:.2f}m/s'.format(real_ego_v_y))
            plt.text(text_x, text_y_start - next(ge), 'yaw_rate: {:.2f}rad/s'.format(real_ego_r))

            if self.action is not None:
                steer, a_x = self.action[0], self.action[1]
                plt.text(text_x, text_y_start - next(ge),
                         r'steer: {:.2f}rad (${:.2f}\degree$)'.format(steer, steer * 180 / np.pi))
                plt.text(text_x, text_y_start - next(ge), 'a_x: {:.2f}m/s^2'.format(a_x))

            text_x, text_y_start = 80, 60
            ge = iter(range(0, 1000, 4))

            # done info
            plt.text(text_x, text_y_start - next(ge), 'done info: {}'.format(self.done_type))

            # reward info
            if self.reward_info is not None:
                for key, val in self.reward_info.items():
                    plt.text(text_x, text_y_start - next(ge), 'rew_{}: {:.4f}'.format(key, val))

            # indicator for trajectory selection
            # text_x, text_y_start = -25, -65
            # ge = iter(range(0, 1000, 6))
            # if traj_return is not None:
            #     for i, value in enumerate(traj_return):
            #         if i==path_index:
            #             plt.text(text_x, text_y_start-next(ge), 'track_error={:.4f}, collision_risk={:.4f}'.format(value[0], value[1]), fontsize=14, color=color[i], fontstyle='italic')
            #         else:
            #             plt.text(text_x, text_y_start-next(ge), 'track_error={:.4f}, collision_risk={:.4f}'.format(value[0], value[1]), fontsize=12, color=color[i], fontstyle='italic')
            ax.add_collection(PatchCollection(patches, match_original=True))
            plt.show()
            plt.pause(0.001)

    def set_traj(self, trajectory):
        """set the real trajectory to reconstruct observation"""
        self.ref_path = trajectory


def test_end2end():
    import tensorflow as tf
    env = CrossroadEnd2endMix()
    env_model = EnvironmentModel()
    obs, all_info = env.reset()
    i = 0
    while i < 100000:
        for j in range(50):
            i += 1
            action = np.array([-0.05, 0.3 + np.random.rand(1)*0.8], dtype=np.float32) # np.random.rand(1)*0.1 - 0.05
            obs, reward, done, info = env.step(action)
            obses, actions = obs[np.newaxis, :], action[np.newaxis, :]
            obses = tf.convert_to_tensor(np.tile(obs, (2, 1)), dtype=tf.float32)
            ref_points = tf.convert_to_tensor(np.tile(info['future_n_point'], (2, 1, 1)), dtype=tf.float32)
            actions = tf.convert_to_tensor(np.tile(actions, (2, 1)), dtype=tf.float32)
            # env_model.reset(obses)
            # for i in range(25):
            #     obses, rewards, punish_term_for_training, real_punish_term, veh2veh4real, veh2road4real, \
            #         veh2bike4real, veh2person4real = env_model.rollout_out(actions + tf.experimental.numpy.random.rand(2)*0.05, ref_points[:, :, i])
            #     # env_model.render()
            env.render(weights=np.zeros(env.other_number,))
            # if done:
            #     break
        obs, _ = env.reset()
        # env.render(weights=np.zeros(env.other_number,))


if __name__ == '__main__':
    test_end2end()