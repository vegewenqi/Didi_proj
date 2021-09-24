#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2020/11/08
# @Author  : Yang Guan (Tsinghua Univ.)
# @FileName: endtoend.py
# =====================================

import warnings
from collections import OrderedDict
from math import cos, sin, pi
import random
from random import choice

import gym
import matplotlib.pyplot as plt
import numpy as np
from gym.utils import seeding

from env_build.dynamics_and_models import VehicleDynamics, ReferencePath, EnvironmentModel
from env_build.endtoend_env_utils import shift_coordination, rotate_coordination, rotate_and_shift_coordination, deal_with_phi, \
    L, W, CROSSROAD_SIZE, LANE_WIDTH, LANE_NUMBER, judge_feasible, MODE2TASK, VEHICLE_MODE_DICT, BIKE_MODE_DICT, PERSON_MODE_DICT, \
    VEH_NUM, BIKE_NUM, PERSON_NUM, TASK_DICT, Para, LIGHT_DICT
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
                 num_future_data=0,
                 mode='training',
                 multi_display=False,
                 **kwargs):
        self.dynamics = VehicleDynamics()
        self.interested_vehs = None
        self.detected_vehicles = None
        self.all_vehicles = None
        self.ego_dynamics = None
        self.num_future_data = num_future_data
        self.init_state = {}
        self.action_number = 2
        self.ego_l, self.ego_w = L, W
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(self.action_number,), dtype=np.float32)

        self.seed()
        self.v_light = None
        self.light_vector = None
        self.step_length = 100  # ms

        self.step_time = self.step_length / 1000.0
        self.init_state = None
        self.obs = None
        self.action = None

        self.done_type = 'not_done_yet'
        self.reward_info = None
        self.ego_info_dim = 6
        self.track_info_dim = 3
        self.light_info_dim = 2
        self.task_info_dim = 3
        self.ref_info_dim = 3
        self.per_path_info_dim = 4
        self.per_bike_info_dim = 10
        self.per_person_info_dim = 10
        self.per_veh_info_dim = 10
        self.mode = mode
        if not multi_display:
            self.traffic = Traffic(self.step_length,
                                   mode=self.mode,
                                   init_n_ego_dict=self.init_state)
            self.reset()
            action = self.action_space.sample()
            observation, _reward, done, _info = self.step(action)
            self._set_observation_space(observation)
            plt.ion()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self, **kwargs):  # kwargs include three keys
        self.v_light = self.traffic.init_light()
        self.training_task = choice(['left', 'straight', 'right'])
        self.task_idx = TASK_DICT[self.training_task]
        self.light_vector = LIGHT_DICT[0] if self.training_task == 'right' else LIGHT_DICT[self.v_light]
        self.ref_path = ReferencePath(self.training_task, self.light_vector, **kwargs)
        self.env_model = EnvironmentModel(self.training_task, self.num_future_data)
        self.veh_mode_dict = VEHICLE_MODE_DICT[self.training_task]
        self.bicycle_mode_dict = BIKE_MODE_DICT[self.training_task]
        self.person_mode_dict = PERSON_MODE_DICT[self.training_task]
        self.veh_num = VEH_NUM[self.training_task]
        self.bike_num = BIKE_NUM[self.training_task]
        self.person_num = PERSON_NUM[self.training_task]
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
        self.obs = self._get_obs()
        self.action = None
        self.reward_info = None
        self.done_type = 'not_done_yet'
        return self.obs

    def close(self):
        del self.traffic

    def step(self, action):
        self.action = self._action_transformation_for_end2end(action)
        reward, self.reward_info = self.compute_reward(self.obs, self.action)
        next_ego_state, next_ego_params = self._get_next_ego_state(self.action)
        ego_dynamics = self._get_ego_dynamics(next_ego_state, next_ego_params)
        self.traffic.set_own_car(dict(ego=ego_dynamics))
        self.traffic.sim_step()
        all_info = self._get_all_info(ego_dynamics)
        self.obs = self._get_obs()
        self.done_type, done = self._judge_done()
        self.reward_info.update({'final_rew': reward})
        all_info.update({'reward_info': self.reward_info, 'ref_index': self.ref_path.ref_index,
                         'task': self.training_task, 'mask': self.vehs_mask})
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
                   miu_r=next_ego_params[3],)
        miu_f, miu_r = out['miu_f'], out['miu_r']
        F_zf, F_zr = self.dynamics.vehicle_params['F_zf'], self.dynamics.vehicle_params['F_zr']
        C_f, C_r = self.dynamics.vehicle_params['C_f'], self.dynamics.vehicle_params['C_r']
        alpha_f_bound, alpha_r_bound = 3 * miu_f * F_zf / C_f, 3 * miu_r * F_zr / C_r
        r_bound = miu_r * self.dynamics.vehicle_params['g'] / (abs(out['v_x'])+1e-8)

        l, w, x, y, phi = out['l'], out['w'], out['x'], out['y'], out['phi']

        def cal_corner_point_of_ego_car():
            x0, y0, a0 = rotate_and_shift_coordination(l / 2, w / 2, 0, -x, -y, -phi)
            x1, y1, a1 = rotate_and_shift_coordination(l / 2, -w / 2, 0, -x, -y, -phi)
            x2, y2, a2 = rotate_and_shift_coordination(-l / 2, w / 2, 0, -x, -y, -phi)
            x3, y3, a3 = rotate_and_shift_coordination(-l / 2, -w / 2, 0, -x, -y, -phi)
            return (x0, y0), (x1, y1), (x2, y2), (x3, y3)
        Corner_point = cal_corner_point_of_ego_car()
        out.update(dict(alpha_f_bound=alpha_f_bound,
                        alpha_r_bound=alpha_r_bound,
                        r_bound=r_bound,
                        Corner_point=Corner_point))

        return out

    def _get_all_info(self, ego_dynamics):  # used to update info, must be called every timestep before _get_obs
        # to fetch info
        self.all_vehicles = self.traffic.n_ego_vehicles['ego']  # coordination 2
        self.ego_dynamics = ego_dynamics  # coordination 2
        self.v_light = self.traffic.v_light

        # all_vehicles
        # dict(x=x, y=y, v=v, phi=a, l=length,
        #      w=width, route=route)

        all_info = dict(all_vehicles=self.all_vehicles,
                        ego_dynamics=self.ego_dynamics,
                        v_light=self.v_light)
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
        elif self._break_red_light():
            return 'break_red_light', 1
        elif self._is_achieve_goal():
            return 'good_done', 1
        else:
            return 'not_done_yet', 0

    def _deviate_too_much(self):
        delta_y, delta_phi, delta_v = self.obs[self.ego_info_dim:self.ego_info_dim+self.track_info_dim]
        return True if abs(delta_y) > 15 else False

    def _break_road_constrain(self):
        results = list(map(lambda x: judge_feasible(*x, self.training_task), self.ego_dynamics['Corner_point']))
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
        return True if self.v_light > 2 and self.ego_dynamics['y'] > -Para.CROSSROAD_SIZE_LON/2 and self.training_task != 'right' else False

    def _is_achieve_goal(self):
        x = self.ego_dynamics['x']
        y = self.ego_dynamics['y']
        if self.training_task == 'left':
            return True if x < -Para.CROSSROAD_SIZE_LAT/2 - 10 and Para.OFFSET_L + Para.GREEN_BELT_LAT < y < Para.OFFSET_L + Para.GREEN_BELT_LAT + Para.LANE_WIDTH_1 * 3 else False
        elif self.training_task == 'right':
            return True if x > Para.CROSSROAD_SIZE_LAT/2 + 10 and Para.OFFSET_R - Para.LANE_WIDTH_1 * 3 < y < Para.OFFSET_R else False
        else:
            assert self.training_task == 'straight'
            return True if y > Para.CROSSROAD_SIZE_LON/2 + 10 and Para.OFFSET_U + Para.GREEN_BELT_LON < x < Para.OFFSET_U + Para.GREEN_BELT_LON + Para.LANE_WIDTH_3 * 2 else False

    def _action_transformation_for_end2end(self, action):  # [-1, 1]
        action = np.clip(action, -1.05, 1.05)
        steer_norm, a_x_norm = action[0], action[1]
        scaled_steer = 0.4 * steer_norm
        scaled_a_x = 2.25*a_x_norm - 0.75  # [-3, 1.5]
        # if self.v_light != 0 and self.ego_dynamics['y'] < -25 and self.training_task != 'right':
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
        next_ego_state, next_ego_params = next_ego_state.numpy()[0],  next_ego_params.numpy()[0]
        next_ego_state[0] = next_ego_state[0] if next_ego_state[0] >= 0 else 0.
        next_ego_state[-1] = deal_with_phi(next_ego_state[-1])
        return next_ego_state, next_ego_params

    def _get_obs(self, exit_='D'):
        ego_x = self.ego_dynamics['x']
        ego_y = self.ego_dynamics['y']
        ego_phi = self.ego_dynamics['phi']
        ego_v_x = self.ego_dynamics['v_x']

        vehs_vector = self._construct_veh_vector_short(exit_)
        ego_vector = self._construct_ego_vector_short()
        tracking_error = self.ref_path.tracking_error_vector(np.array([ego_x], dtype=np.float32),
                                                             np.array([ego_y], dtype=np.float32),
                                                             np.array([ego_phi], dtype=np.float32),
                                                             np.array([ego_v_x], dtype=np.float32),
                                                             self.num_future_data).numpy()[0]
        self.light_vector = LIGHT_DICT[0] if self.training_task == 'right' else LIGHT_DICT[self.v_light]
        vector = np.concatenate((ego_vector, tracking_error, self.light_vector, self.task_idx, self.ref_path.ref_encoder, vehs_vector), axis=0)
        vector = vector.astype(np.float32)
        vector = self.convert_vehs_to_rela(vector)

        return vector

    def convert_vehs_to_rela(self, obs_abso):
        ego_road_infos = obs_abso[:self.ego_info_dim + self.track_info_dim + self.num_future_data * self.per_path_info_dim
                                                                                + self.light_info_dim + self.task_info_dim+ self.ref_info_dim]
        obses_others = obs_abso[self.ego_info_dim + self.track_info_dim + self.num_future_data * self.per_path_info_dim
                                + self.light_info_dim + self.task_info_dim + self.ref_info_dim:]
        bike_infos = obses_others[:self.bike_num * self.per_bike_info_dim]
        person_infos = obses_others[self.bike_num * self.per_bike_info_dim:self.bike_num * self.per_bike_info_dim + self.person_num * self.per_person_info_dim]
        veh_infos = obses_others[self.bike_num * self.per_bike_info_dim + self.person_num * self.per_person_info_dim:]

        ego_vx, ego_vy, ego_r, ego_x, ego_y, ego_phi = ego_road_infos[:self.ego_info_dim]
        ego = np.array(([ego_x, ego_y] + [0.] * (self.per_bike_info_dim - 2))*int(len(bike_infos)/self.per_bike_info_dim), dtype=np.float32)
        bikes_rela = bike_infos - ego
        ego = np.array(([ego_x, ego_y] + [0.] * (self.per_person_info_dim - 2))*int(len(person_infos)/self.per_person_info_dim), dtype=np.float32)
        persons_rela = person_infos - ego
        ego = np.array(([ego_x, ego_y] + [0.] * (self.per_veh_info_dim - 2))*int(len(veh_infos)/self.per_veh_info_dim), dtype=np.float32)
        vehs_rela = veh_infos - ego
        out = np.concatenate((ego_road_infos, bikes_rela, persons_rela, vehs_rela), axis=0)
        return out

    def convert_vehs_to_abso(self, obs_rela):
        ego_road_infos =obs_rela[:self.ego_info_dim + self.track_info_dim + self.num_future_data * self.per_path_info_dim
                                                                                + self.light_info_dim + self.task_info_dim + self.ref_info_dim]
        obses_others = obs_rela[self.ego_info_dim + self.track_info_dim + self.num_future_data * self.per_path_info_dim
                                                                                + self.light_info_dim + self.task_info_dim + self.ref_info_dim:]
        bike_infos = obses_others[:self.bike_num * self.per_bike_info_dim]
        person_infos = obses_others[self.bike_num * self.per_bike_info_dim:self.bike_num * self.per_bike_info_dim + self.person_num * self.per_person_info_dim]
        veh_infos = obses_others[self.bike_num * self.per_bike_info_dim + self.person_num * self.per_person_info_dim:]

        ego_vx, ego_vy, ego_r, ego_x, ego_y, ego_phi = ego_road_infos[:self.ego_info_dim]
        ego = np.array(([ego_x, ego_y] + [0.] * (self.per_bike_info_dim - 2))*int(len(bike_infos)/self.per_bike_info_dim), dtype=np.float32)
        bikes_abso = bike_infos + ego
        ego = np.array(([ego_x, ego_y] + [0.] * (self.per_person_info_dim - 2))*int(len(person_infos)/self.per_person_info_dim), dtype=np.float32)
        persons_abso = person_infos + ego
        ego = np.array(([ego_x, ego_y] + [0.] * (self.per_veh_info_dim - 2))*int(len(veh_infos)/self.per_veh_info_dim), dtype=np.float32)
        vehs_abso = veh_infos + ego
        out = np.concatenate((ego_road_infos, bikes_abso, persons_abso, vehs_abso), axis=0)
        return out

    def _construct_ego_vector_short(self):
        ego_v_x = self.ego_dynamics['v_x']
        ego_v_y = self.ego_dynamics['v_y']
        ego_r = self.ego_dynamics['r']
        ego_x = self.ego_dynamics['x']
        ego_y = self.ego_dynamics['y']
        ego_phi = self.ego_dynamics['phi']
        ego_feature = [ego_v_x, ego_v_y, ego_r, ego_x, ego_y, ego_phi]
        return np.array(ego_feature, dtype=np.float32)

    def _construct_veh_vector_short(self, exit_='D'):
        ego_x = self.ego_dynamics['x']
        ego_y = self.ego_dynamics['y']
        vehs_vector = []
        self.vehs_mask = []

        name_settings = dict(D=dict(do='1o', di='1i', ro='2o', ri='2i', uo='3o', ui='3i', lo='4o', li='4i'),
                             R=dict(do='2o', di='2i', ro='3o', ri='3i', uo='4o', ui='4i', lo='1o', li='1i'),
                             U=dict(do='3o', di='3i', ro='4o', ri='4i', uo='1o', ui='1i', lo='2o', li='2i'),
                             L=dict(do='4o', di='4i', ro='1o', ri='1i', uo='2o', ui='2i', lo='3o', li='3i'))

        name_setting = name_settings[exit_]

        def filter_interested_participants(vs, task):
            dl, du, dr, rd, rl, ru, ur, ud, ul, lu, lr, ld = [], [], [], [], [], [], [], [], [], [], [], []
            du_b, dr_b, rl_b, ru_b, ud_b, ul_b, lr_b, ld_b = [], [], [], [], [], [], [], []
            i1_0, o1_0, i2_0, o2_0, i3_0, o3_0, i4_0, o4_0, c0, c1, c2, c3, c_w0, c_w1, c_w2, c_w3 = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []

            # slice or fill to some number
            def slice_or_fill(sorted_list, fill_value, num):
                mask = [True] * len(sorted_list)
                if len(sorted_list) >= num:
                    return sorted_list[:num], mask[:num]
                else:
                    while len(sorted_list) < num:
                        sorted_list.append(fill_value)
                        mask.extend([False])
                    return sorted_list, mask

            for v in vs:
                if v['type'] in ['bicycle_1', 'bicycle_2', 'bicycle_3']:
                    v.update(partici_type=[1., 0., 0.])
                    v.update(turn_rad=0.0)
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
                    v.update(partici_type=[0., 1., 0.])
                    road_list = v['road']
                    v.update(turn_rad=0.0)
                    # print(road_list)
                    # if road_list == ':1i_0':
                    #     i1_0.append(v)
                    # elif road_list == ':1o_0':
                    #     o1_0.append(v)
                    # elif road_list == ':2i_0':
                    #     i2_0.append(v)
                    # elif road_list == ':2o_0':
                    #     o2_0.append(v)
                    # elif road_list == ':3i_0':
                    #     i3_0.append(v)
                    # elif road_list == ':3o_0':
                    #     o3_0.append(v)
                    # elif road_list == ':4i_0':
                    #     i4_0.append(v)
                    # elif road_list == ':4o_0':
                    #     o4_0.append(v)
                    if road_list == ':0_c0':
                        c0.append(v)
                    elif road_list == ':0_c1':
                        c1.append(v)
                    elif road_list == ':0_c2':
                        c2.append(v)
                    elif road_list == ':0_c3':
                        c3.append(v)
                    # elif road_list == 'c_w0':
                    #     c_w0.append(v)
                    # elif road_list == 'c_w1':
                    #     c_w1.append(v)
                    # elif road_list == 'c_w2':
                    #     c_w2.append(v)
                    # elif road_list == 'c_w3':
                    #     c_w3.append(v)

                else:
                    v.update(partici_type=[0., 0., 1.])
                    route_list = v['route']
                    start = route_list[0]
                    end = route_list[1]
                    if start == name_setting['do'] and end == name_setting['li']:
                        v.update(turn_rad=1/(CROSSROAD_SIZE/2+0.5*LANE_WIDTH))
                        dl.append(v)
                    elif start == name_setting['do'] and end == name_setting['ui']:
                        v.update(turn_rad=0.)
                        du.append(v)
                    elif start == name_setting['do'] and end == name_setting['ri']:
                        v.update(turn_rad=-1 / (CROSSROAD_SIZE/2-2.5*LANE_WIDTH))
                        dr.append(v)

                    elif start == name_setting['ro'] and end == name_setting['di']:
                        v.update(turn_rad=1 / (CROSSROAD_SIZE / 2 + 0.5 * LANE_WIDTH))
                        rd.append(v)
                    elif start == name_setting['ro'] and end == name_setting['li']:
                        v.update(turn_rad=0.)
                        rl.append(v)
                    elif start == name_setting['ro'] and end == name_setting['ui']:
                        v.update(turn_rad=-1 / (CROSSROAD_SIZE/2-2.5*LANE_WIDTH))
                        ru.append(v)

                    elif start == name_setting['uo'] and end == name_setting['ri']:
                        v.update(turn_rad=1 / (CROSSROAD_SIZE / 2 + 0.5 * LANE_WIDTH))
                        ur.append(v)
                    elif start == name_setting['uo'] and end == name_setting['di']:
                        v.update(turn_rad=0.)
                        ud.append(v)
                    elif start == name_setting['uo'] and end == name_setting['li']:
                        v.update(turn_rad=-1 / (CROSSROAD_SIZE/2-2.5*LANE_WIDTH))
                        ul.append(v)

                    elif start == name_setting['lo'] and end == name_setting['ui']:
                        v.update(turn_rad=1 / (CROSSROAD_SIZE / 2 + 0.5 * LANE_WIDTH))
                        lu.append(v)
                    elif start == name_setting['lo'] and end == name_setting['ri']:
                        v.update(turn_rad=0.)
                        lr.append(v)
                    elif start == name_setting['lo'] and end == name_setting['di']:
                        v.update(turn_rad=-1 / (CROSSROAD_SIZE/2-2.5*LANE_WIDTH))
                        ld.append(v)

            # fetch bicycle in range
            if task == 'straight':
                du_b = list(filter(lambda v: ego_y - 2 < v['y'] < CROSSROAD_SIZE / 2 and v['x'] < ego_x + 8, du_b))
            elif task == 'right':
                du_b = list(filter(lambda v: ego_y - 2 < v['y'] < 0 and v['x'] < ego_x + 8, du_b))
            # dr_b = list(filter(lambda v: v['x'] < CROSSROAD_SIZE / 2 + 10 and v['y'] > ego_y - 2, dr_b))  # interest of right
            # rl_b = rl_b  # not interest in case of traffic light
            # ru_b = list(filter(lambda v: v['x'] < CROSSROAD_SIZE / 2 + 10 and v['y'] < CROSSROAD_SIZE / 2 + 10, ru_b))  # interest of straight
            ud_b = list(filter(lambda v: max(ego_y - 2, -CROSSROAD_SIZE/2) < v['y'] < CROSSROAD_SIZE / 2 and ego_x > v['x'], ud_b))  # interest of left
            # ul_b = list(filter(lambda v: -CROSSROAD_SIZE / 2 - 10 < v['x'] < ego_x and v['y'] < CROSSROAD_SIZE / 2, ul_b))  # interest of left
            lr_b = list(filter(lambda v: 0 < v['x'] < CROSSROAD_SIZE / 2 + 10, lr_b))  # interest of right
            # ld_b = ld_b  # not interest in case of traffic light

            # sort
            du_b = sorted(du_b, key=lambda v: v['y'])
            # dr_b = sorted(dr_b, key=lambda v: (v['y'], v['x']))
            # ru_b = sorted(ru_b, key=lambda v: (-v['x'], v['y']), reverse=True)
            ud_b = sorted(ud_b, key=lambda v: v['y'])
            # ul_b = sorted(ul_b, key=lambda v: (-v['y'], -v['x']), reverse=True)
            lr_b = sorted(lr_b, key=lambda v: -v['x'])

            mode2fillvalue_b = dict(
                du_b=dict(type="bicycle_1", x=LANE_WIDTH * LANE_NUMBER + 1, y=-(CROSSROAD_SIZE / 2 + 30), v=0,
                        phi=90, w=0.48, l=2, route=('1o', '3i'), partici_type=[1., 0., 0.], turn_rad=0.),
                # dr=dict(type="bicycle_1", x=LANE_WIDTH * LANE_NUMBER + 1, y=-(CROSSROAD_SIZE / 2 + 30), v=0,
                #         phi=90, w=0.48, l=2, route=('1o', '2i')),
                # ru=dict(type="bicycle_1", x=(CROSSROAD_SIZE / 2 + 15), y=LANE_WIDTH * LANE_NUMBER + 1, v=0,
                #         phi=180, w=0.48, l=2, route=('2o', '3i')),
                ud_b=dict(type="bicycle_1", x=-LANE_WIDTH * LANE_NUMBER - 1, y=(CROSSROAD_SIZE / 2 + 20), v=0,
                        phi=-90, w=0.48, l=2, route=('3o', '1i'), partici_type=[1., 0., 0.], turn_rad=0.),
                # ul=dict(type="bicycle_1", x=-LANE_WIDTH * LANE_NUMBER - 1, y=(CROSSROAD_SIZE / 2 + 20), v=0,
                #         phi=-90, w=0.48, l=2, route=('3o', '4i')),
                lr_b=dict(type="bicycle_1", x=-(CROSSROAD_SIZE / 2 + 20), y=-LANE_WIDTH * LANE_NUMBER - 1,
                        v=0, phi=0, w=0.48, l=2, route=('4o', '2i'), partici_type=[1., 0., 0.], turn_rad=0.))

            tmp_b = OrderedDict()
            masks_b = OrderedDict()
            for mode, num in BIKE_MODE_DICT[task].items():
                tmp_b[mode], masks_b[mode] = slice_or_fill(eval(mode), mode2fillvalue_b[mode], num)

            # fetch person in range
            c1 = list(filter(lambda v: v['y'] < 6 and v['x'] > ego_x - 6, c1))  # interest of right
            c2 = list(filter(lambda v: 0 < v['x'] and v['y'] > ego_y - 4, c2))  # interest of right
            c3 = list(filter(lambda v: -6 < v['y'] and v['x'] < ego_x + 6, c3))  # interest of left

            # sort
            c1 = sorted(c1, key=lambda v: (abs(v['y'] - ego_y), v['x']))
            c2 = sorted(c2, key=lambda v: (abs(v['x'] - ego_x), v['y']))
            c3 = sorted(c3, key=lambda v: (abs(v['y'] - ego_y), -v['x']))

            mode2fillvalue_p = dict(
                c1=dict(type='DEFAULT_PEDTYPE', x=LANE_WIDTH*LANE_NUMBER+3, y=-(CROSSROAD_SIZE / 2 + 30), v=0, phi=90, w=0.525,l=0.75, road="0_c1", partici_type=[0., 1., 0.], turn_rad=0.),
                c2=dict(type='DEFAULT_PEDTYPE', x=-(CROSSROAD_SIZE/2+20), y=-(LANE_WIDTH*LANE_NUMBER+3), v=0, phi=0, w=0.525, l=0.75, road="0_c2", partici_type=[0., 1., 0.], turn_rad=0.),
                c3=dict(type='DEFAULT_PEDTYPE', x=-(LANE_WIDTH*LANE_NUMBER+3), y=(CROSSROAD_SIZE/2+20), v=0, phi=-90, w=0.525, l=0.75, road="0_c3", partici_type=[0., 1., 0.], turn_rad=0.))

            tmp_p = OrderedDict()
            masks_p = OrderedDict()
            for mode, num in PERSON_MODE_DICT[task].items():
                tmp_p[mode], masks_p[mode] = slice_or_fill(eval(mode), mode2fillvalue_p[mode], num)

            # fetch veh in range
            dl = list(filter(lambda v: v['x'] > -CROSSROAD_SIZE/2-10 and v['y'] > ego_y-2, dl))  # interest of left straight
            du = list(filter(lambda v: ego_y-2 < v['y'] < CROSSROAD_SIZE/2+10 and v['x'] < ego_x+5, du))  # interest of left straight

            dr = list(filter(lambda v: v['x'] < CROSSROAD_SIZE/2+10 and v['y'] > ego_y, dr))  # interest of right

            rd = rd  # not interest in case of traffic light
            rl = rl  # not interest in case of traffic light
            ru = list(filter(lambda v: v['x'] < CROSSROAD_SIZE/2+10 and v['y'] < CROSSROAD_SIZE/2+10, ru))  # interest of straight

            if task == 'straight':
                ur = list(filter(lambda v: v['x'] < ego_x + 7 and ego_y < v['y'] < CROSSROAD_SIZE/2+10, ur))  # interest of straight
            elif task == 'right':
                ur = list(filter(lambda v: v['x'] < CROSSROAD_SIZE/2+10 and v['y'] < CROSSROAD_SIZE/2, ur))  # interest of right
            ud = list(filter(lambda v: max(ego_y-2, -CROSSROAD_SIZE/2) < v['y'] < CROSSROAD_SIZE/2 and ego_x > v['x'], ud))  # interest of left
            ul = list(filter(lambda v: -CROSSROAD_SIZE/2-10 < v['x'] < ego_x and v['y'] < CROSSROAD_SIZE/2, ul))  # interest of left

            lu = lu  # not interest in case of traffic light
            lr = list(filter(lambda v: -CROSSROAD_SIZE/2-10 < v['x'] < CROSSROAD_SIZE/2+10, lr))  # interest of right
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
                dl=dict(type="car_1", x=LANE_WIDTH/2, y=-(CROSSROAD_SIZE/2+30), v=0, phi=90, w=2.5, l=5, route=('1o', '4i'), partici_type=[0., 0., 1.], turn_rad=1/(CROSSROAD_SIZE/2+0.5*LANE_WIDTH)),
                du=dict(type="car_1", x=LANE_WIDTH*1.5, y=-(CROSSROAD_SIZE/2+30), v=0, phi=90, w=2.5, l=5, route=('1o', '3i'), partici_type=[0., 0., 1.], turn_rad=0.),
                dr=dict(type="car_1", x=LANE_WIDTH*(LANE_NUMBER-0.5), y=-(CROSSROAD_SIZE/2+30), v=0, phi=90, w=2.5, l=5, route=('1o', '2i'), partici_type=[0., 0., 1.], turn_rad=-1 / (CROSSROAD_SIZE/2-2.5*LANE_WIDTH)),
                ru=dict(type="car_1", x=(CROSSROAD_SIZE/2+15), y=LANE_WIDTH*(LANE_NUMBER-0.5), v=0, phi=180, w=2.5, l=5, route=('2o', '3i'), partici_type=[0., 0., 1.], turn_rad=-1 / (CROSSROAD_SIZE/2-2.5*LANE_WIDTH)),
                ur=dict(type="car_1", x=-LANE_WIDTH/2, y=(CROSSROAD_SIZE/2+20), v=0, phi=-90, w=2.5, l=5, route=('3o', '2i'), partici_type=[0., 0., 1.], turn_rad=1/(CROSSROAD_SIZE/2+0.5*LANE_WIDTH)),
                ud=dict(type="car_1", x=-LANE_WIDTH*1.5, y=(CROSSROAD_SIZE/2+20), v=0, phi=-90, w=2.5, l=5, route=('3o', '1i'), partici_type=[0., 0., 1.], turn_rad=0.),
                ul=dict(type="car_1", x=-LANE_WIDTH*(LANE_NUMBER-0.5), y=(CROSSROAD_SIZE/2+20), v=0, phi=-90, w=2.5, l=5, route=('3o', '4i'), partici_type=[0., 0., 1.], turn_rad=-1 / (CROSSROAD_SIZE/2-2.5*LANE_WIDTH)),
                lr=dict(type="car_1", x=-(CROSSROAD_SIZE/2+20), y=-LANE_WIDTH*1.5, v=0, phi=0, w=2.5, l=5, route=('4o', '2i'), partici_type=[0., 0., 1.], turn_rad=0.))

            tmp_v = OrderedDict()
            masks_v = OrderedDict()
            for mode, num in VEHICLE_MODE_DICT[task].items():
                tmp_v[mode], masks_v[mode] = slice_or_fill(eval(mode), mode2fillvalue[mode], num)

            tmp1 = dict(tmp_b, **tmp_p)
            tmp = dict(tmp1, **tmp_v)
            masks_dict = dict(dict(masks_b, **masks_p), **masks_v)
            return tmp, masks_dict

        list_of_interested_veh_dict = []
        self.interested_vehs, self.mask_mode = filter_interested_participants(self.all_vehicles, self.training_task)
        for part in list(self.interested_vehs.values()):
            list_of_interested_veh_dict.extend(part)
        for part in list(self.mask_mode.values()):
            self.vehs_mask.extend(part)

        for veh in list_of_interested_veh_dict:
            veh_x, veh_y, veh_v, veh_phi, veh_l, veh_w, veh_partici_type, veh_turn_rad = veh['x'], veh['y'], veh['v'], veh['phi'], \
                                                                           veh['l'], veh['w'], veh['partici_type'], veh['turn_rad']
            vehs_vector.extend([veh_x, veh_y, veh_v, veh_phi, veh_l, veh_w])
            vehs_vector.extend(veh_partici_type)
            vehs_vector.extend([veh_turn_rad])
        return np.array(vehs_vector, dtype=np.float32)

    def recover_orig_position_fn(self, transformed_x, transformed_y, x, y, d):  # x, y, d are used to transform
        # coordination
        transformed_x, transformed_y, _ = rotate_coordination(transformed_x, transformed_y, 0, -d)
        orig_x, orig_y = shift_coordination(transformed_x, transformed_y, -x, -y)
        return orig_x, orig_y

    def _reset_init_state(self):
        if self.training_task == 'left':
            random_index = int(np.random.random()*(900+500)) + 700
        elif self.training_task == 'straight':
            random_index = int(np.random.random()*(1200+500)) + 700
        else:
            random_index = int(np.random.random()*(420+500)) + 700

        x, y, phi, exp_v = self.ref_path.indexs2points(random_index)
        v = exp_v * np.random.random()
        if self.training_task == 'left':
            routeID = 'dl'
        elif self.training_task == 'straight':
            routeID = 'du'
        else:
            assert self.training_task == 'right'
            routeID = 'dr'
        return dict(ego=dict(v_x=v,
                             v_y=0,
                             r=0,
                             x=x.numpy(),
                             y=y.numpy(),
                             phi=phi.numpy(),
                             l=self.ego_l,
                             w=self.ego_w,
                             routeID=routeID,
                             ))

    def compute_reward(self, obs, action):
        obses, actions = obs[np.newaxis, :], action[np.newaxis, :]

        # extract infos for each kind of participants
        start = 0; end = start + self.ego_info_dim + self.track_info_dim + self.per_path_info_dim * self.num_future_data + self.light_info_dim + self.task_info_dim
        obses_ego = obses[:, start:end]
        start = end; end = start + self.per_bike_info_dim * self.bike_num
        obses_bike = obses[:, start:end]
        start = end; end = start + self.per_person_info_dim * self.person_num
        obses_person = obses[:, start:end]
        start = end; end = start + self.per_veh_info_dim * self.veh_num
        obses_veh = obses[:, start:end]

        reward, _, _, _, _, _, _, reward_dict = self.env_model.compute_rewards(obses_ego, obses_bike, obses_person, obses_veh, actions)
        for k, v in reward_dict.items():
            reward_dict[k] = v.numpy()[0]
        return reward.numpy()[0], reward_dict

    def render(self, mode='human'):
        if mode == 'human':
            # plot basic map
            square_length = 50
            extension = 40
            dotted_line_style = '--'
            solid_line_style = '-'

            plt.cla()
            ax = plt.axes([-0.05, -0.05, 1.1, 1.1])
            ax.axis("equal")
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
            ax.add_patch(plt.Rectangle((-Para.CROSSROAD_SIZE_LAT / 2 - extension, Para.OFFSET_L),
                                                     extension, Para.GREEN_BELT_LAT, edgecolor='black', facecolor='green', linewidth=1))
            ax.add_patch(plt.Rectangle((Para.CROSSROAD_SIZE_LAT / 2 , Para.OFFSET_R),
                                                     extension, Para.GREEN_BELT_LAT, edgecolor='black', facecolor='green', linewidth=1))
            ax.add_patch(plt.Rectangle((Para.OFFSET_U, Para.CROSSROAD_SIZE_LON / 2),
                                                     Para.GREEN_BELT_LON, extension, edgecolor='black', facecolor='green', linewidth=1))
            ax.add_patch(plt.Rectangle((Para.OFFSET_D-Para.GREEN_BELT_LON, -Para.CROSSROAD_SIZE_LON / 2-extension),
                                                     Para.GREEN_BELT_LON, extension, edgecolor='black', facecolor='green', linewidth=1))

            # Left out lane
            for i in range(1, Para.LANE_NUMBER_LAT_OUT + 2):
                lane_width_flag = [Para.LANE_WIDTH_1, Para.LANE_WIDTH_1, Para.LANE_WIDTH_1, Para.PERSON_LANE_WIDTH + Para.BIKE_LANE_WIDTH]
                base = Para.OFFSET_L + Para.GREEN_BELT_LAT
                linestyle = dotted_line_style if i < Para.LANE_NUMBER_LAT_OUT else solid_line_style
                linewidth = 1 if i < Para.LANE_NUMBER_LAT_OUT else 1
                plt.plot([-Para.CROSSROAD_SIZE_LAT / 2 - extension, -Para.CROSSROAD_SIZE_LAT / 2], [base + sum(lane_width_flag[:i]), base + sum(lane_width_flag[:i])],
                         linestyle=linestyle, color='black', linewidth=linewidth)
            # Left in lane
            for i in range(1, Para.LANE_NUMBER_LAT_IN + 2):
                lane_width_flag = [Para.LANE_WIDTH_1, Para.LANE_WIDTH_1, Para.LANE_WIDTH_1, Para.LANE_WIDTH_1, Para.PERSON_LANE_WIDTH + Para.BIKE_LANE_WIDTH]
                base = Para.OFFSET_L
                linestyle = dotted_line_style if i < Para.LANE_NUMBER_LAT_IN else solid_line_style
                linewidth = 1 if i < Para.LANE_NUMBER_LAT_IN else 1
                plt.plot([-Para.CROSSROAD_SIZE_LAT / 2 - extension, -Para.CROSSROAD_SIZE_LAT / 2], [base - sum(lane_width_flag[:i]), base - sum(lane_width_flag[:i])],
                         linestyle=linestyle, color='black', linewidth=linewidth)

            # Right out lane
            for i in range(1, Para.LANE_NUMBER_LAT_OUT + 2):
                lane_width_flag = [Para.LANE_WIDTH_1, Para.LANE_WIDTH_1, Para.LANE_WIDTH_1, Para.PERSON_LANE_WIDTH + Para.BIKE_LANE_WIDTH]
                base = Para.OFFSET_R
                linestyle = dotted_line_style if i < Para.LANE_NUMBER_LAT_OUT else solid_line_style
                linewidth = 1 if i < Para.LANE_NUMBER_LAT_OUT else 1
                plt.plot([Para.CROSSROAD_SIZE_LAT / 2, Para.CROSSROAD_SIZE_LAT / 2 + extension], [base - sum(lane_width_flag[:i]), base - sum(lane_width_flag[:i])],
                         linestyle=linestyle, color='black', linewidth=linewidth)

            # Right in lane
            for i in range(1, Para.LANE_NUMBER_LAT_IN + 2):
                lane_width_flag = [Para.LANE_WIDTH_1, Para.LANE_WIDTH_1, Para.LANE_WIDTH_1, Para.LANE_WIDTH_1, Para.PERSON_LANE_WIDTH + Para.BIKE_LANE_WIDTH]
                base = Para.OFFSET_R + Para.GREEN_BELT_LAT
                linestyle = dotted_line_style if i < Para.LANE_NUMBER_LAT_IN else solid_line_style
                linewidth = 1 if i < Para.LANE_NUMBER_LAT_IN else 1
                plt.plot([Para.CROSSROAD_SIZE_LAT / 2, Para.CROSSROAD_SIZE_LAT / 2 + extension], [base + sum(lane_width_flag[:i]), base + sum(lane_width_flag[:i])],
                         linestyle=linestyle, color='black', linewidth=linewidth)

            # Up in lane
            for i in range(1, Para.LANE_NUMBER_LON_IN + 2):
                lane_width_flag = [Para.LANE_WIDTH_2, Para.LANE_WIDTH_3, Para.LANE_WIDTH_3, Para.PERSON_LANE_WIDTH+Para.BIKE_LANE_WIDTH]
                base = Para.OFFSET_U
                linestyle = dotted_line_style if i < Para.LANE_NUMBER_LON_IN else solid_line_style
                linewidth = 1 if i < Para.LANE_NUMBER_LON_IN else 1
                plt.plot([base - sum(lane_width_flag[:i]) , base - sum(lane_width_flag[:i])],[Para.CROSSROAD_SIZE_LON / 2 + extension, Para.CROSSROAD_SIZE_LON / 2],
                         linestyle=linestyle, color='black', linewidth=linewidth)

            # Up out lane
            for i in range(1, Para.LANE_NUMBER_LON_OUT + 2):
                lane_width_flag = [Para.LANE_WIDTH_3, Para.LANE_WIDTH_3, Para.PERSON_LANE_WIDTH + Para.BIKE_LANE_WIDTH]
                base = Para.OFFSET_U + Para.GREEN_BELT_LON
                linestyle = dotted_line_style if i < Para.LANE_NUMBER_LON_OUT else solid_line_style
                linewidth = 1 if i < Para.LANE_NUMBER_LON_OUT else 1
                plt.plot([base + sum(lane_width_flag[:i]) , base + sum(lane_width_flag[:i])], [Para.CROSSROAD_SIZE_LON / 2 + extension, Para.CROSSROAD_SIZE_LON / 2],
                         linestyle=linestyle, color='black', linewidth=linewidth)

            # Down in lane
            for i in range(1, Para.LANE_NUMBER_LON_IN + 2):
                lane_width_flag = [Para.LANE_WIDTH_2, Para.LANE_WIDTH_3, Para.LANE_WIDTH_3, Para.PERSON_LANE_WIDTH+Para.BIKE_LANE_WIDTH]
                base = Para.OFFSET_D
                linestyle = dotted_line_style if i < Para.LANE_NUMBER_LON_IN else solid_line_style
                linewidth = 1 if i < Para.LANE_NUMBER_LON_IN else 1
                plt.plot([base + sum(lane_width_flag[:i]) , base + sum(lane_width_flag[:i])],[-Para.CROSSROAD_SIZE_LON / 2 - extension, -Para.CROSSROAD_SIZE_LON / 2],
                         linestyle=linestyle, color='black', linewidth=linewidth)

            # Down out lane
            for i in range(1, Para.LANE_NUMBER_LON_OUT + 2):
                lane_width_flag = [Para.LANE_WIDTH_3, Para.LANE_WIDTH_3, Para.PERSON_LANE_WIDTH + Para.BIKE_LANE_WIDTH]
                base = Para.OFFSET_D - Para.GREEN_BELT_LON
                linestyle = dotted_line_style if i < Para.LANE_NUMBER_LON_OUT else solid_line_style
                linewidth = 1 if i < Para.LANE_NUMBER_LON_OUT else 1
                plt.plot([base - sum(lane_width_flag[:i]) , base - sum(lane_width_flag[:i])], [-Para.CROSSROAD_SIZE_LON / 2 - extension, -Para.CROSSROAD_SIZE_LON / 2],
                         linestyle=linestyle, color='black', linewidth=linewidth)

            # Oblique
            plt.plot([-Para.CROSSROAD_SIZE_LAT / 2, Para.OFFSET_U - (Para.LANE_NUMBER_LON_IN-1) * Para.LANE_WIDTH_3 - Para.LANE_WIDTH_2 - Para.BIKE_LANE_WIDTH - Para.PERSON_LANE_WIDTH], [Para.OFFSET_L + Para.GREEN_BELT_LAT + Para.LANE_NUMBER_LAT_OUT * Para.LANE_WIDTH_1 + Para.BIKE_LANE_WIDTH + Para.PERSON_LANE_WIDTH, Para.CROSSROAD_SIZE_LON / 2],
                     color='black', linewidth=1)
            plt.plot([-Para.CROSSROAD_SIZE_LAT / 2,
                      Para.OFFSET_D- Para.GREEN_BELT_LON - Para.LANE_NUMBER_LON_OUT * Para.LANE_WIDTH_3 - Para.BIKE_LANE_WIDTH - Para.PERSON_LANE_WIDTH],
                     [Para.OFFSET_L - Para.LANE_NUMBER_LAT_IN * Para.LANE_WIDTH_1 - Para.BIKE_LANE_WIDTH - Para.PERSON_LANE_WIDTH,
                         -Para.CROSSROAD_SIZE_LON / 2],
                     color='black', linewidth=1)
            plt.plot([Para.CROSSROAD_SIZE_LAT / 2,
                      Para.OFFSET_D + (Para.LANE_NUMBER_LON_IN-1) * Para.LANE_WIDTH_3 + Para.LANE_WIDTH_2 + Para.BIKE_LANE_WIDTH + Para.PERSON_LANE_WIDTH],
                     [Para.OFFSET_R - Para.LANE_NUMBER_LAT_OUT * Para.LANE_WIDTH_1 - Para.BIKE_LANE_WIDTH - Para.PERSON_LANE_WIDTH, -Para.CROSSROAD_SIZE_LON / 2],
                     color='black', linewidth=1)
            plt.plot([Para.CROSSROAD_SIZE_LAT / 2,
                      Para.OFFSET_U + Para.GREEN_BELT_LON + Para.LANE_NUMBER_LON_OUT * Para.LANE_WIDTH_3 + Para.BIKE_LANE_WIDTH + Para.PERSON_LANE_WIDTH],
                     [Para.OFFSET_R + Para.GREEN_BELT_LAT + Para.LANE_NUMBER_LAT_IN * Para.LANE_WIDTH_1 + Para.BIKE_LANE_WIDTH + Para.PERSON_LANE_WIDTH, Para.CROSSROAD_SIZE_LON / 2],
                     color='black', linewidth=1)

            # stop line
            lane_width_flag = [Para.LANE_WIDTH_2, Para.LANE_WIDTH_3, Para.LANE_WIDTH_3, Para.PERSON_LANE_WIDTH + Para.BIKE_LANE_WIDTH]   # Down
            plt.plot([Para.OFFSET_D, Para.OFFSET_D+sum(lane_width_flag[:Para.LANE_NUMBER_LON_IN])], [-Para.CROSSROAD_SIZE_LON / 2, -Para.CROSSROAD_SIZE_LON / 2], color='black')
            lane_width_flag = [Para.LANE_WIDTH_2, Para.LANE_WIDTH_3, Para.LANE_WIDTH_3, Para.PERSON_LANE_WIDTH + Para.BIKE_LANE_WIDTH]   # Up
            plt.plot([-sum(lane_width_flag[:Para.LANE_NUMBER_LON_IN])+Para.OFFSET_U, Para.OFFSET_U], [Para.CROSSROAD_SIZE_LON / 2, Para.CROSSROAD_SIZE_LON / 2], color='black')
            lane_width_flag = [Para.LANE_WIDTH_1, Para.LANE_WIDTH_1, Para.LANE_WIDTH_1, Para.LANE_WIDTH_1, Para.PERSON_LANE_WIDTH + Para.BIKE_LANE_WIDTH]
            plt.plot([-Para.CROSSROAD_SIZE_LAT / 2, -Para.CROSSROAD_SIZE_LAT / 2], [Para.OFFSET_L, Para.OFFSET_L-sum(lane_width_flag[:Para.LANE_NUMBER_LAT_IN])], color='black') # left
            lane_width_flag = [Para.LANE_WIDTH_1, Para.LANE_WIDTH_1, Para.LANE_WIDTH_1, Para.LANE_WIDTH_1, Para.PERSON_LANE_WIDTH + Para.BIKE_LANE_WIDTH]
            plt.plot([Para.CROSSROAD_SIZE_LAT / 2, Para.CROSSROAD_SIZE_LAT / 2], [Para.OFFSET_R+Para.GREEN_BELT_LAT,  Para.OFFSET_R+Para.GREEN_BELT_LAT+sum(lane_width_flag[:Para.LANE_NUMBER_LAT_IN])], color='black')

            v_light = self.v_light
            if v_light == 0 or v_light == 1:
                v_color, h_color = 'green', 'red'
            elif v_light == 2:
                v_color, h_color = 'orange', 'red'
            elif v_light == 3 or v_light == 4:
                v_color, h_color = 'red', 'green'
            else:
                v_color, h_color = 'red', 'orange'
            #
            # plt.plot([0, (LANE_NUMBER-1)*lane_width], [-square_length / 2, -square_length / 2],
            #          color=v_color, linewidth=light_line_width)
            # plt.plot([(LANE_NUMBER-1)*lane_width, LANE_NUMBER * lane_width], [-square_length / 2, -square_length / 2],
            #          color='green', linewidth=light_line_width)
            #
            # plt.plot([-LANE_NUMBER * lane_width, -(LANE_NUMBER-1)*lane_width], [square_length / 2, square_length / 2],
            #          color='green', linewidth=light_line_width)
            # plt.plot([-(LANE_NUMBER-1)*lane_width, 0], [square_length / 2, square_length / 2],
            #          color=v_color, linewidth=light_line_width)
            #
            # plt.plot([-square_length / 2, -square_length / 2], [0, -(LANE_NUMBER-1)*lane_width],
            #          color=h_color, linewidth=light_line_width)
            # plt.plot([-square_length / 2, -square_length / 2], [-(LANE_NUMBER-1)*lane_width, -LANE_NUMBER * lane_width],
            #          color='green', linewidth=light_line_width)
            #
            # plt.plot([square_length / 2, square_length / 2], [(LANE_NUMBER-1)*lane_width, 0],
            #          color=h_color, linewidth=light_line_width)
            # plt.plot([square_length / 2, square_length / 2], [LANE_NUMBER * lane_width, (LANE_NUMBER-1)*lane_width],
            #          color='green', linewidth=light_line_width)
            #
            # # ----------人行横道--------------
            # jj = 3.5
            # for ii in range(23):
            #     if ii <= 3:
            #         continue
            #     ax.add_patch(plt.Rectangle((-square_length / 2 + jj + ii * 1.6, -square_length / 2 + 0.5), 0.8, 4,
            #                                color='lightgray', alpha=0.5))
            #     ii += 1
            # for ii in range(23):
            #     if ii <= 3:
            #         continue
            #     ax.add_patch(plt.Rectangle((-square_length / 2 + jj + ii * 1.6, square_length / 2 - 0.5 - 4), 0.8, 4,
            #                                color='lightgray', alpha=0.5))
            #     ii += 1
            # for ii in range(23):
            #     if ii <= 3:
            #         continue
            #     ax.add_patch(
            #         plt.Rectangle((-square_length / 2 + 0.5, square_length / 2 - jj - 0.8 - ii * 1.6), 4, 0.8,
            #                       color='lightgray',
            #                       alpha=0.5))
            #     ii += 1
            # for ii in range(23):
            #     if ii <= 3:
            #         continue
            #     ax.add_patch(
            #         plt.Rectangle((square_length / 2 - 0.5 - 4, square_length / 2 - jj - 0.8 - ii * 1.6), 4, 0.8,
            #                       color='lightgray',
            #                       alpha=0.5))
            #     ii += 1
            #
            def is_in_plot_area(x, y, tolerance=5):
                if -square_length / 2 - extension + tolerance < x < square_length / 2 + extension - tolerance and \
                        -square_length / 2 - extension + tolerance < y < square_length / 2 + extension - tolerance:
                    return True
                else:
                    return False

            def draw_rotate_rec(x, y, a, l, w, color, linestyle='-'):
                RU_x, RU_y, _ = rotate_coordination(l / 2, w / 2, 0, -a)
                RD_x, RD_y, _ = rotate_coordination(l / 2, -w / 2, 0, -a)
                LU_x, LU_y, _ = rotate_coordination(-l / 2, w / 2, 0, -a)
                LD_x, LD_y, _ = rotate_coordination(-l / 2, -w / 2, 0, -a)
                ax.plot([RU_x + x, RD_x + x], [RU_y + y, RD_y + y], color=color, linestyle=linestyle)
                ax.plot([RU_x + x, LU_x + x], [RU_y + y, LU_y + y], color=color, linestyle=linestyle)
                ax.plot([LD_x + x, RD_x + x], [LD_y + y, RD_y + y], color=color, linestyle=linestyle)
                ax.plot([LD_x + x, LU_x + x], [LD_y + y, LU_y + y], color=color, linestyle=linestyle)

            def plot_phi_line(type, x, y, phi, color):
                if type in ['bicycle_1', 'bicycle_2', 'bicycle_3']:
                    line_length = 2
                elif type == 'DEFAULT_PEDTYPE':
                    line_length = 1
                else:
                    line_length = 5
                x_forw, y_forw = x + line_length * cos(phi*pi/180.),\
                                 y + line_length * sin(phi*pi/180.)
                plt.plot([x, x_forw], [y, y_forw], color=color, linewidth=0.5)

            # plot cars
            for veh in self.all_vehicles:
                veh_x = veh['x']
                veh_y = veh['y']
                veh_phi = veh['phi']
                veh_l = veh['l']
                veh_w = veh['w']
                veh_type = veh['type']
                if veh_type in ['bicycle_1', 'bicycle_2', 'bicycle_3']:
                    veh_color = 'navy'
                elif veh_type == 'DEFAULT_PEDTYPE':
                    veh_color = 'purple'
                else:
                    veh_color = 'black'
                if is_in_plot_area(veh_x, veh_y):
                    plot_phi_line(veh_type, veh_x, veh_y, veh_phi, veh_color)
                    draw_rotate_rec(veh_x, veh_y, veh_phi, veh_l, veh_w, veh_color)

            # plot_interested vehs
            for mode, num in self.veh_mode_dict.items():
                for i in range(num):
                    veh = self.interested_vehs[mode][i]
                    mask = self.mask_mode[mode][i]
                    veh_x = veh['x']
                    veh_y = veh['y']
                    veh_phi = veh['phi']
                    veh_l = veh['l']
                    veh_w = veh['w']
                    veh_type = veh['type']
                    task2color = {'left': 'b', 'straight': 'c', 'right': 'm'}

                    if is_in_plot_area(veh_x, veh_y):
                        plot_phi_line(veh_type, veh_x, veh_y, veh_phi, 'black')
                        task = MODE2TASK[mode]
                        color = task2color[task]
                        draw_rotate_rec(veh_x, veh_y, veh_phi, veh_l, veh_w, color, linestyle=':')
                        plt.text(veh_x, veh_y, str(mask)[0])

            # plot_interested bicycle
            for mode, num in self.bicycle_mode_dict.items():
                for i in range(num):
                    veh = self.interested_vehs[mode][i]
                    mask = self.mask_mode[mode][i]
                    veh_x = veh['x']
                    veh_y = veh['y']
                    veh_phi = veh['phi']
                    veh_l = veh['l']
                    veh_w = veh['w']
                    veh_type = veh['type']
                    task2color = {'left': 'b', 'straight': 'c', 'right': 'm'}

                    if is_in_plot_area(veh_x, veh_y):
                        plot_phi_line(veh_type, veh_x, veh_y, veh_phi, 'black')
                        task = MODE2TASK[mode]
                        color = task2color[task]
                        draw_rotate_rec(veh_x, veh_y, veh_phi, veh_l, veh_w, color, linestyle=':')
                        plt.text(veh_x, veh_y, str(mask)[0])

            # plot_interested person
            for mode, num in self.person_mode_dict.items():
                for i in range(num):
                    veh = self.interested_vehs[mode][i]
                    mask = self.mask_mode[mode][i]
                    veh_x = veh['x']
                    veh_y = veh['y']
                    veh_phi = veh['phi']
                    veh_l = veh['l']
                    veh_w = veh['w']
                    veh_type = veh['type']
                    task2color = {'left': 'b', 'straight': 'c', 'right': 'm'}

                    if is_in_plot_area(veh_x, veh_y):
                        plot_phi_line(veh_type, veh_x, veh_y, veh_phi, 'black')
                        task = MODE2TASK[mode]
                        color = task2color[task]
                        draw_rotate_rec(veh_x, veh_y, veh_phi, veh_l, veh_w, color, linestyle=':')
                        plt.text(veh_x, veh_y, str(mask)[0])

            # plot own car
            # dict(v_x=ego_dict['v_x'],
            #      v_y=ego_dict['v_y'],
            #      r=ego_dict['r'],
            #      x=ego_dict['x'],
            #      y=ego_dict['y'],
            #      phi=ego_dict['phi'],
            #      l=ego_dict['l'],
            #      w=ego_dict['w'],
            #      Corner_point=self.cal_corner_point_of_ego_car(ego_dict)
            #      alpha_f_bound=alpha_f_bound,
            #      alpha_r_bound=alpha_r_bound,
            #      r_bound=r_bound)

            ego_v_x = self.ego_dynamics['v_x']
            ego_v_y = self.ego_dynamics['v_y']
            ego_r = self.ego_dynamics['r']
            ego_x = self.ego_dynamics['x']
            ego_y = self.ego_dynamics['y']
            ego_phi = self.ego_dynamics['phi']
            ego_l = self.ego_dynamics['l']
            ego_w = self.ego_dynamics['w']
            ego_alpha_f = self.ego_dynamics['alpha_f']
            ego_alpha_r = self.ego_dynamics['alpha_r']
            alpha_f_bound = self.ego_dynamics['alpha_f_bound']
            alpha_r_bound = self.ego_dynamics['alpha_r_bound']
            r_bound = self.ego_dynamics['r_bound']

            plot_phi_line('self_car', ego_x, ego_y, ego_phi, 'red')
            draw_rotate_rec(ego_x, ego_y, ego_phi, ego_l, ego_w, 'red')

            # plot future data
            tracking_info = self.obs[self.ego_info_dim:self.ego_info_dim + self.track_info_dim]
            future_path = self.obs[self.ego_info_dim + self.track_info_dim: self.ego_info_dim + self.track_info_dim +
                                                                            self.per_path_info_dim * self.num_future_data]
            for i in range(self.num_future_data):
                delta_x, delta_y, delta_phi, exp_v = future_path[i*self.per_path_info_dim: (i+1)*self.per_path_info_dim]
                path_x, path_y, path_phi = ego_x+delta_x, ego_y+delta_y, ego_phi-delta_phi
                plt.plot(path_x, path_y, 'g.')
                plot_phi_line('self_car', path_x, path_y, path_phi, 'g')

            delta_, _, _ = tracking_info[:3]
            ax.plot(self.ref_path.path[0], self.ref_path.path[1], color='g')
            indexs, points = self.ref_path.find_closest_point(np.array([ego_x], np.float32), np.array([ego_y],np.float32))
            path_x, path_y, path_phi, path_v= points[0][0], points[1][0], points[2][0], points[3][0]
            plt.plot(path_x, path_y, 'g.')
            delta_x, delta_y, delta_phi = ego_x - path_x, ego_y - path_y, ego_phi - path_phi

            # plot real time traj
            color = ['blue', 'coral', 'darkcyan', 'pink']
            for i, item in enumerate(self.ref_path.path_list['green']):
                if i == self.ref_path.ref_index:
                    plt.plot(item[0], item[1], color=color[i], alpha=1.0)
                else:
                    plt.plot(item[0], item[1], color=color[i], alpha=0.3)
                    # indexs, points = item.find_closest_point(np.array([ego_x], np.float32), np.array([ego_y], np.float32))
                    # path_x, path_y, path_phi = points[0][0], points[1][0], points[2][0]
                    # plt.plot(path_x, path_y,  color=color[i])

            # for j, item_point in enumerate(self.real_path.feature_points_all):
            #     for k in range(len(item_point)):
            #         plt.scatter(item_point[k][0], item_point[k][1], c='g')

            # text
            text_x, text_y_start = -110, 60
            ge = iter(range(0, 1000, 4))
            plt.text(text_x, text_y_start - next(ge), 'ego_x: {:.2f}m'.format(ego_x))
            plt.text(text_x, text_y_start - next(ge), 'ego_y: {:.2f}m'.format(ego_y))
            plt.text(text_x, text_y_start - next(ge), 'path_x: {:.2f}m'.format(path_x))
            plt.text(text_x, text_y_start - next(ge), 'path_y: {:.2f}m'.format(path_y))
            plt.text(text_x, text_y_start - next(ge), 'delta_: {:.2f}m'.format(delta_))
            plt.text(text_x, text_y_start - next(ge), 'delta_x: {:.2f}m'.format(delta_x))
            plt.text(text_x, text_y_start - next(ge), 'delta_y: {:.2f}m'.format(delta_y))
            plt.text(text_x, text_y_start - next(ge), r'ego_phi: ${:.2f}\degree$'.format(ego_phi))
            plt.text(text_x, text_y_start - next(ge), r'path_phi: ${:.2f}\degree$'.format(path_phi))
            plt.text(text_x, text_y_start - next(ge), r'delta_phi: ${:.2f}\degree$'.format(delta_phi))

            plt.text(text_x, text_y_start - next(ge), 'v_x: {:.2f}m/s'.format(ego_v_x))
            plt.text(text_x, text_y_start - next(ge), 'exp_v: {:.2f}m/s'.format(path_v))
            plt.text(text_x, text_y_start - next(ge), 'v_y: {:.2f}m/s'.format(ego_v_y))
            plt.text(text_x, text_y_start - next(ge), 'yaw_rate: {:.2f}rad/s'.format(ego_r))
            plt.text(text_x, text_y_start - next(ge), 'yaw_rate bound: [{:.2f}, {:.2f}]'.format(-r_bound, r_bound))

            plt.text(text_x, text_y_start - next(ge), r'$\alpha_f$: {:.2f} rad'.format(ego_alpha_f))
            plt.text(text_x, text_y_start - next(ge), r'$\alpha_f$ bound: [{:.2f}, {:.2f}] '.format(-alpha_f_bound,
                                                                                                        alpha_f_bound))
            plt.text(text_x, text_y_start - next(ge), r'$\alpha_r$: {:.2f} rad'.format(ego_alpha_r))
            plt.text(text_x, text_y_start - next(ge), r'$\alpha_r$ bound: [{:.2f}, {:.2f}] '.format(-alpha_r_bound,
                                                                                                        alpha_r_bound))
            if self.action is not None:
                steer, a_x = self.action[0], self.action[1]
                plt.text(text_x, text_y_start - next(ge), r'steer: {:.2f}rad (${:.2f}\degree$)'.format(steer, steer * 180 / np.pi))
                plt.text(text_x, text_y_start - next(ge), 'a_x: {:.2f}m/s^2'.format(a_x))

            text_x, text_y_start = 80, 60
            ge = iter(range(0, 1000, 4))

            # done info
            plt.text(text_x, text_y_start - next(ge), 'done info: {}'.format(self.done_type))

            # reward info
            if self.reward_info is not None:
                for key, val in self.reward_info.items():
                    plt.text(text_x, text_y_start - next(ge), '{}: {:.4f}'.format(key, val))

            # indicator for trajectory selection
            # text_x, text_y_start = -25, -65
            # ge = iter(range(0, 1000, 6))
            # if traj_return is not None:
            #     for i, value in enumerate(traj_return):
            #         if i==path_index:
            #             plt.text(text_x, text_y_start-next(ge), 'track_error={:.4f}, collision_risk={:.4f}'.format(value[0], value[1]), fontsize=14, color=color[i], fontstyle='italic')
            #         else:
            #             plt.text(text_x, text_y_start-next(ge), 'track_error={:.4f}, collision_risk={:.4f}'.format(value[0], value[1]), fontsize=12, color=color[i], fontstyle='italic')

            plt.show()
            plt.pause(0.001)

    def set_traj(self, trajectory):
        """set the real trajectory to reconstruct observation"""
        self.ref_path = trajectory


def test_end2end():
    env = CrossroadEnd2endMix(num_future_data=1)
    obs = env.reset()
    env_model = EnvironmentModel(training_task=env.training_task, num_future_data=env.num_future_data)
    i = 0
    while i < 100000:
        for j in range(200):
            i += 1
            # action=2*np.random.random(2)-1
            if obs[4] < -18:
                action = np.array([0, 1], dtype=np.float32)
            elif obs[3] <= -18:
                action = np.array([0, 0], dtype=np.float32)
            else:
                action = np.array([0.2, 0.33], dtype=np.float32)
            obs, reward, done, info = env.step(action)
            obses, actions = obs[np.newaxis, :], action[np.newaxis, :]
            # extract infos for each kind of participants
            ego_dim = env.ego_info_dim + env.track_info_dim + env.num_future_data * env.per_path_info_dim + env.light_info_dim + env.task_info_dim + env.ref_info_dim
            obses_ego = obses[:, env.ego_info_dim + env.track_info_dim + env.num_future_data * env.per_path_info_dim:ego_dim]
            print(obses_ego)
            # obses_others = obses[:, ego_dim:]
            # env_model.reset(np.tile(obses_ego, (2, 1)), np.tile(obses_others, (2, 1)), [env.ref_path.ref_index, random.randint(0, 2)])
            # env_model.mode = 'training'
            # for _ in range(10):
            #     obses_ego, obses_others, rewards, punish_term_for_training, \
            #         real_punish_term, veh2veh4real, veh2road4real, veh2bike4real, veh2person4real = env_model.rollout_out(np.tile(actions, (2, 1)))
            # print(obses_ego.shape, obses_others.shape)
            env.render()
            # if done:
            #     break
        obs = env.reset()
        # env_model = EnvironmentModel(training_task=env.training_task, num_future_data=env.num_future_data)
        env.render()


if __name__ == '__main__':
    test_end2end()