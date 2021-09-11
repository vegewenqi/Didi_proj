#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2020/12/12
# @Author  : Yang Guan; Yangang Ren (Tsinghua Univ.)
# @FileName: hier_decision.py
# =====================================

import datetime
import shutil
import time
import json
import os
import heapq
from math import cos, sin, pi

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from env_build.dynamics_and_models import EnvironmentModel, ReferencePath
from env_build.endtoend import CrossroadEnd2endMixPiFix
from env_build.endtoend_env_utils import rotate_coordination, CROSSROAD_SIZE, LANE_WIDTH, LANE_NUMBER, MODE2TASK
from multi_path_generator import MultiPathGenerator
from env_build.utils.load_policy import LoadPolicy
from env_build.utils.misc import TimerStat
from env_build.utils.recorder import Recorder


class HierarchicalDecision(object):
    def __init__(self, task, train_exp_dir, ite, logdir=None):
        self.task = task
        self.policy = LoadPolicy('../utils/models/{}/{}'.format(task, train_exp_dir), ite)
        self.args = self.policy.args
        self.env = CrossroadEnd2endMixPiFix(training_task=self.task, mode='testing')
        self.model = EnvironmentModel(self.task, mode='selecting')
        self.recorder = Recorder()
        self.episode_counter = -1
        self.step_counter = -1
        self.obs = None
        self.stg = MultiPathGenerator()
        self.step_timer = TimerStat()
        self.ss_timer = TimerStat()
        self.logdir = logdir
        if self.logdir is not None:
            config = dict(task=task, train_exp_dir=train_exp_dir, ite=ite)
            with open(self.logdir + '/config.json', 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=4)
        self.fig = plt.figure(figsize=(8, 8))
        plt.ion()
        self.hist_posi = []
        self.old_index = 0
        self.path_list = self.stg.generate_path(self.task)
        # ------------------build graph for tf.function in advance-----------------------
        for i in range(LANE_NUMBER):
            obs = self.env.reset()[np.newaxis, :]
            obs_ego, obs_bike, obs_person, obs_veh = self.split_obses(obs)
            self.is_safe(obs_ego, obs_bike, obs_person, obs_veh, i)
        obs = self.env.reset()[np.newaxis, :]
        obs_ego, obs_bike, obs_person, obs_veh = self.split_obses(obs)
        obs_ego_with_specific_shape = np.tile(obs_ego, (LANE_NUMBER, 1))
        obs_bike_with_specific_shape = np.tile(obs_bike, (LANE_NUMBER, 1))
        obs_person_with_specific_shape = np.tile(obs_person, (LANE_NUMBER, 1))
        obs_veh_with_specific_shape = np.tile(obs_veh, (LANE_NUMBER, 1))

        self.policy.run_batch(obs_ego_with_specific_shape, obs_bike_with_specific_shape, obs_person_with_specific_shape, obs_veh_with_specific_shape)
        self.policy.obj_value_batch(obs_ego_with_specific_shape, obs_bike_with_specific_shape, obs_person_with_specific_shape, obs_veh_with_specific_shape)
        # ------------------build graph for tf.function in advance-----------------------
        self.reset()

    def reset(self,):
        self.obs = self.env.reset()
        self.recorder.reset()
        self.old_index = 0
        self.hist_posi = []
        if self.logdir is not None:
            self.episode_counter += 1
            os.makedirs(self.logdir + '/episode{}/figs'.format(self.episode_counter))
            self.step_counter = -1
            self.recorder.save(self.logdir)
            # if self.episode_counter >= 1:
                # select_and_rename_snapshots_of_an_episode(self.logdir, self.episode_counter-1, 12)
                # self.recorder.plot_and_save_ith_episode_curves(self.episode_counter-1,
                #                                                self.logdir + '/episode{}/figs'.format(self.episode_counter-1),
                #                                                isshow=False)
        return self.obs

    # @tf.function
    # def is_safe(self, obs, path_index):
    #     self.model.ref_path.set_path(path_index)
    #     action = self.policy.run_batch(obs)
    #     veh2veh4real = self.model.ss(obs, action, lam=0.1)
    #     return False if veh2veh4real[0] > 0 else True

    @tf.function
    def is_safe(self, obs_ego, obs_bike, obs_person, obs_veh, path_index):
        self.model.add_traj(obs_ego, obs_bike, obs_person, obs_veh, path_index)
        punish = 0.
        for step in range(5):
            action = self.policy.run_batch(obs_ego, obs_bike, obs_person, obs_veh)
            obs_ego, obs_bike, obs_person, obs_veh, _, _, _, veh2veh4real, veh2road4real, veh2bike4real, veh2person4real \
                = self.model.rollout_out(action)
            punish += veh2veh4real[0] + veh2bike4real[0] + veh2person4real[0]
        return False if punish > 0 else True

    def safe_shield(self, real_obs, path_index):
        action_safe_set = [[[0., -1.]]]
        real_obs = tf.convert_to_tensor(real_obs[np.newaxis, :], dtype=tf.float32)
        obs_ego, obs_bike, obs_person, obs_veh = self.split_obses(real_obs.numpy())
        if not self.is_safe(obs_ego, obs_bike, obs_person, obs_veh, path_index):
            print('SAFETY SHIELD STARTED!')
            return np.array(action_safe_set[0], dtype=np.float32).squeeze(0), True
        else:
            return self.policy.run_batch(obs_ego, obs_bike, obs_person, obs_veh).numpy()[0], False

    def step(self):
        self.step_counter += 1
        with self.step_timer:
            obs_list = []
            # select optimal path
            for path in self.path_list:
                self.env.set_traj(path)
                obs_list.append(self.env._get_obs())
            all_obs = tf.stack(obs_list, axis=0).numpy()
            obs_ego, obs_bike, obs_person, obs_veh = self.split_obses(all_obs)
            path_values = self.policy.obj_value_batch(obs_ego, obs_bike, obs_person, obs_veh).numpy()
            old_value = path_values[self.old_index]
            # value is to approximate (- sum of reward)
            new_index, new_value = int(np.argmin(path_values)), min(path_values)
            # rule for equal traj value
            path_index_error = []
            if self.step_counter % 3 == 0:
                if heapq.nsmallest(2, path_values)[0] == heapq.nsmallest(2, path_values)[1]:
                    for i in range(len(path_values)):
                        if path_values[i] == min(path_values):
                            index_error = abs(self.old_index - i)
                            path_index_error.append(index_error)
                    # new_index_new = min(path_index_error) + self.old_index if min(path_index_error) + self.old_index < 4 else self.old_index - min(path_index_error)
                    new_index_new = self.old_index - min(path_index_error) if self.old_index - min(path_index_error) > -1 else self.old_index + min(path_index_error)
                    new_value_new = path_values[new_index_new]
                    path_index = self.old_index if old_value - new_value_new < 0.1 else new_index_new
                else:
                    path_index = self.old_index if old_value - new_value < 0.1 else new_index
                self.old_index = path_index
            else:
                path_index = self.old_index
            self.env.set_traj(self.path_list[path_index])
            self.obs_real = obs_list[path_index]

            # obtain safe action
            with self.ss_timer:
                safe_action, is_ss = self.safe_shield(self.obs_real, path_index)
            print('ALL TIME:', self.step_timer.mean, 'ss', self.ss_timer.mean)
        self.render(self.path_list, path_values, path_index)
        self.recorder.record(self.obs_real, safe_action, self.step_timer.mean,
                             path_index, path_values, self.ss_timer.mean, is_ss)
        self.obs, r, done, info = self.env.step(safe_action)
        return done

    def render(self, traj_list, path_values, path_index):
        square_length = CROSSROAD_SIZE
        extension = 40
        lane_width = LANE_WIDTH
        light_line_width = 3
        dotted_line_style = '--'
        solid_line_style = '-'

        plt.cla()
        ax = plt.axes([-0.05, -0.05, 1.1, 1.1])
        for ax in self.fig.get_axes():
            ax.axis('off')
        ax.axis("equal")

        # ----------arrow--------------
        plt.arrow(lane_width / 2, -square_length / 2 - 10, 0, 5, color='orange')
        plt.arrow(lane_width / 2, -square_length / 2 - 10 + 5, -0.5, 0, color='orange', head_width=1)
        plt.arrow(lane_width * 1.5, -square_length / 2 - 10, 0, 4, color='orange', head_width=1)
        plt.arrow(lane_width * 2.5, -square_length / 2 - 10, 0, 5, color='orange')
        plt.arrow(lane_width * 2.5, -square_length / 2 - 10 + 5, 0.5, 0, color='orange', head_width=1)

        # ----------horizon--------------

        plt.plot([-square_length / 2 - extension, -square_length / 2], [0.3, 0.3], color='orange')
        plt.plot([-square_length / 2 - extension, -square_length / 2], [-0.3, -0.3], color='orange')
        plt.plot([square_length / 2 + extension, square_length / 2], [0.3, 0.3], color='orange')
        plt.plot([square_length / 2 + extension, square_length / 2], [-0.3, -0.3], color='orange')

        #
        for i in range(1, LANE_NUMBER + 1):
            linestyle = dotted_line_style if i < LANE_NUMBER else solid_line_style
            linewidth = 1 if i < LANE_NUMBER else 2
            plt.plot([-square_length / 2 - extension, -square_length / 2], [i * lane_width, i * lane_width],
                     linestyle=linestyle, color='black', linewidth=linewidth)
            plt.plot([square_length / 2 + extension, square_length / 2], [i * lane_width, i * lane_width],
                     linestyle=linestyle, color='black', linewidth=linewidth)
            plt.plot([-square_length / 2 - extension, -square_length / 2], [-i * lane_width, -i * lane_width],
                     linestyle=linestyle, color='black', linewidth=linewidth)
            plt.plot([square_length / 2 + extension, square_length / 2], [-i * lane_width, -i * lane_width],
                     linestyle=linestyle, color='black', linewidth=linewidth)

        for i in range(4, 5 + 1):
            linestyle = dotted_line_style if i < 5 else solid_line_style
            linewidth = 1 if i < 5 else 2
            plt.plot([-square_length / 2 - extension, -square_length / 2],
                     [3 * lane_width + (i - 3) * 2, 3 * lane_width + (i - 3) * 2],
                     linestyle=linestyle, color='black', linewidth=linewidth)
            plt.plot([square_length / 2 + extension, square_length / 2],
                     [3 * lane_width + (i - 3) * 2, 3 * lane_width + (i - 3) * 2],
                     linestyle=linestyle, color='black', linewidth=linewidth)
            plt.plot([-square_length / 2 - extension, -square_length / 2],
                     [-3 * lane_width - (i - 3) * 2, -3 * lane_width - (i - 3) * 2],
                     linestyle=linestyle, color='black', linewidth=linewidth)
            plt.plot([square_length / 2 + extension, square_length / 2],
                     [-3 * lane_width - (i - 3) * 2, -3 * lane_width - (i - 3) * 2],
                     linestyle=linestyle, color='black', linewidth=linewidth)

        # ----------vertical----------------
        plt.plot([0.3, 0.3], [-square_length / 2 - extension, -square_length / 2], color='orange')
        plt.plot([-0.3, -0.3], [-square_length / 2 - extension, -square_length / 2], color='orange')
        plt.plot([0.3, 0.3], [square_length / 2 + extension, square_length / 2], color='orange')
        plt.plot([-0.3, -0.3], [square_length / 2 + extension, square_length / 2], color='orange')

        #
        for i in range(1, LANE_NUMBER + 1):
            linestyle = dotted_line_style if i < LANE_NUMBER else solid_line_style
            linewidth = 1 if i < LANE_NUMBER else 2
            plt.plot([i * lane_width, i * lane_width], [-square_length / 2 - extension, -square_length / 2],
                     linestyle=linestyle, color='black', linewidth=linewidth)
            plt.plot([i * lane_width, i * lane_width], [square_length / 2 + extension, square_length / 2],
                     linestyle=linestyle, color='black', linewidth=linewidth)
            plt.plot([-i * lane_width, -i * lane_width], [-square_length / 2 - extension, -square_length / 2],
                     linestyle=linestyle, color='black', linewidth=linewidth)
            plt.plot([-i * lane_width, -i * lane_width], [square_length / 2 + extension, square_length / 2],
                     linestyle=linestyle, color='black', linewidth=linewidth)

        for i in range(4, 5 + 1):
            linestyle = dotted_line_style if i < 5 else solid_line_style
            linewidth = 1 if i < 5 else 2
            plt.plot([3 * lane_width + (i - 3) * 2, 3 * lane_width + (i - 3) * 2],
                     [-square_length / 2 - extension, -square_length / 2],
                     linestyle=linestyle, color='black', linewidth=linewidth)
            plt.plot([3 * lane_width + (i - 3) * 2, 3 * lane_width + (i - 3) * 2],
                     [square_length / 2 + extension, square_length / 2],
                     linestyle=linestyle, color='black', linewidth=linewidth)
            plt.plot([-3 * lane_width - (i - 3) * 2, -3 * lane_width - (i - 3) * 2],
                     [-square_length / 2 - extension, -square_length / 2],
                     linestyle=linestyle, color='black', linewidth=linewidth)
            plt.plot([-3 * lane_width - (i - 3) * 2, -3 * lane_width - (i - 3) * 2],
                     [square_length / 2 + extension, square_length / 2],
                     linestyle=linestyle, color='black', linewidth=linewidth)

        v_light = self.env.v_light
        if v_light == 0 or v_light == 1:
            v_color, h_color = 'green', 'red'
        elif v_light == 2:
            v_color, h_color = 'orange', 'red'
        elif v_light == 3 or v_light == 4:
            v_color, h_color = 'red', 'green'
        else:
            v_color, h_color = 'red', 'orange'

        plt.plot([0, (LANE_NUMBER - 1) * lane_width], [-square_length / 2, -square_length / 2],
                 color=v_color, linewidth=light_line_width)
        plt.plot([(LANE_NUMBER - 1) * lane_width, LANE_NUMBER * lane_width], [-square_length / 2, -square_length / 2],
                 color='green', linewidth=light_line_width)

        plt.plot([-LANE_NUMBER * lane_width, -(LANE_NUMBER - 1) * lane_width], [square_length / 2, square_length / 2],
                 color='green', linewidth=light_line_width)
        plt.plot([-(LANE_NUMBER - 1) * lane_width, 0], [square_length / 2, square_length / 2],
                 color=v_color, linewidth=light_line_width)

        plt.plot([-square_length / 2, -square_length / 2], [0, -(LANE_NUMBER - 1) * lane_width],
                 color=h_color, linewidth=light_line_width)
        plt.plot([-square_length / 2, -square_length / 2], [-(LANE_NUMBER - 1) * lane_width, -LANE_NUMBER * lane_width],
                 color='green', linewidth=light_line_width)

        plt.plot([square_length / 2, square_length / 2], [(LANE_NUMBER - 1) * lane_width, 0],
                 color=h_color, linewidth=light_line_width)
        plt.plot([square_length / 2, square_length / 2], [LANE_NUMBER * lane_width, (LANE_NUMBER - 1) * lane_width],
                 color='green', linewidth=light_line_width)

        # ----------Oblique--------------

        plt.plot([LANE_NUMBER * lane_width + 4, square_length / 2],
                 [-square_length / 2, -LANE_NUMBER * lane_width - 4],
                 color='black', linewidth=2)
        plt.plot([LANE_NUMBER * lane_width + 4, square_length / 2],
                 [square_length / 2, LANE_NUMBER * lane_width + 4],
                 color='black', linewidth=2)
        plt.plot([-LANE_NUMBER * lane_width - 4, -square_length / 2],
                 [-square_length / 2, -LANE_NUMBER * lane_width - 4],
                 color='black', linewidth=2)
        plt.plot([-LANE_NUMBER * lane_width - 4, -square_length / 2],
                 [square_length / 2, LANE_NUMBER * lane_width + 4],
                 color='black', linewidth=2)

        # ----------人行横道--------------
        jj = 3.5
        for ii in range(23):
            if ii <= 3:
                continue
            ax.add_patch(plt.Rectangle((-square_length / 2 + jj + ii * 1.6, -square_length / 2 + 0.5), 0.8, 4,
                                       color='lightgray', alpha=0.5))
            ii += 1
        for ii in range(23):
            if ii <= 3:
                continue
            ax.add_patch(plt.Rectangle((-square_length / 2 + jj + ii * 1.6, square_length / 2 - 0.5 - 4), 0.8, 4,
                                       color='lightgray', alpha=0.5))
            ii += 1
        for ii in range(23):
            if ii <= 3:
                continue
            ax.add_patch(
                plt.Rectangle((-square_length / 2 + 0.5, square_length / 2 - jj - 0.8 - ii * 1.6), 4, 0.8,
                              color='lightgray',
                              alpha=0.5))
            ii += 1
        for ii in range(23):
            if ii <= 3:
                continue
            ax.add_patch(
                plt.Rectangle((square_length / 2 - 0.5 - 4, square_length / 2 - jj - 0.8 - ii * 1.6), 4, 0.8,
                              color='lightgray',
                              alpha=0.5))
            ii += 1

        def is_in_plot_area(x, y, tolerance=5):
            if -square_length / 2 - extension + tolerance < x < square_length / 2 + extension - tolerance and \
                    -square_length / 2 - extension + tolerance < y < square_length / 2 + extension - tolerance:
                return True
            else:
                return False

        def draw_rotate_rec(x, y, a, l, w, c):
            bottom_left_x, bottom_left_y, _ = rotate_coordination(-l / 2, w / 2, 0, -a)
            ax.add_patch(plt.Rectangle((x + bottom_left_x, y + bottom_left_y), w, l, edgecolor=c,
                                       facecolor='white', angle=-(90 - a), zorder=50))

        def draw_bike(x, y, a, l, w, c):
            bottom_left_x1, bottom_left_y1, _ = rotate_coordination(-l / 2, w / 4, 0, -a)
            ax.add_patch(plt.Rectangle((x + bottom_left_x1, y + bottom_left_y1), w/2, l, edgecolor=c,
                                       facecolor=c, angle=-(90 - a), zorder=50))
            bottom_left_x2, bottom_left_y2, _ = rotate_coordination(-l / 4, w / 2, 0, -a)
            ax.add_patch(plt.Rectangle((x + bottom_left_x2, y + bottom_left_y2), w, l/4, edgecolor=c,
                                       facecolor=c, angle=-(90 - a), zorder=50))

        def plot_phi_line(type, x, y, phi, color):
            if type in ['bicycle_1', 'bicycle_2', 'bicycle_3']:
                line_length = 1.5
            elif type == 'DEFAULT_PEDTYPE':
                line_length = 1
            else:
                line_length = 3
            x_forw, y_forw = x + line_length * cos(phi*pi/180.),\
                             y + line_length * sin(phi*pi/180.)
            plt.plot([x, x_forw], [y, y_forw], color=color, linewidth=0.5)

        # plot cars
        for veh in self.env.all_vehicles:
            veh_x = veh['x']
            veh_y = veh['y']
            veh_phi = veh['phi']
            veh_l = veh['l']
            veh_w = veh['w']
            veh_type = veh['type']
            if is_in_plot_area(veh_x, veh_y):
                if veh_type in ['bicycle_1', 'bicycle_2', 'bicycle_3']:
                    veh_color = 'steelblue'
                    draw_bike(veh_x, veh_y, veh_phi, veh_l, veh_w, veh_color)
                elif veh_type == 'DEFAULT_PEDTYPE':
                    veh_color = 'purple'
                    draw_rotate_rec(veh_x, veh_y, veh_phi, veh_l, veh_w, veh_color)
                else:
                    veh_color = 'black'
                    draw_rotate_rec(veh_x, veh_y, veh_phi, veh_l, veh_w, veh_color)
                plot_phi_line(veh_type, veh_x, veh_y, veh_phi, veh_color)


        # plot_interested vehs
        for mode, num in self.env.veh_mode_dict.items():
            for i in range(num):
                veh = self.env.interested_vehs[mode][i]
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
                    draw_rotate_rec(veh_x, veh_y, veh_phi, veh_l, veh_w, color)

        # plot_interested bicycle
        for mode, num in self.env.bicycle_mode_dict.items():
            for i in range(num):
                veh = self.env.interested_vehs[mode][i]
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
                    draw_bike(veh_x, veh_y, veh_phi, veh_l, veh_w, color)

        # plot_interested person
        for mode, num in self.env.person_mode_dict.items():
            for i in range(num):
                veh = self.env.interested_vehs[mode][i]
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
                    draw_rotate_rec(veh_x, veh_y, veh_phi, veh_l, veh_w, color)

        ego_v_x = self.env.ego_dynamics['v_x']
        ego_v_y = self.env.ego_dynamics['v_y']
        ego_r = self.env.ego_dynamics['r']
        ego_x = self.env.ego_dynamics['x']
        ego_y = self.env.ego_dynamics['y']
        ego_phi = self.env.ego_dynamics['phi']
        ego_l = self.env.ego_dynamics['l']
        ego_w = self.env.ego_dynamics['w']
        ego_alpha_f = self.env.ego_dynamics['alpha_f']
        ego_alpha_r = self.env.ego_dynamics['alpha_r']
        alpha_f_bound = self.env.ego_dynamics['alpha_f_bound']
        alpha_r_bound = self.env.ego_dynamics['alpha_r_bound']
        r_bound = self.env.ego_dynamics['r_bound']

        plot_phi_line('self_car', ego_x, ego_y, ego_phi, 'fuchsia')
        draw_rotate_rec(ego_x, ego_y, ego_phi, ego_l, ego_w, 'fuchsia')
        self.hist_posi.append((ego_x, ego_y))

        # plot history
        xs = [pos[0] for pos in self.hist_posi]
        ys = [pos[1] for pos in self.hist_posi]
        plt.scatter(np.array(xs), np.array(ys), color='fuchsia', alpha=0.2)

        # plot future data
        tracking_info = self.obs[self.env.ego_info_dim:self.env.ego_info_dim + self.env.per_tracking_info_dim * (self.env.num_future_data+1)]
        future_path = tracking_info[self.env.per_tracking_info_dim:]
        for i in range(self.env.num_future_data):
            delta_x, delta_y, delta_phi = future_path[i*self.env.per_tracking_info_dim:
                                                      (i+1)*self.env.per_tracking_info_dim]
            path_x, path_y, path_phi = ego_x+delta_x, ego_y+delta_y, ego_phi-delta_phi
            plt.plot(path_x, path_y, 'g.')
            plot_phi_line('self_car', path_x, path_y, path_phi, 'g')

        delta_, _, _ = tracking_info[:3]
        indexs, points = self.env.ref_path.find_closest_point(np.array([ego_x], np.float32), np.array([ego_y], np.float32))
        path_x, path_y, path_phi = points[0][0], points[1][0], points[2][0]
        # plt.plot(path_x, path_y, 'g.')
        delta_x, delta_y, delta_phi = ego_x - path_x, ego_y - path_y, ego_phi - path_phi

        # plot real time traj
        color = ['blue', 'coral', 'darkcyan']
        try:
            for i, item in enumerate(traj_list):
                if i == path_index:
                    plt.plot(item.path[0], item.path[1], color=color[i])
                else:
                    plt.plot(item.path[0], item.path[1], color=color[i], alpha=0.3)
                indexs, points = item.find_closest_point(np.array([ego_x], np.float32), np.array([ego_y], np.float32))
                path_x, path_y, path_phi = points[0][0], points[1][0], points[2][0]
                plt.plot(path_x, path_y, color=color[i])
        except Exception:
            pass

        # text
        text_x, text_y_start = -120, 60
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
        plt.text(text_x, text_y_start - next(ge), 'exp_v: {:.2f}m/s'.format(self.env.exp_v))
        plt.text(text_x, text_y_start - next(ge), 'v_y: {:.2f}m/s'.format(ego_v_y))
        plt.text(text_x, text_y_start - next(ge), 'yaw_rate: {:.2f}rad/s'.format(ego_r))
        plt.text(text_x, text_y_start - next(ge), 'yaw_rate bound: [{:.2f}, {:.2f}]'.format(-r_bound, r_bound))

        plt.text(text_x, text_y_start - next(ge), r'$\alpha_f$: {:.2f} rad'.format(ego_alpha_f))
        plt.text(text_x, text_y_start - next(ge), r'$\alpha_f$ bound: [{:.2f}, {:.2f}] '.format(-alpha_f_bound,
                                                                                                alpha_f_bound))
        plt.text(text_x, text_y_start - next(ge), r'$\alpha_r$: {:.2f} rad'.format(ego_alpha_r))
        plt.text(text_x, text_y_start - next(ge), r'$\alpha_r$ bound: [{:.2f}, {:.2f}] '.format(-alpha_r_bound,
                                                                                                alpha_r_bound))
        if self.env.action is not None:
            steer, a_x = self.env.action[0], self.env.action[1]
            plt.text(text_x, text_y_start - next(ge), r'steer: {:.2f}rad (${:.2f}\degree$)'.format(steer, steer * 180 / np.pi))
            plt.text(text_x, text_y_start - next(ge), 'a_x: {:.2f}m/s^2'.format(a_x))

        text_x, text_y_start = 70, 60
        ge = iter(range(0, 1000, 4))

        # done info
        plt.text(text_x, text_y_start - next(ge), 'done info: {}'.format(self.env.done_type))

        # reward info
        if self.env.reward_info is not None:
            for key, val in self.env.reward_info.items():
                plt.text(text_x, text_y_start - next(ge), '{}: {:.4f}'.format(key, val))

        # indicator for trajectory selection
        text_x, text_y_start = 25, -30
        ge = iter(range(0, 1000, 6))
        if path_values is not None:
            for i, value in enumerate(path_values):
                if i == path_index:
                    plt.text(text_x, text_y_start - next(ge), 'Path cost={:.4f}'.format(value), fontsize=14,
                             color=color[i], fontstyle='italic')
                else:
                    plt.text(text_x, text_y_start - next(ge), 'Path cost={:.4f}'.format(value), fontsize=12,
                             color=color[i], fontstyle='italic')
        plt.show()
        plt.pause(0.001)
        if self.logdir is not None:
            plt.savefig(self.logdir + '/episode{}'.format(self.episode_counter) + '/step{}.pdf'.format(self.step_counter))

    def split_obses(self, obses):   # shape of obs is two-dimension
        start = 0;
        end = self.args.state_ego_dim + self.args.state_track_dim
        obs_ego = obses[:, start:end]
        start = end;
        end = start + self.args.state_bike_dim
        obs_bike = obses[:, start:end]
        start = end;
        end = start + self.args.state_person_dim
        obs_person = obses[:, start:end]
        start = end;
        end = start + self.args.state_veh_dim
        obs_veh = obses[:, start:end]

        obs_bike = np.reshape(obs_bike, [-1, self.args.per_bike_dim])
        obs_person = np.reshape(obs_person, [-1, self.args.per_person_dim])
        obs_veh = np.reshape(obs_veh, [-1, self.args.per_veh_dim])

        return obs_ego, obs_bike, obs_person, obs_veh

def plot_and_save_ith_episode_data(logdir, i):
    recorder = Recorder()
    recorder.load(logdir)
    save_dir = logdir + '/episode{}/figs'.format(i)
    os.makedirs(save_dir, exist_ok=True)
    recorder.plot_and_save_ith_episode_curves(i, save_dir, True)


def main():
    time_now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    logdir = './results/{time}'.format(time=time_now)
    os.makedirs(logdir)
    hier_decision = HierarchicalDecision('left', 'experiment-2021-08-09-17-07-28', 30000, logdir)
    # 'left', 'experiment-2021-03-15-16-39-00', 180000
    # 'straight', 'experiment-2021-03-15-19-16-13', 175000
    # 'right', 'experiment-2021-03-15-21-02-51', 195000

    for i in range(300):
        for _ in range(200):
            done = hier_decision.step()
            if done: break
        hier_decision.reset()


def plot_static_path():
    square_length = CROSSROAD_SIZE
    extension = 20
    lane_width = LANE_WIDTH
    light_line_width = 3
    dotted_line_style = '--'
    solid_line_style = '-'
    fig = plt.figure(figsize=(8, 8))
    ax = plt.axes([0, 0, 1, 1])
    for ax in fig.get_axes():
        ax.axis('off')
    ax.axis("equal")

    # ----------arrow--------------
    plt.arrow(lane_width / 2 , -square_length / 2 - 10, 0, 3, color='darkviolet')
    plt.arrow(lane_width / 2 , -square_length / 2 - 10 + 3, -0.5, 1.0, color='darkviolet', head_width=0.7)
    plt.arrow(lane_width * 1.5, -square_length / 2 - 10, 0, 4, color='darkviolet', head_width=0.7)
    plt.arrow(lane_width * 2.5, -square_length / 2 - 10, 0, 3, color='darkviolet')
    plt.arrow(lane_width * 2.5, -square_length / 2 - 10 + 3, 0.5, 1.0, color='darkviolet', head_width=0.7)

    # ----------horizon--------------

    plt.plot([-square_length / 2 - extension, -square_length / 2], [0.2, 0.2], color='orange')
    plt.plot([-square_length / 2 - extension, -square_length / 2], [-0.2, -0.2], color='orange')
    plt.plot([square_length / 2 + extension, square_length / 2], [0.2, 0.2], color='orange')
    plt.plot([square_length / 2 + extension, square_length / 2], [-0.2, -0.2], color='orange')

    for i in range(1, LANE_NUMBER + 1):
    # for i in range(1, 5+1):
        linestyle = dotted_line_style if i < LANE_NUMBER else solid_line_style
        linewidth = 1 if i < LANE_NUMBER else 1
        plt.plot([-square_length / 2 - extension, -square_length / 2], [i * lane_width, i * lane_width],
                 linestyle=linestyle, color='black', linewidth=linewidth)
        plt.plot([square_length / 2 + extension, square_length / 2], [i * lane_width, i * lane_width],
                 linestyle=linestyle, color='black', linewidth=linewidth)
        plt.plot([-square_length / 2 - extension, -square_length / 2], [-i * lane_width, -i * lane_width],
                 linestyle=linestyle, color='black', linewidth=linewidth)
        plt.plot([square_length / 2 + extension, square_length / 2], [-i * lane_width, -i * lane_width],
                 linestyle=linestyle, color='black', linewidth=linewidth)

    for i in range(4, 5+1):
        linestyle = dotted_line_style if i < 5 else solid_line_style
        linewidth = 1 if i < 5 else 2
        plt.plot([-square_length / 2 - extension, -square_length / 2], [3 * lane_width + (i-3)* 2, 3 * lane_width + (i-3)* 2],
                 linestyle=linestyle, color='black', linewidth=linewidth)
        plt.plot([square_length / 2 + extension, square_length / 2], [3 * lane_width + (i-3)* 2, 3 * lane_width + (i-3)* 2],
                 linestyle=linestyle, color='black', linewidth=linewidth)
        plt.plot([-square_length / 2 - extension, -square_length / 2], [-3 * lane_width - (i-3)* 2, -3 * lane_width - (i-3)* 2],
                 linestyle=linestyle, color='black', linewidth=linewidth)
        plt.plot([square_length / 2 + extension, square_length / 2], [-3 * lane_width - (i-3)* 2, -3 * lane_width - (i-3)* 2],
                 linestyle=linestyle, color='black', linewidth=linewidth)

    # ----------vertical----------------
    plt.plot([0.2, 0.2], [-square_length / 2 - extension, -square_length / 2], color='orange')
    plt.plot([-0.2, -0.2], [-square_length / 2 - extension, -square_length / 2], color='orange')
    plt.plot([0.2, 0.2], [square_length / 2 + extension, square_length / 2], color='orange')
    plt.plot([-0.2, -0.2], [square_length / 2 + extension, square_length / 2], color='orange')

    for i in range(1, LANE_NUMBER + 1):
        linestyle = dotted_line_style if i < LANE_NUMBER else solid_line_style
        linewidth = 1
        plt.plot([i * lane_width, i * lane_width], [-square_length / 2 - extension, -square_length / 2],
                 linestyle=linestyle, color='black', linewidth=linewidth)
        plt.plot([i * lane_width, i * lane_width], [square_length / 2 + extension, square_length / 2],
                 linestyle=linestyle, color='black', linewidth=linewidth)
        plt.plot([-i * lane_width, -i * lane_width], [-square_length / 2 - extension, -square_length / 2],
                 linestyle=linestyle, color='black', linewidth=linewidth)
        plt.plot([-i * lane_width, -i * lane_width], [square_length / 2 + extension, square_length / 2],
                 linestyle=linestyle, color='black', linewidth=linewidth)

    for i in range(4, 5+1):
        linestyle = dotted_line_style if i < 5 else solid_line_style
        linewidth = 1 if i < 5 else 2
        plt.plot([3 * lane_width + (i-3)* 2, 3 * lane_width + (i-3)* 2], [-square_length / 2 - extension, -square_length / 2],
                 linestyle=linestyle, color='black', linewidth=linewidth)
        plt.plot([3 * lane_width + (i-3)* 2, 3 * lane_width + (i-3)* 2], [square_length / 2 + extension, square_length / 2],
                 linestyle=linestyle, color='black', linewidth=linewidth)
        plt.plot([-3 * lane_width - (i-3)* 2, -3 * lane_width - (i-3)* 2], [-square_length / 2 - extension, -square_length / 2],
                 linestyle=linestyle, color='black', linewidth=linewidth)
        plt.plot([-3 * lane_width - (i-3)* 2, -3 * lane_width - (i-3)* 2], [square_length / 2 + extension, square_length / 2],
                 linestyle=linestyle, color='black', linewidth=linewidth)

    # stop line
    plt.plot([0, LANE_NUMBER * lane_width + 2 * 2], [-square_length / 2, -square_length / 2],
             color='black', linewidth=light_line_width, alpha=0.3)
    plt.plot([LANE_NUMBER * lane_width + 2 * 2, 0], [square_length / 2, square_length / 2],
             color='black', linewidth=light_line_width, alpha=0.3)
    plt.plot([-square_length / 2, -square_length / 2], [0, LANE_NUMBER * lane_width + 2 * 2],
             color='black', linewidth=light_line_width, alpha=0.3)
    plt.plot([square_length / 2, square_length / 2], [-LANE_NUMBER * lane_width - 2 * 2, 0],
             color='black', linewidth=light_line_width, alpha=0.3)

    # ----------Oblique--------------
    plt.plot([LANE_NUMBER * lane_width+4, square_length / 2], [-square_length / 2, -LANE_NUMBER * lane_width -4],
             color='black', linewidth=2)
    plt.plot([LANE_NUMBER * lane_width+4, square_length / 2], [square_length / 2, LANE_NUMBER * lane_width+4],
             color='black', linewidth=2)
    plt.plot([-LANE_NUMBER * lane_width-4, -square_length / 2], [-square_length / 2, -LANE_NUMBER * lane_width-4],
             color='black', linewidth=2)
    plt.plot([-LANE_NUMBER * lane_width-4, -square_length / 2], [square_length / 2, LANE_NUMBER * lane_width+4],
             color='black', linewidth=2)

    # 人行横道
    # for renxingdao in range(5):
    jj = 3.5
    for ii in range(23):
        if ii <= 3:
            continue
        ax.add_patch(plt.Rectangle((-square_length / 2 + jj + ii * 1.6, -square_length / 2 + 0.5), 0.8, 4, color='lightgray', alpha=0.5))

        # plt.plot([-square_length / 2 + jj + ii * 1.6, -square_length / 2 + jj + 0.8 + ii * 1.6],
        #          [-square_length / 2 + 0.5, -square_length / 2 + 0.5], color='silver', linewidth=1, alpha=0.5)
        # plt.plot([-square_length / 2 + jj + ii * 1.6, -square_length / 2 + jj + 0.8 + ii * 1.6],
        #          [-square_length / 2 + 0.5 +4 , -square_length / 2 + 0.5+4], color='silver', linewidth=1, alpha=0.5)
        # plt.plot([-square_length / 2 + jj + ii * 1.6, -square_length / 2 + jj + ii * 1.6],
        #          [-square_length / 2 + 0.5, -square_length / 2 + 0.5 +4], color='silver', linewidth=1, alpha=0.5)
        # plt.plot([-square_length / 2 + jj + 0.8+ ii*1.6, -square_length / 2 + jj + 0.8+ ii*1.6],
        #          [-square_length / 2 + 0.5, -square_length / 2 + 0.5 + 4], color='silver', linewidth=1, alpha=0.5)
        ii += 1
    for ii in range(23):
        if ii <= 3:
            continue
        ax.add_patch(plt.Rectangle((-square_length / 2 + jj + ii * 1.6, square_length / 2 - 0.5 - 4), 0.8, 4, color='lightgray',alpha=0.5))

        # plt.plot([-square_length / 2 + jj + ii * 1.6, -square_length / 2 + jj + 0.8 + ii * 1.6],
        #          [square_length / 2 - 0.5, square_length / 2 - 0.5], color='silver', linewidth=1, alpha=0.5)
        # plt.plot([-square_length / 2 + jj + ii * 1.6, -square_length / 2 + jj + 0.8 + ii * 1.6],
        #          [square_length / 2 - 0.5 -4 , square_length / 2 - 0.5-4], color='silver', linewidth=1, alpha=0.5)
        # plt.plot([-square_length / 2 + jj + ii * 1.6, -square_length / 2 + jj + ii * 1.6],
        #          [square_length / 2 - 0.5, square_length / 2 - 0.5 -4], color='silver', linewidth=1, alpha=0.5)
        # plt.plot([-square_length / 2 + jj + 0.8+ ii*1.6, -square_length / 2 + jj + 0.8+ ii*1.6],
        #          [square_length / 2 - 0.5, square_length / 2 - 0.5 - 4], color='silver', linewidth=1, alpha=0.5)
        ii += 1
    for ii in range(23):
        if ii <= 3:
            continue
        ax.add_patch(
            plt.Rectangle((-square_length / 2 + 0.5, square_length / 2 - jj - 0.8 - ii * 1.6), 4,0.8,  color='lightgray',
                          alpha=0.5))
        # plt.plot([-square_length / 2 + 0.5, -square_length / 2 + 0.5],
        #          [square_length / 2 - jj - ii * 1.6, square_length / 2 - jj - 0.8 - ii * 1.6], color='silver', linewidth=1, alpha=0.5)
        # plt.plot([-square_length / 2 + 0.5 +4, -square_length / 2 + 0.5+4],
        #          [square_length / 2 - jj - ii * 1.6, square_length / 2 - jj - 0.8 - ii * 1.6], color='silver', linewidth=1, alpha=0.5)
        # plt.plot([-square_length / 2 + 0.5, -square_length / 2 + 0.5 +4],
        #          [square_length / 2 - jj - ii * 1.6, square_length / 2 - jj - ii * 1.6], color='silver', linewidth=1, alpha=0.5)
        # plt.plot([-square_length / 2 + 0.5, -square_length / 2 + 0.5+4],
        #          [square_length / 2 - jj - 0.8 - ii * 1.6, square_length / 2 - jj - 0.8 - ii * 1.6], color='silver', linewidth=1, alpha=0.5)
        ii += 1
    for ii in range(23):
        if ii <= 3:
            continue
        ax.add_patch(
            plt.Rectangle((square_length / 2 - 0.5 -4, square_length / 2 - jj - 0.8 - ii * 1.6), 4, 0.8,
                          color='lightgray',
                          alpha=0.5))
        # plt.plot([square_length / 2 - 0.5, square_length / 2 - 0.5],
        #          [square_length / 2 - jj - ii * 1.6, square_length / 2 - jj - 0.8 - ii * 1.6], color='silver', linewidth=1, alpha=0.5)
        # plt.plot([square_length / 2 - 0.5 -4, square_length / 2 - 0.5-4],
        #          [square_length / 2 - jj - ii * 1.6, square_length / 2 - jj - 0.8 - ii * 1.6], color='silver', linewidth=1, alpha=0.5)
        # plt.plot([square_length / 2 - 0.5, square_length / 2 - 0.5 -4],
        #          [square_length / 2 - jj - ii * 1.6, square_length / 2 - jj - ii * 1.6], color='silver', linewidth=1, alpha=0.5)
        # plt.plot([square_length / 2 - 0.5, square_length / 2 - 0.5-4],
        #          [square_length / 2 - jj - 0.8 - ii * 1.6, square_length / 2 - jj - 0.8 - ii * 1.6], color='silver', linewidth=1, alpha=0.5)
        ii += 1


    for task in ['left', 'straight', 'right']:
        path = ReferencePath(task)
        path_list = path.path_list
        control_points = path.control_points
        color = ['royalblue', 'orangered', 'teal']

        for i, (path_x, path_y, _) in enumerate(path_list):
            plt.plot(path_x[600:-600], path_y[600:-600], color=color[i])
        for i, four_points in enumerate(control_points):
            for point in four_points:
                plt.scatter(point[0], point[1], color=color[i], s=20, alpha=0.7)
            plt.plot([four_points[0][0], four_points[1][0]], [four_points[0][1], four_points[1][1]], linestyle='--', color=color[i], alpha=0.5)
            plt.plot([four_points[1][0], four_points[2][0]], [four_points[1][1], four_points[2][1]], linestyle='--', color=color[i], alpha=0.5)
            plt.plot([four_points[2][0], four_points[3][0]], [four_points[2][1], four_points[3][1]], linestyle='--', color=color[i], alpha=0.5)

    plt.savefig('./multipath_planning.pdf')
    plt.show()


def select_and_rename_snapshots_of_an_episode(logdir, epinum, num):
    file_list = os.listdir(logdir + '/episode{}'.format(epinum))
    file_num = len(file_list) - 1
    intervavl = file_num // (num-1)
    start = file_num % (num-1)
    print(start, file_num, intervavl)
    selected = [start//2] + [start//2+intervavl*i-1 for i in range(1, num)]
    print(selected)
    if file_num > 0:
        for i, j in enumerate(selected):
            shutil.copyfile(logdir + '/episode{}/step{}.pdf'.format(epinum, j),
                            logdir + '/episode{}/figs/{}.pdf'.format(epinum, i))


if __name__ == '__main__':
    main()
    # plot_static_path()
    # plot_and_save_ith_episode_data('./results/good/2021-03-15-23-56-21', 0)
    # select_and_rename_snapshots_of_an_episode('./results/good/2021-03-15-23-56-21', 0, 12)


