#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2020/6/10
# @Author  : Yang Guan (Tsinghua Univ.)
# @FileName: buffer.py
# =====================================

import logging
import random

import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class ReplayBufferWithAttention(object):
    def __init__(self, args, buffer_id):
        self.args = args
        self.buffer_id = buffer_id
        self._storage = []
        self._maxsize = self.args.max_buffer_size
        self._next_idx = 0
        self.replay_starts = self.args.replay_starts
        self.replay_batch_size = self.args.replay_batch_size
        self.stats = {}
        self.replay_times = 0
        logger.info('Buffer initialized')

    def get_stats(self):
        self.stats.update(dict(storage=len(self._storage)))
        return self.stats

    def __len__(self):
        return len(self._storage)

    def add(self, obs, act, reward, obs_tp1, done, future_n_point, mask):
        data = (obs, act, reward, obs_tp1, done, future_n_point, mask)
        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes):
        obses, acts, rewards, obses_tp1, dones, future_n_points, masks = [], [], [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            obs, act, reward, obs_tp1, done, future_n_point, mask = data
            obses.append(np.array(obs, copy=False))
            acts.append(np.array(act, copy=False))
            rewards.append(reward)
            obses_tp1.append(np.array(obs_tp1, copy=False))
            dones.append(done)
            future_n_points.append(future_n_point)
            masks.append(np.array(mask, copy=False))

        return np.array(obses, dtype=np.float32), np.array(acts, dtype=np.float32),\
               np.array(rewards, dtype=np.float32), np.array(obses_tp1, dtype=np.float32), \
               np.array(dones, dtype=np.float32), np.array(future_n_points, dtype=np.float32), \
               np.array(masks, dtype=np.float32)

    def sample_idxes(self, batch_size):
        return np.array([random.randint(0, len(self._storage) - 1) for _ in range(batch_size)], dtype=np.int32)

    def sample_with_idxes(self, idxes):
        return list(self._encode_sample(idxes)) + [idxes,]

    def sample(self, batch_size):
        idxes = self.sample_idxes(batch_size)
        return self.sample_with_idxes(idxes)

    def add_batch(self, batch):
        for trans in batch:
            self.add(*trans)

    def replay(self):
        if len(self._storage) < self.replay_starts:
            return None
        if self.buffer_id == 1 and self.replay_times % self.args.buffer_log_interval == 0:
            logger.info('Buffer info: {}'.format(self.get_stats()))

        self.replay_times += 1
        return self.sample(self.replay_batch_size)
