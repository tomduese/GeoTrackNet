# coding: utf-8

# MIT License
# 
# Copyright (c) 2018 Duong Nguyen
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ==============================================================================

"""
Input pipelines script for Tensorflow graph.
This script is adapted from the original script of FIVO.
"""


import numpy as np
from math import radians, cos, sin, asin, sqrt
import sys
sys.path.append('..')
import os
import pickle
import tensorflow as tf


LAT, LON, SOG, COG, HEADING, ROT, NAV_STT, TIMESTAMP, MMSI = list(range(9))

# The default number of threads used to process data in parallel.
DEFAULT_PARALLELISM = 12

def create_dense_vect(msg,lat_bins = 300, lon_bins = 300, sog_bins = 30 ,cog_bins = 72):
    lat, lon, sog, cog = msg[0], msg[1], msg[2], msg[3]
    data_dim = lat_bins + lon_bins + sog_bins + cog_bins
    dense_vect = np.zeros(data_dim)
    dense_vect[int(lat*lat_bins)] = 1.0
    dense_vect[int(lon*lon_bins) + lat_bins] = 1.0
    dense_vect[int(sog*sog_bins) + lat_bins + lon_bins] = 1.0
    dense_vect[int(cog*cog_bins) + lat_bins + lon_bins + sog_bins] = 1.0
    return dense_vect

def sparse_AIS_to_dense(msgs_,num_timesteps, mmsis, time_start, time_end):
    # lat_bins = 200; lon_bins = 300; sog_bins = 30; cog_bins = 72
    msgs_[msgs_ == 1] = 0.99999
    dense_msgs = []
    for msg in msgs_:
        # lat_bins, lon_bins, sog_bins, cog_bins are from "create_AIS_dataset" scope 
        dense_msgs.append(create_dense_vect(msg,
                                            lat_bins = lat_bins,
                                            lon_bins = lon_bins,
                                            sog_bins = sog_bins,
                                            cog_bins = cog_bins))
    dense_msgs = np.array(dense_msgs)
    return dense_msgs, num_timesteps, mmsis, time_start, time_end

def create_AIS_dataset(dataset_path,
                       mean_path,
                       batch_size,
                       data_dim,
                       lat_bins,
                       lon_bins,
                       sog_bins,
                       cog_bins,
                       num_parallel_calls=DEFAULT_PARALLELISM,
                       shuffle=True,
                       repeat=True):
    total_bins = lat_bins+lon_bins+sog_bins+cog_bins
    


    # Load the data from disk.
    with tf.io.gfile.GFile(dataset_path, "rb") as f:
        raw_data = pickle.load(f)

    num_examples = len(raw_data)
    dirname = os.path.dirname(dataset_path)

    with open(mean_path,"rb") as f:
        mean = pickle.load(f)

    def aistrack_generator():
        for k in list(raw_data.keys()):
            tmp = raw_data[k][::2,[LAT,LON,SOG,COG]] # 10 min
            tmp[tmp == 1] = 0.99999
            yield tmp, len(tmp), raw_data[k][0,MMSI], raw_data[k][0,TIMESTAMP], raw_data[k][-1,TIMESTAMP]

    dataset = tf.data.Dataset.from_generator(
                              aistrack_generator,
                              output_types=(tf.float64, tf.int64, tf.int64, tf.float32, tf.float32))
            
    if repeat: dataset = dataset.repeat()
    if shuffle: dataset = dataset.shuffle(num_examples)             
              
    dataset = dataset.map(
        lambda msg_, num_timesteps, mmsis, time_start, time_end: tf.py_function(
            sparse_AIS_to_dense,
            [msg_, num_timesteps, mmsis, time_start, time_end],
            [tf.float64, tf.int64, tf.int64, tf.float32, tf.float32]),
        num_parallel_calls=num_parallel_calls)
              

    mean = tf.constant(mean, dtype=tf.float32, shape=[1, 1, mean.shape[0]])

    def process_AIS_batch(data, lengths, mmsis, time_start, time_end):
        """Create mean-centered and time-major next-step prediction Tensors."""
        data = tf.cast(tf.transpose(data, perm=[1, 0, 2]), dtype=tf.float32)
        lengths = tf.cast(lengths, dtype=tf.int32)
        mmsis = tf.cast(mmsis, dtype=tf.int32)
        targets = data

        # Mean center the inputs.
        inputs = data - mean

        # Shift the inputs one step forward in time. Also remove the last
        # timestep so that targets and inputs are the same length.
        inputs = tf.pad(inputs, [[1, 0], [0, 0], [0, 0]], mode="CONSTANT")[:-1]

        # Mask out unused timesteps.
        inputs *= tf.expand_dims(tf.transpose(tf.sequence_mask(lengths, dtype=inputs.dtype)), 2)

        return inputs, targets, lengths, mmsis, time_start, time_end

    dataset = dataset.map(process_AIS_batch, num_parallel_calls=num_parallel_calls)


    # dataset = dataset.prefetch(num_examples)
    dataset = dataset.prefetch(50)
    itr = iter(dataset)
    inputs, targets, lengths, mmsis, time_starts, time_ends = itr.get_next()
    return inputs, targets, mmsis, time_starts, time_ends, lengths, tf.constant(mean, dtype=tf.float32)

