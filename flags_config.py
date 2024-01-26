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
Flag configuration.
Adapted from the original script of FIVO.
"""

import os
import tensorflow as tf
import pickle
import math
import argparse




## Bretagne dataset
# LAT_MIN = 46.5
# LAT_MAX = 50.5
# LON_MIN = -8.0
# LON_MAX = -3.0

# ## Aruba
# LAT_MIN = 11.0
# LAT_MAX = 14.0
# LON_MIN = -71.0
# LON_MAX = -68.0

## Gulf of Mexico
"""
LAT_MIN = 26.5
LAT_MAX = 30.0
LON_MIN = -97.5
LON_MAX = -87
"""

SPEED_MAX = 30.0  # knots
FIG_DPI = 150

# Shared flags.
parser = argparse.ArgumentParser(description='Description of your program')
parser.add_argument('--mode', type=str, default='train',
                    help="The mode of the binary. Must be 'train', "
                         "'save_logprob', 'local_logprob', "
                         "'contrario_detection', 'visualisation', "
                         "'traj_reconstruction', or 'traj_speed'.")

# Model flags
parser.add_argument('--bound', type=str, default='elbo',
                    help='The bound to optimize. Can be "elbo", or "fivo".')
parser.add_argument('--latent_size', type=int, default=64,
                    help='The size of the latent state of the model.')
parser.add_argument('--log_dir', type=str, default='./chkpt',
                    help='The directory to keep checkpoints and summaries in.')

# Data and sampling flags
parser.add_argument('--batch_size', type=int, default=32,
                    help='Batch size.')
parser.add_argument('--num_samples', type=int, default=16,
                    help='The number of samples (or particles) for multisample algorithms.')
parser.add_argument('--ll_thresh', type=float, default=-17.47,
                    help='Log likelihood for anomaly detection.')

# Dataset flags
parser.add_argument('--dataset_dir', type=str, default='./data/central_med/preprocessed/',
                    help='Dataset directory')
parser.add_argument('--trainingset_name', type=str, default='ct_centralmed_train_track.pkl',
                    help='Path to load the trainingset from.')
parser.add_argument('--testset_name', type=str, default='ct_centralmed_valid_track.pkl',
                    help='Path to load the testset from.')
parser.add_argument('--split', type=str, default='train',
                    help='Split to evaluate the model on. Can be "train", "valid", or "test".')
parser.add_argument('--missing_data', type=bool, default=False,
                    help='If true, a part of input track will be deleted.')

# Model flags
parser.add_argument('--model', type=str, default='vrnn',
                    help='Model choice. Currently only "vrnn" is supported.')
parser.add_argument('--random_seed', type=int, default=None,
                    help='A random seed for seeding the TensorFlow graph.')


# Track flags
parser.add_argument('--interval_max', type=float, default=2*3600,
                    help='Maximum interval between two successive AIS messages (in second).')
parser.add_argument('--min_duration', type=int, default=4,
                    help='Min duration (hour) of a vessel track')

# Four-hot-encoding flags
parser.add_argument("--lat_min", type=float, default=30.0,
                    help="Lat min.")
parser.add_argument("--lat_max", type=float, default=39.0,
                    help="Lat max.")
parser.add_argument("--lon_min", type=float, default=10.0,
                    help="Lon min.")
parser.add_argument("--lon_max", type=float, default=21.0,
                    help="Lon max.")
parser.add_argument('--onehot_lat_reso', type=float, default=0.01,
                    help='Resolution of the lat one-hot vector (degree)')
parser.add_argument('--onehot_lon_reso', type=float, default=0.01,
                    help='Resolution of the lat one-hot vector (degree)')
parser.add_argument('--onehot_sog_reso', type=float, default=1,
                    help='Resolution of the SOG one-hot vector (knot)')
parser.add_argument('--onehot_cog_reso', type=float, default=5,
                    help='Resolution of the COG one-hot vector (degree)')

# A contrario detection flags
parser.add_argument('--cell_lat_reso', type=float, default=0.1,
                    help='Lat resolution of each small cell when applying local thresholding')
parser.add_argument('--cell_lon_reso', type=float, default=0.1,
                    help='Lon resolution of each small cell when applying local thresholding')

parser.add_argument('--contrario_eps', type=float, default=1e-9,
                    help='A contrario eps.')
parser.add_argument('--print_log', type=bool, default=False,
                    help='If true, print the current state of the program to the screen.')



# Training flags
parser.add_argument('--normalize_by_seq_len', type=bool, default=True,
                    help='If true, normalize the loss by the number of timesteps per sequence.')
parser.add_argument('--learning_rate', type=float, default=0.0003,
                    help='The learning rate for ADAM.')
parser.add_argument('--max_steps', type=int, default=80000,
                    help='The number of gradient update steps to train for.')
parser.add_argument('--summarize_every', type=int, default=100,
                    help='The number of steps between summaries.')

# Distributed training flags
parser.add_argument('--master', type=str, default='',
                    help='The BNS name of the TensorFlow master to use.')
parser.add_argument('--task', type=int, default=0,
                    help='Task id of the replica running the training.')
parser.add_argument('--ps_tasks', type=int, default=0,
                    help='Number of tasks in the ps job. If 0, no ps job is used.')
parser.add_argument('--stagger_workers', type=bool, default=True,
                    help='If true, bring one worker online every 1000 steps.')


# Fix tf >=1.8.0 flags bug
parser.add_argument('--f', type=str, default='', help='Kernel')
parser.add_argument('--data_dim', type=int, default=0, help='Data dimension')
parser.add_argument('--log_filename', type=str, default='', help='Log filename')
parser.add_argument('--logdir_name', type=str, default='', help='Log dir name')
parser.add_argument('--logdir', type=str, default='', help='Log directory')
parser.add_argument('--trainingset_path', type=str, default='', help='Training set path')
parser.add_argument('--testset_path', type=str, default='', help='Test set path')



parser.add_argument('--onehot_lon_bins', type=int, default=0,
                    help='Number of equal-width bins for the lat one-hot vector (degree)')
parser.add_argument('--onehot_sog_bins', type=int, default=1,
                    help='Number of equal-width bins for the SOG one-hot vector (knot)')
parser.add_argument('--onehot_cog_bins', type=int, default=5,
                    help='Number of equal-width bins for the COG one-hot vector (degree)')
parser.add_argument('--n_lat_cells', type=int, default=0,
                    help='Number of lat cells')
parser.add_argument('--n_lon_cells', type=int, default=0,
                    help='Number of lon cells')

config = parser.parse_args()


## CONFIGS
#===============================================

## FOUR-HOT VECTOR 
config.onehot_lat_bins = math.ceil((config.lat_max-config.lat_min)/config.onehot_lat_reso)
config.onehot_lon_bins = math.ceil((config.lon_max-config.lon_min)/config.onehot_lon_reso)
config.onehot_sog_bins = math.ceil(SPEED_MAX/config.onehot_sog_reso)
config.onehot_cog_bins = math.ceil(360/config.onehot_cog_reso)

config.data_dim  = config.onehot_lat_bins + config.onehot_lon_bins\
                 + config.onehot_sog_bins + config.onehot_cog_bins # error with data_dimension

## LOCAL THRESHOLDING
config.n_lat_cells = math.ceil((config.lat_max-config.lat_min)/config.cell_lat_reso)
config.n_lon_cells = math.ceil((config.lon_max-config.lon_min)/config.cell_lon_reso)


## PATH
if config.mode == "train":
    config.testset_name = config.trainingset_name
elif config.testset_name == "":
    config.testset_name = config.trainingset_name.replace("_train","_test")
config.trainingset_path = os.path.join(config.dataset_dir,config.trainingset_name)
config.testset_path = os.path.join(config.dataset_dir,config.testset_name)

print("Training set: " + config.trainingset_path)
print("Test set: " + config.testset_path)


# log
log_dir = config.bound + "-"\
     + os.path.basename(config.trainingset_name)\
     + "-data_dim-" + str(config.data_dim)\
     + "-latent_size-" + str(config.latent_size)\
     + "-batch_size-" + str(config.batch_size)
config.logdir = os.path.join(config.log_dir,log_dir)
if not os.path.exists(config.logdir):
    if config.mode == "train":
        os.makedirs(config.logdir)
    else:
        raise ValueError(config.logdir + " doesnt exist")

if config.log_filename == "":
    config.log_filename = os.path.basename(config.logdir)
