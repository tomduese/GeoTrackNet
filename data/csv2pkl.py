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
A script to merge AIS messages into AIS tracks.
"""
import numpy as np
import pandas as pd
import os
import pickle
from datetime import datetime
import time
from tqdm import tqdm 

## PARAMS
#======================================


# ## central mad
LAT_MIN = 31.0
LAT_MAX = 39.0
LON_MIN = 9.5
LON_MAX = 21.0

dataset_path = "./data/central_med/"
l_csv_filename =["raw_data.csv"]

pkl_filename = "centralmed_track.pkl"
pkl_filename_train = "centralmed_train_track.pkl"
pkl_filename_valid = "centralmed_valid_track.pkl"
pkl_filename_test  = "centralmed_test_track.pkl"

cargo_tanker_filename = "centralmed_cargo_tanker.npy"

t_train_min = time.mktime(time.strptime("01/11/2023 00:00:00", "%d/%m/%Y %H:%M:%S"))
t_train_max = time.mktime(time.strptime("20/11/2023 23:59:59", "%d/%m/%Y %H:%M:%S"))
t_valid_min = time.mktime(time.strptime("21/11/2023 00:00:00", "%d/%m/%Y %H:%M:%S"))
t_valid_max = time.mktime(time.strptime("25/11/2023 23:59:59", "%d/%m/%Y %H:%M:%S"))
t_test_min  = time.mktime(time.strptime("26/11/2023 00:00:00", "%d/%m/%Y %H:%M:%S"))
t_test_max  = time.mktime(time.strptime("30/11/2023 23:59:59", "%d/%m/%Y %H:%M:%S"))
t_min = time.mktime(time.strptime("01/11/2023 00:00:00", "%d/%m/%Y %H:%M:%S"))
t_max = time.mktime(time.strptime("30/11/2023 23:59:59", "%d/%m/%Y %H:%M:%S"))

#========================================================================
LAT_RANGE = LAT_MAX - LAT_MIN
LON_RANGE = LON_MAX - LON_MIN
SOG_MAX = 30.0  # the SOG is truncated to 30.0 knots max.

EPOCH = datetime(1970, 1, 1)
LAT, LON, SOG, COG, HEADING, ROT, NAV_STT, TIMESTAMP, MMSI, SHIPTYPE  = list(range(10))
message_cols = [
        "lat",
        "lon",
        "speed",
        "course",
        "heading",
        "rot",
        "navstatus_int",
        "timestamp",
        "mmsi",
    ]
conversion_dict = {
    'lat': float,
    'lon': float,
    'speed': float,
    'course': float,
    'heading': int,
    'rot': float,
    'navstatus_int': int,
    'timestamp': int,
    'mmsi': int,
    'vessel_type_int': int
}



CARGO_TANKER_ONLY = True
if  CARGO_TANKER_ONLY:
    pkl_filename = "ct_"+pkl_filename
    pkl_filename_train = "ct_"+pkl_filename_train
    pkl_filename_valid = "ct_"+pkl_filename_valid
    pkl_filename_test  = "ct_"+pkl_filename_test
    
print(pkl_filename_train)


## LOADING CSV FILES
#======================================
df = pd.read_csv('data/central_med/converted_raw_data.csv', sep=';')
df = df.astype(conversion_dict)



#del l_l_msg
print("Total number of AIS messages: ",len(df))

print(f'Lat min: {df.lat.min()}, Lat max: {df.lat.max()}')
print(f'Lon min: {df.lon.min()}, Lon max: {df.lon.max()}')
print(f'Ts min: {datetime.utcfromtimestamp(df.timestamp.min())}, Ts max: {datetime.utcfromtimestamp(df.timestamp.max())}')


## Vessel Type    
#======================================
print("Selecting vessel type ...")
def sublist(lst1, lst2):
   ls1 = [element for element in lst1 if element in lst2]
   ls2 = [element for element in lst2 if element in lst1]
   return (len(ls1) != 0) and (ls1 == ls2)

VesselTypes = df.groupby('mmsi')['vessel_type_int'].first().to_dict()
l_mmsi = sorted(df.mmsi.unique())

    
l_cargo_tanker = []
l_fishing = []
for mmsi_ in list(VesselTypes.keys()):
    if sublist([VesselTypes[mmsi_]], list(range(70,80))) or sublist([VesselTypes[mmsi_]], list(range(80,90))):
        l_cargo_tanker.append(mmsi_)
    if sublist([VesselTypes[mmsi_]], [30]):
        l_fishing.append(mmsi_)

print("Total number of vessels: ",len(VesselTypes))
print("Total number of cargos/tankers: ",len(l_cargo_tanker))
print("Total number of fishing: ",len(l_fishing))

print("Saving vessels' type list to ", cargo_tanker_filename)
np.save(cargo_tanker_filename,l_cargo_tanker)
np.save(cargo_tanker_filename.replace("_cargo_tanker.npy","_fishing.npy"),l_fishing)


## FILTERING 
#======================================
# Selecting AIS messages in the ROI and in the period of interest.

## LAT LON
df = df[df['lat']>=LAT_MIN]
df = df[df['lat']<=LAT_MAX]
df = df[df['lon']>=LON_MIN]
df = df[df['lon']<=LON_MAX]
# SOG
df = df[df['speed']>=0]
df = df[df['speed']<=SOG_MAX]
# COG
df = df[df['course']>=0]
df = df[df['course']<=360]

# TIME
df = df[df['timestamp']>=0]

df = df[df['timestamp']>=t_min]
df = df[df['timestamp']<=t_max]
df_train = df[df['timestamp']>=t_train_min]
df_train = df_train[df_train['timestamp']<=t_train_max]
df_valid = df[df['timestamp']>=t_valid_min]
df_valid = df_valid[df_valid['timestamp']<=t_valid_max]
df_test  = df[df['timestamp']>=t_test_min]
df_test  = df_test[df_test['timestamp']<=t_test_max]

print("Total msgs: ",len(df))
print("Number of msgs in the training set: ",len(df_train))
print("Number of msgs in the validation set: ",len(df_valid))
print("Number of msgs in the test set: ",len(df_test))


## MERGING INTO DICT
#======================================
# Creating AIS tracks from the list of AIS messages.
# Each AIS track is formatted by a dictionary.
print("Convert to dicts of vessel's tracks...")


def create_dataset(df) -> dict:
    grouped_df = df.groupby('mmsi')
    dictionary = {
        key: np.array(group.sort_values(by="timestamp")[message_cols])
        for key, group in tqdm(grouped_df)
    }
    if CARGO_TANKER_ONLY:
        dictionary = {
            key: value for key, value in dictionary.items() if key in l_cargo_tanker
        }
    dictionary = {
        key: np.array(sorted(value, key=lambda m_entry: m_entry[TIMESTAMP]))
        for key, value in tqdm(dictionary.items())
    }
    return dictionary


# Group by 'mmsi' and iterate over the groups
Vs_train = create_dataset(df_train)

# Validation set
Vs_valid = create_dataset(df_valid)

# Test set
Vs_test = create_dataset(df_test)


## PICKLING
#======================================
for filename, filedict in zip([pkl_filename_train,pkl_filename_valid,pkl_filename_test],
                              [Vs_train,Vs_valid,Vs_test]
                             ):
    print("Writing to ", os.path.join(dataset_path,filename),"...")
    with open(os.path.join(dataset_path,filename),"wb") as f:
        pickle.dump(filedict,f)
    print("Total number of tracks: ", len(filedict))
