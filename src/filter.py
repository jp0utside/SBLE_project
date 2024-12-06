import pandas as pd
import numpy as np
import math
from trip import trip
from parse import *
from datetime import datetime

"""
Function to handle initial filtering of data to be later passed into major and minor processing functions

Args:
    trip (trip object): Trip to process data from.
    include_pretrip (bool): Include pre-trip data in frames.
    only_dominant_major (bool): Only include data which comes from the major most frequently seen during the trip.
    normalize_zero (bool): Adjust rssi readings of 0 because non-zero readings are measured negatively.
    zero_val (int): Value to normalize rssi readings of zero to.

Returns:
    data (DataFrame): Pandas data frame containing trip readings, adjusted based on inputted filter options.
"""
def prelim_filter(trip, include_pretrip, only_dominant_major, normalize_zero, zero_val):
    data = trip.data.copy()

    if not include_pretrip:
        if trip.on_bus != -1:
            data = data.loc[data["timestamp"] >= trip.on_bus]
    
    if only_dominant_major:
        if trip.major != -1:
            data = data.loc[data["major"] == trip.major]

    if normalize_zero:
        data.loc[data["rssi"] == 0, "rssi"] = zero_val
    
    data = data.reset_index()
    
    return data


"""
Function to get rssi associated with each minor, using only data collected at the same time.

Args:
    trip (trip object): Trip to process data from.
    exclude_unmatched (bool): Exclude data points collected from one beacon that do not have a corresponding point from the other.
    include_pretrip (bool): Include pre-trip data in frames.
    only_dominant_major (bool): Only include data which comes from the major most frequently seen during the trip.
    normalize_zero (bool): Adjust rssi readings of 0 because non-zero readings are measured negatively.
    zero_val (int): Value to normalize rssi readings of zero to.
    exclude_zeros (bool): Exclude all rows with a 0 value for either rssi reading.

Returns:
    join (DataFrame): Pandas data frame containing trip data, joined based on timestamp and opposing minors.
"""
def get_joint_rssi(trip, exclude_unmatched = True, include_pretrip = True, only_dominant_major = True, 
                   normalize_zero = True, zero_val = -100, exclude_zeros = False):
    data = prelim_filter(trip, include_pretrip, only_dominant_major, normalize_zero, zero_val)


    m1 = data.loc[data["minor"] == 1].copy()
    m2 = data.loc[data["minor"] == 2].copy()

    m1.drop_duplicates(subset = ['timestamp', 'minor'], keep = 'first', inplace = True)
    m2.drop_duplicates(subset = ['timestamp', 'minor'], keep = 'first', inplace = True)

    if m2.shape[0] > m1.shape[0]:
        join = m2.set_index("timestamp", drop = False).join(m1.set_index("timestamp"), lsuffix="_2", rsuffix="_1")
    else:
        join = m1.set_index("timestamp", drop = False).join(m2.set_index("timestamp"), lsuffix="_1", rsuffix="_2")
    
    if exclude_unmatched:
        join = join.dropna(subset=["rssi_1", "rssi_2"])
    
    if exclude_zeros:
        join = join.loc[join["rssi_1"] != -100]
        join = join.loc[join["rssi_2"] != -100]

    return join

"""
Function to get master dataset for use.
Compiles the relevant SBLE data points for each trip and adds additional columns to store relevant trip information.
Matches data points by timestamp to merge readings from minors 1 and 2.
Returns array of pandas DataFrames, each representing a trip, to be used for model training.

Args:
    trips (list): List of trip objects to extract data from
    n (int): Optional parameter to return data frames using average rssi values across n samples.
    exclude_unmatched (bool): Passed to preliminary filter, option to exclude data points collected from one beacon that do not have a corresponding point from the other.
    include_pretrip (bool): Passed to preliminary filter, option to include pre-trip data in frames
    only_dominant_major (bool): Passed to preliminary filter, option to only include data which comes from the major most frequently seen during the trip
    normalize_zero (bool): Passed to preliminary filter, option to adjust rssi readings of 0 because non-zero readings are measured negatively
    zero_val (int): Passed to preliminary filter, value to normalize rssi readings of zero to
    aggregate_feats (bool): Option to average position data between matched rssi readings than having both values in the dataset
    exclude_null_trips (bool): Remove empty trips from the dataset, as other filters may cause datasets to be emptied
    exclude_zeros (bool): Passed to preliminary filter, exclude all rows with a 0 value for either rssi reading 
    trim_end_zeros (bool): Remove all rows with two zero value rssi readings which fall at the end of data frames
    trim_double_zeros (bool): Remove all rosw with two zero value rssi readings
    normalize_acc (bool): Divides all rssi accuracy values by acc_val. If acc_val is 1 it normalizes all accuracy values to the largest accuracy value.
    acc_val (bool): Value used to normalize rssi readings.

Returns:
    df (array): Array holding pandas DataFrames storing filterd and tagged datasets from each trip.
"""
def get_tagged_dataset(trips, n = 1, exclude_unmatched = True, include_pretrip = False, only_dominant_major = True, 
                   normalize_zero = True, zero_val = -100, aggregate_feats = True, exclude_null_trips = True, exclude_zeros = False, trim_end_zeros = False, trim_all_zeros = False, trim_double_zeros = False,
                   normalize_acc = False, acc_val = 1):
    df = []
    seat_to_num = {"front" : 0, "middle" : 1, "back" : 2}

    for i, trip in enumerate(trips):
        if trip.seat != "none" and trip.data.shape[0] > 0:
            join = get_joint_rssi(trip, exclude_unmatched, include_pretrip, only_dominant_major, normalize_zero, zero_val, exclude_zeros)
            join["seat"] = seat_to_num[trip.seat]
            join["rssi_diff"] = join["rssi_2"] - join["rssi_1"] #Trying rssi_2 - rssi_1
            if normalize_acc and join.shape[0] > 0:
                if acc_val == 1:
                    max_acc = max([max(join["rssi_accuracy_1"]), max(join["rssi_accuracy_2"])])
                else:
                    max_acc = acc_val
                join["rssi_accuracy_1"] = join["rssi_accuracy_1"]/max_acc
                join["rssi_accuracy_2"] = join["rssi_accuracy_2"]/max_acc
                join["rssi_1_adj"] = join["rssi_1"]/join["rssi_accuracy_1"]
                join["rssi_2_adj"] = join["rssi_2"]/join["rssi_accuracy_2"]
            else:
                join["rssi_1_adj"] = join["rssi_1"]*join["rssi_accuracy_1"]
                join["rssi_2_adj"] = join["rssi_2"]*join["rssi_accuracy_2"]
            join.loc[join["rssi_accuracy_1"] < 0, "rssi_1_adj"] = zero_val
            join.loc[join["rssi_accuracy_2"] < 0, "rssi_2_adj"] = zero_val
            join["rssi_diff_adj"] = join["rssi_2_adj"] - join["rssi_1_adj"]
            join["trip_idx"] = trip.trip_idx
            join["group"] = i
            join["on_bus"] = join["timestamp"] >= trip.on_bus
            df.append(join)

            # print("i: {}, trip.data.shape: {}, join.shape: {}".format(i, trip.data.shape, join.shape))
    
    for i in range(len(df)):
        frame = df.pop(0)
        dts = [datetime.fromtimestamp(x) for x in frame["timestamp"]]
        weekdays = [dt.weekday() for dt in dts]
        hours = [dt.hour for dt in dts]
        minutes = [dt.minute for dt in dts]
        seconds = [dt.second for dt in dts]
        frame["weekday"] = weekdays
        frame["hour"] = hours
        frame["minute"] = minutes
        frame["second"] = seconds
        frame["weektime"] = frame["weekday"]*10000 + frame["hour"]*100 + frame["minute"]
        frame["weekminute"] = frame["weekday"]*1440 + frame["hour"]*60 + frame["minute"]
        df.append(frame)

    if trim_all_zeros:
        for i in range(len(df)):
            frame = df.pop(0)
            frame = frame.loc[(frame["rssi_accuracy_1"] > 0) & (frame["rssi_accuracy_2"] > 0)]
            df.append(frame)
    elif trim_double_zeros:
        for i in range(len(df)):
            frame = df.pop(0)
            frame = frame.loc[(frame["rssi_accuracy_1"] > 0) | (frame["rssi_accuracy_2"] > 0)]
            df.append(frame)
    elif trim_end_zeros:
        for i in range(len(df)):
            frame = df.pop(0)
            while frame.shape[0] > 0:
                if frame.iloc[-1]["rssi_1"] == -100 and frame.iloc[-1]["rssi_2"] == [-100]:
                    frame = frame.iloc[:-1]
                else:
                    break
            df.append(frame)

    if aggregate_feats:
        for i in range(len(df)):
            frame = df.pop(0)
            frame = aggregate_columns(frame)
            df.append(frame)
    
    if exclude_null_trips:
        filtered_dfs = list(filter(lambda x: x.shape[0] > 0, df))
        df = filtered_dfs

    if n != 1:
        for i in range(len(df)):
            frame = df.pop(0)
            frame["rssi_1"] = frame["rssi_1"].rolling(window=n).mean()
            frame["rssi_2"] = frame["rssi_2"].rolling(window=n).mean()
            frame["rssi_diff"] = frame["rssi_diff"].rolling(window=n).mean()
            df.append(frame.iloc[(n-1):])

    return df

"""
Helper function to combine position data between each minor reading by averaging.

Args:
    data (DataFrame): DataFrame with already combined columns for each corresponding minor reading, but storing each
        position feature from each reading with column titles having extra _1 or _2 indicating which minor it comes from.

Returns:
    new_data (DataFrame): DataFrame with only one column per position features, averaged between readings
"""
def aggregate_columns(data):
    unique = ["level_0", "index", "username", "major", "minor", "rssi", "rssi_accuracy"]
    cols = [x[:-2] for x in list(data.columns)]
    new_data = data.copy()
    while cols:
        col = cols.pop(0)
        if col not in unique and col in cols:
            new_data[col] = (new_data[col+"_1"] + new_data[col+"_2"])/2
            new_data = new_data.drop(columns = [col+"_1", col+"_2"])
            cols.remove(col)
    return new_data


"""
------------------------------------------------------------
ARCHIVE
------------------------------------------------------------
"""
def get_multilevel_frame(data):
    new_data = pad_data(data)
    return np.stack(new_data)

def pad_data(data):
    max_len = max([frame.shape[0] for frame in data])
    new_data = []
    for frame in data:
        if frame.shape[0] < max_len and frame.shape[0] > 0:
            padding = pd.DataFrame(0, index = range(max_len - frame.shape[0]), columns=frame.columns)
            padding.loc[:, "rssi_1"] = -100
            padding.loc[:, "rssi_accuracy_1"] = -1
            padding.loc[:, "rssi_2"] = -100
            padding.loc[:, "rssi_accuracy_2"] = -1
            padding.loc[:, "seat"] = -1
            new_frame = pd.concat([frame, padding], ignore_index=True)
            new_data.append(new_frame)
    return new_data

def get_seat_loc_data(trips):
    data = get_tagged_dataset(trips, trim_end_zeros=True)
    return data

def get_seat_loc_data_padded(trips, features):
    data = get_tagged_dataset(trips, trim_end_zeros=True)
    data = pad_data(data)
    X_data = [frame[features] for frame in data]
    y_data = [frame["seat"] for frame in data]
    X_data = np.stack(X_data)
    y_data = np.stack(y_data)
    y_data = y_data.reshape(y_data.shape[0], y_data.shape[1], 1)
    return X_data, y_data

def get_on_bus_data(trips, features):
    data = get_tagged_dataset(trips, exclude_unmatched = True, include_pretrip = True)
    data = pad_data(data)
    X_data = [frame[features] for frame in data]
    y_data = [frame["on_bus"] for frame in data]
    X_data = np.stack(X_data)
    y_data = np.stack(y_data)
    y_data = y_data.reshape(y_data.shape[0], y_data.shape[1], 1)
    return X_data, y_data


"""
Function to calculate average RSSI difference for each trip.
Each average RSSI difference is calculated and put into an array corresponding to the trip's seat location.
"""
def get_avg_rssi_diff(data):
    new_data = []

    for user in data.username_1.unique():
        for trip_idx in data.loc[data["username_1"] == user, "trip_idx"].unique():
            vals = data.loc[(data["username_1"] == user) & (data["trip_idx"] == trip_idx), "rssi_diff"].to_list()
            vals = list(filter(lambda x: x != 0, vals))

            if len(vals) != 0:
                avg = sum(vals)/len(vals)

                seat = data.loc[(data["username_1"] == user) & (data["trip_idx"] == trip_idx), "seat"].unique()[0]
                if seat == 0:
                    new_data.append([avg, 'front'])
                elif seat == 1:
                    new_data.append([avg, 'middle'])
                elif seat == 2:
                    new_data.append([avg, 'back'])
    return pd.DataFrame(new_data, columns=["rssi", "seat"])

"""
Function to get RSSI difference metrics in a 2d array format rather than through a dataframe
Primarily used to compare to matlab data
"""
def get_rssi_diff(trips):
    final = []

    new = deepen_trips(trips)

    for user in new:
        temp = []
        if len(user) == 0:
            final.append([])
        else:
            for trip in user:
                data = get_joint_rssi(trip)
                data.loc[:, "rssi_diff"] = data.loc[:, "rssi_2"] - data.loc[:, "rssi_1"] #Trying rssi_2 - rssi_1
                temp.append(data["rssi_diff"].tolist())
        final.append(temp)
    return final

"""
Function to get all data generated for a given user over the given trip, regardless of major or minor.
"""
def get_raw_data(trips, sble = []):
    if not sble:
        sble = get_sble_data()
    
    if isinstance(trip, list):
        frames = []
        for trip in trips:
            user_data = sble.loc[sble["username"] == trip.user]
            temp = user_data.loc[user_data["timestamp"] >= trip.start]
            temp = temp.loc[temp["timestamp"] <= trip.end]
            frames.append(temp)
        return frames
    else:
        user_data = sble.loc[sble["username"] == trip.user]
        temp = user_data.loc[user_data["timestamp"] >= trip.start]
        temp = temp.loc[temp["timestamp"] <= trip.end]
        return(temp)

"""
Function to get rssi data as an average of values across some number of readings.
"""
def get_average_dataset(trips, n, exclude_unmatched = True, include_pretrip = True, only_dominant_major = True, 
                   normalize_zero = True, zero_val = -100, exclude_zeros = False):
    df = []

    for trip in trips:
        if trip.seat != "none" and trip.data.shape[0] > 0:
            join = get_joint_rssi(trip, exclude_unmatched, include_pretrip, only_dominant_major, normalize_zero, zero_val, exclude_zeros)
            join["seat"] = trip.seat
            join.loc[:, "rssi_diff"] = join.loc[:, "rssi_2"] - join.loc[:, "rssi_1"] #Trying rssi_2 - rssi_1
            join["trip_idx"] = trip.trip_idx 
            df.append(join)
    
    for frame in df:
        frame["rssi_1"] = frame["rssi_1"].rolling(window=n).mean()
        frame["rssi_2"] = frame["rssi_2"].rolling(window=n).mean()
        frame["rssi_diff"] = frame["rssi_diff"].rolling(window=n).mean()
        frame = frame.iloc[(n-1):]

    data = pd.concat(df)
    return data