import pandas as pd
import math
from trip import trip
from parse import *

"""
Function to handle initial filtering of data to be later passed into major and minor processing functions
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
All trips are then put into a single dataframe for use.
"""
def get_tagged_dataset(trips, n = 1, exclude_unmatched = True, include_pretrip = False, only_dominant_major = True, 
                   normalize_zero = True, zero_val = -100, exclude_zeros = False, trim_end_zeros = False, trim_all_zeros = False, trim_double_zeros = False,
                   normalize_acc = False, acc_val = 1):
    df = []

    for trip in trips:
        if trip.seat != "none" and trip.data.shape[0] > 0:
            join = get_joint_rssi(trip, exclude_unmatched, include_pretrip, only_dominant_major, normalize_zero, zero_val, exclude_zeros)
            join["seat"] = trip.seat
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
            df.append(join)
    
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

    if n != 1:
        for i in range(len(df)):
            frame = df.pop(0)
            frame["rssi_1"] = frame["rssi_1"].rolling(window=n).mean()
            frame["rssi_2"] = frame["rssi_2"].rolling(window=n).mean()
            frame["rssi_diff"] = frame["rssi_diff"].rolling(window=n).mean()
            df.append(frame.iloc[(n-1):])

    return df

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
                if seat == "front":
                    new_data.append([avg, 'front'])
                elif seat == "middle":
                    new_data.append([avg, 'middle'])
                elif seat == "back":
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

"""
This is old code which tries to generate aggregate metrics for training (i.e. weighted avg of rssi from each beacon).
"""

# """
# Function to extract RSSI of minors 1 and 2
# Data points consist of each RSSI reading of the given minor, corrolated with the next RSSI
# point for the opposite minor. 
# """

# def get_minor_rssi_next_minor(trip, include_pretrip = False, only_dominant_major = True, normalize_zero = True, zero_val = -100):
#     data = prelim_filter(trip, include_pretrip, only_dominant_major, normalize_zero, zero_val)


#     #Graphing each data point as follows:
#     #   For each minor = 1, graphing the data point with rssi of next minor = 2 data point
#     #   Same for minor = 2
#     minor_one = []
#     minor_two = []


#     for i in range(data.shape[0]-1):
#         j = i + 1
#         if data.iloc[i]["minor"] == 1:
#             br = False
#             while data.iloc[j]["minor"] == 1:
#                 j += 1
#                 if j >= data.shape[0]:
#                     br = True
#                     break
#             if not br:
#                 minor_one.append(data.iloc[i]["rssi"])
#                 minor_two.append(data.iloc[j]["rssi"])
#         else:
#             br = False
#             while data.iloc[j]["minor"] == 2:
#                 j += 1
#                 if j >= data.shape[0]:
#                     br = True
#                     break
#             if not br:
#                 minor_one.append(data.iloc[j]["rssi"])
#                 minor_two.append(data.iloc[i]["rssi"])
        
#     return minor_one, minor_two

# """
# Function to extract RSSI data of minor against average of previously seen and next seen RSSI of opposite minor
# """

# def get_minor_rssi_average_nearest_minor(trip, include_pretrip = False, only_dominant_major = True, normalize_zero = True, zero_val = -100):
#     data = prelim_filter(trip, include_pretrip, only_dominant_major, normalize_zero, zero_val)

#     #Graphing each data point as follows:
#     #   For each minor = 1, graphing the data point with rssi of next minor = 2 data point
#     #   Same for minor = 2
#     minor_one = []
#     minor_two = []

#     one_idx = 0
#     two_idx = 0
#     br = False

#     if one_idx < data.shape[0] and data.iloc[one_idx]["minor"] != 1:
#         one_idx, br = advance(one_idx, 1, data)

#     if two_idx < data.shape[0] and data.iloc[two_idx]["minor"] != 2:
#         two_idx, br = advance(two_idx, 2, data)
    
#     if not br:
#         while (one_idx < data.shape[0]) and (two_idx < data.shape[0]) and (data.iloc[one_idx]["minor"] == 1) and (data.iloc[two_idx]["minor"] == 2):
#             if one_idx < two_idx:
#                 one_sum = data.iloc[one_idx]["rssi"]
#                 next_one, br = advance(two_idx, 1, data)

#                 if br:
#                     break
#                 else:
#                     one_sum += data.iloc[next_one]["rssi"]
#                     one_idx, br = advance(one_idx, 1, data)
#                     minor_one.append(one_sum/2)
#                     minor_two.append(data.iloc[two_idx]["rssi"])
#             else:
#                 two_sum = data.iloc[two_idx]["rssi"]
#                 next_two, br = advance(one_idx, 2, data)

#                 if br:
#                     break
#                 else:
#                     two_sum += data.iloc[next_two]["rssi"]
#                     two_idx, br = advance(two_idx, 2, data)
#                     minor_two.append(two_sum/2)
#                     minor_one.append(data.iloc[one_idx]["rssi"])
#     return minor_one, minor_two

# def get_minor_rssi_unweighted_average(trip, include_pretrip = False, only_dominant_major = True, normalize_zero = True, zero_val = -100):
#     data = prelim_filter(trip, include_pretrip, only_dominant_major, normalize_zero, zero_val)

#     #Graphing each data point as follows:
#     #   For each minor = 1, graphing the data point with rssi of next minor = 2 data point
#     #   Same for minor = 2
#     minor_one = []
#     minor_two = []

#     one_data = []
#     two_data = []

#     for i in range(data.shape[0]):
#         if data.iloc[i]["minor"] == 1:
#             one_data.append(data.iloc[i]["rssi"])
#         else:
#             two_data.append(data.iloc[i]["rssi"])
#         if not((not one_data) or (not two_data)):
#             minor_one.append(sum(one_data)/len(one_data))
#             minor_two.append(sum(two_data)/len(two_data))
#     return minor_one, minor_two

# def get_minor_rssi_weighted_sum_earlier(trip, multiplier, include_pretrip = False, only_dominant_major = True, normalize_zero = True, zero_val = -100):
#     data = prelim_filter(trip, include_pretrip, only_dominant_major, normalize_zero, zero_val)

#     #Graphing each data point as follows:
#     #   For each minor = 1, graphing the data point with rssi of next minor = 2 data point
#     #   Same for minor = 2
#     minor_one = []
#     minor_two = []

#     one_data = []
#     two_data = []

#     weight = 1

#     for i in range(data.shape[0]):
#         if data.iloc[i]["minor"] == 1:
#             one_data.append(data.iloc[i]["rssi"]*weight)
#             weight = weight*multiplier
#         else:
#             two_data.append(data.iloc[i]["rssi"]*weight)
#             weight = weight*multiplier
#         if not((not one_data) or (not two_data)):
#             minor_one.append(sum(one_data))
#             minor_two.append(sum(two_data))
#     return minor_one, minor_two

# def get_minor_rssi_weighted_avg_earlier(trip, multiplier, include_pretrip = False, only_dominant_major = True, normalize_zero = True, zero_val = -100):
#     data = prelim_filter(trip, include_pretrip, only_dominant_major, normalize_zero, zero_val)

#     #Graphing each data point as follows:
#     #   For each minor = 1, graphing the data point with rssi of next minor = 2 data point
#     #   Same for minor = 2
#     minor_one = []
#     minor_two = []

#     one_data = []
#     two_data = []

#     weight = 1
#     one_weight = 0
#     two_weight = 0

#     for i in range(data.shape[0]):
#         if data.iloc[i]["minor"] == 1:
#             one_data.append(data.iloc[i]["rssi"]*weight)
#             one_weight += weight
#             weight = weight*multiplier
#         else:
#             two_data.append(data.iloc[i]["rssi"]*weight)
#             two_weight += weight
#             weight = weight*multiplier
#         if not((not one_data) or (not two_data)):
#             minor_one.append(sum(one_data)/one_weight)
#             minor_two.append(sum(two_data)/two_weight)
#     return minor_one, minor_two

# def get_minor_rssi_weighted_avg_bustime(trip, multiplier, include_pretrip = True, only_dominant_major = True, normalize_zero = True, zero_val = -100):
#     data = prelim_filter(trip, include_pretrip, only_dominant_major, normalize_zero, zero_val)

#     #Graphing each data point as follows:
#     #   For each minor = 1, graphing the data point with rssi of next minor = 2 data point
#     #   Same for minor = 2
#     minor_one = []
#     minor_two = []

#     one_data = []
#     two_data = []

#     weight = 1
#     one_weight = 0
#     two_weight = 0

#     for i in range(data.shape[0]):
#         if data.iloc[i]["timestamp"] < trip.on_bus:
#             if data.iloc[i]["minor"] == 1:
#                 one_data.append(data.iloc[i]["rssi"]*weight)
#                 one_weight += weight
#             else:
#                 two_data.append(data.iloc[i]["rssi"]*weight)
#                 two_weight += weight
#             weight = weight / multiplier
#             if not((not one_data) or (not two_data)):
#                 minor_one.append(sum(one_data)/one_weight)
#                 minor_two.append(sum(two_data)/two_weight)
#         else:
#             if data.iloc[i]["minor"] == 1:
#                 one_data.append(data.iloc[i]["rssi"]*weight)
#                 one_weight += weight
#             else:
#                 two_data.append(data.iloc[i]["rssi"]*weight)
#                 two_weight += weight
#             weight = weight * multiplier
#             if not((not one_data) or (not two_data)):
#                 minor_one.append(sum(one_data)/one_weight)
#                 minor_two.append(sum(two_data)/two_weight)
#     return minor_one, minor_two

# def advance(idx, minor, data):
#     idx += 1
#     br = True
#     while idx < data.shape[0]:
#         if data.iloc[idx]["minor"] != minor:
#             idx += 1
#         else:
#             br = False
#             break
#     return idx, br