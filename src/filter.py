import pandas as pd
import math
from trip import trip

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
                   normalize_zero = True, zero_val = -100):
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

    return join

"""
Function to get master dataset for use.
Compiles the relevant SBLE data points for each trip and adds additional columns to store relevant trip information.
All trips are then put into a single dataframe for use.
"""
def get_tagged_dataset(trips, exclude_unmatched = True, include_pretrip = True, only_dominant_major = True, 
                   normalize_zero = True, zero_val = -100):
    df = pd.DataFrame()

    for trip in trips:
        if trip.seat != "none":
            join = get_joint_rssi(trip, exclude_unmatched, include_pretrip, only_dominant_major, normalize_zero, zero_val)
            join["seat"] = trip.seat
            join.loc[:, "rssi_diff"] = join.loc[:, "rssi_2"] - join.loc[:, "rssi_1"] #Trying rssi_2 - rssi_1
            join["trip_idx"] = trip.trip_idx 
            df = pd.concat([df, join])
    
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