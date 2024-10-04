import pandas as pd
import math
import os
import glob
from trip import trip
from filter import *

def pandas_format():
    pd.options.display.float_format = '{:.10f}'.format
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    

def get_stop_data():
    stop_data = pd.read_csv("/Users/Jake/Computer Science/SBLE_project/data/stops.csv")
    return stop_data

def get_sble_data():
    dir_path = "/Users/Jake/Computer Science/SBLE_project/data"
    files = glob.glob(dir_path + "/*_data*")
    dfs = []
    for file in files:
        df = pd.read_csv(file)
        dfs.append(df)
    data = pd.concat(dfs, ignore_index=True)
    data = data.sort_values(by=['timestamp', 'rssi'], ascending=True)
    data = data.reset_index(drop=True)
    return data

def get_notif_data():
    dir_path = "/Users/Jake/Computer Science/SBLE_project/data"
    files = glob.glob(dir_path + "/*_notification*")
    dfs = []
    for file in files:
        df = pd.read_csv(file)
        dfs.append(df)
    data = pd.concat(dfs, ignore_index=True)
    data = data.sort_values(by=['timestamp'], ascending=True)
    return data

def get_users(data):
    names = data["username"].unique()
    return names

def add_datetimes(data):
    data["datetime"] = pd.to_datetime(data["timestamp"], unit='s')
    return data

"""
Function to generate array of trip objects using SBLE and notification data
Parsing process follows procedure used in matlab code
"""
def get_trips(sble_data, notif_data):
    all_trips = []
    users = get_users(sble_data)
    for user in users:
        print("Parsing for user: " + user)
        trips = []
        user_sble = sble_data.loc[sble_data["username"] == user]
        user_notif = notif_data.loc[notif_data["username"] == user]
        

        #Have to reset indices of the new lists, as they carry over from the original one
        user_sble = user_sble.reset_index()
        user_notif = user_notif.reset_index()

        nidx = 0

        #Finding trips by iterating through notifications and updating the trips
        #Loops until dimension - 1 because if the last notification is a collecting_data = true it should be thrown out
        #If it is anything else it should be covered inside the loop already
        while nidx < user_notif.shape[0] - 1:

            eof = False
            #Iterating through user notifications until I find one that isn't collecting_data
            #Covering for case where a "collecting_data" == true notification is fired erroneously
            #(Because of something like student walking past bus but not getting on)
            while (user_notif.iloc[nidx + 1]["message_type"] == "collecting_data") and (user_notif.iloc[nidx + 1]["message"] == "true"):
                nidx += 1
                if nidx == user_notif.shape[0] - 1:
                    eof = True
                    break
            
            if eof:
                break
            
            #Saving start time of collecting_data notification in new trip
            new_trip = trip(user, user_notif.iloc[nidx]["timestamp"])

            #Incrementing notification index to next notification
            nidx += 1

            #Variable to track whether or not a trip actually happened
            no_trip = False

            #Iterating through notifications to find and handle the next event
            while user_notif.iloc[nidx]["message_type"] != "collecting_data":
                match user_notif.iloc[nidx]["message_type"]:
                    case "sitting_on_bus":
                        if user_notif.iloc[nidx]["message"] == "true":
                            new_trip.on_bus = user_notif.iloc[nidx]["timestamp"]
                        else:
                            #If user responded no to "sitting_on_bus" prompt, no trip happened, and this instance should be discarded
                            no_trip = True
                            break
                    case "seat_location":
                        new_trip.seat = user_notif.iloc[nidx]["message"]
                        new_trip.seat_time = user_notif.iloc[nidx]["timestamp"]
                if nidx == user_notif.shape[0] - 1:
                    eof = True
                    break
                nidx += 1
            
            if not no_trip:
                #Covering for the case where a collecting_data = false was not received
                #In this case setting end time to timestamp of next collecting_data = true
                #Setting didNotMarkExit to true to use later
                #If last notification for user is "sitting_on_bus" or "seat_location", setting 
                #noFinalCollectingData to true
                new_trip.end = user_notif.iloc[nidx]["timestamp"]
                if (user_notif.iloc[nidx]["message"] == "true"):
                    new_trip.didNotMarkExit = True
                elif eof:
                    new_trip.didNotMarkExit = True
                    new_trip.noFinalCollectingData = True

                trips.append(new_trip)
                
            nidx += 1
        # print("Trips created For user")

        #Given empty trip objects which hold only notification data assigned to them,
        #use this information to fill the trip object with the appropriate sble data.
        #Also assigns trip major data for major which it saw for the longest duration.
        # print("Filling in data for trips")
        for t in trips:
            start_idx = 0
            end_idx = 0
            sble_idx = 0
            while (sble_idx < user_sble.shape[0]):
                if (user_sble.iloc[sble_idx]["timestamp"] < t.start):
                    sble_idx += 1
                else:
                    break
            
            start_idx = sble_idx

            if t.noFinalCollectingData:
                sble_idx = user_sble.shape[0] - 1
            else:
                while (sble_idx < user_sble.shape[0] - 1):
                    if (user_sble.iloc[sble_idx + 1]["timestamp"] < t.end):
                        sble_idx += 1
                    else:
                        break
        
            end_idx = sble_idx
            t.data = user_sble.iloc[start_idx:end_idx+1, :].copy()

            # print(t.data.shape)
            # t.print()
            if t.data.shape[0] > 0:
                major_dur = [0]*(t.data["major"].max() + 1)

                # print("unique: " + str(t.data["major"].unique()))
                # print("major_dur: " + str(major_dur))

                major = 0
                freq = 0
                for i in t.data["major"]:
                    if i != major:
                        # print("major: " + str(major))
                        # print("freq: " + str(freq))
                        major_dur[major] = max(major_dur[major], freq)
                        major = i
                        freq = 1
                    else:
                        freq += 1
                major_dur[major] = max(major_dur[major], freq)
                t.major = major_dur.index(max(major_dur))

        # print("Data filled in")
        trips = group_sort(trips)

        # print("Finding trips to merge")
        #With all trips for the given user filled, merge trips which may have been separated
        #due to app malfunction or user error
        if len(trips) > 1:
            new_trips = []


            #Merging trips based on the following criteria:
            #   1. First trip did not mark exit
            #   2. The majors of both trips are the same
            #   3. Trip start times are within 20 minutes of each other
            #   4. Trips do not have conflicting seat reports

            t = 0
            while t < (len(trips)):
                if t == (len(trips) - 1):
                    new_trips.append(trips[-1])
                    break
                else:
                    if (trips[t].didNotMarkExit == True) and (trips[t].major == trips[t+1].major) and (abs(trips[t].start - trips[t+1].start) <= 1200):
                        if (trips[t].seat == trips[t+1].seat) | (not ((trips[t].seat != "none") and (trips[t+1].seat != "none"))):
                            new_trips.append(merge_trips(trips[t], trips[t+1]))
                            t += 2
                        else:
                            new_trips.append(trips[t])
                            t += 1
                    else:
                        new_trips.append(trips[t])
                        t += 1
            
            trips = new_trips
        # print("Finished merging trips")
        # print()
        all_trips.extend(trips)

    add_trip_numbers(all_trips)
    
    return all_trips

"""
Function to get trip objects WITHOUT merging trips as outlined in the matlab code
"""

def get_trips_unmerged(sble_data, notif_data):
    all_trips = []
    users = get_users(sble_data)
    for user in users:
        trips = []
        user_sble = sble_data.loc[sble_data["username"] == user]
        user_notif = notif_data.loc[notif_data["username"] == user]
        

        #Have to reset indices of the new lists, as they carry over from the original one
        user_sble = user_sble.reset_index()
        user_notif = user_notif.reset_index()

        nidx = 0

        #Finding trips by iterating through notifications and updating the trips
        #Loops until dimension - 1 because if the last notification is a collecting_data = true it should be thrown out
        #If it is anything else it should be covered inside the loop already
        while nidx < user_notif.shape[0] - 1:

            eof = False
            #Iterating through user notifications until I find one that isn't collecting_data
            #Covering for case where a "collecting_data" == true notification is fired erroneously
            #(Because of something like student walking past bus but not getting on)
            while (user_notif.iloc[nidx + 1]["message_type"] == "collecting_data") and (user_notif.iloc[nidx + 1]["message"] == "true"):
                nidx += 1
                if nidx == user_notif.shape[0] - 1:
                    eof = True
                    break
            
            if eof:
                break
            
            #Saving start time of collecting_data notification in new trip
            new_trip = trip(user, user_notif.iloc[nidx]["timestamp"])

            #Incrementing notification index to next notification
            nidx += 1

            #Variable to track whether or not a trip actually happened
            no_trip = False

            #Iterating through notifications to find and handle the next event
            while user_notif.iloc[nidx]["message_type"] != "collecting_data":
                match user_notif.iloc[nidx]["message_type"]:
                    case "sitting_on_bus":
                        if user_notif.iloc[nidx]["message"] == "true":
                            new_trip.on_bus = user_notif.iloc[nidx]["timestamp"]
                        else:
                            #If user responded no to "sitting_on_bus" prompt, no trip happened, and this instance should be discarded
                            no_trip = True
                            break
                    case "seat_location":
                        new_trip.seat = user_notif.iloc[nidx]["message"]
                        new_trip.seat_time = user_notif.iloc[nidx]["timestamp"]
                if nidx == user_notif.shape[0] - 1:
                    eof = True
                    break
                nidx += 1
            
            if not no_trip:
                #Covering for the case where a collecting_data = false was not received
                #In this case setting end time to timestamp of next collecting_data = true
                #Setting didNotMarkExit to true to use later
                #If last notification for user is "sitting_on_bus" or "seat_location", setting 
                #noFinalCollectingData to true
                new_trip.end = user_notif.iloc[nidx]["timestamp"]
                if (user_notif.iloc[nidx]["message"] == "true"):
                    new_trip.didNotMarkExit = True
                elif eof:
                    new_trip.didNotMarkExit = True
                    new_trip.noFinalCollectingData = True

                trips.append(new_trip)
                
            nidx += 1

        #Given empty trip objects which hold only notification data assigned to them,
        #use this information to fill the trip object with the appropriate sble data.
        #Also assigns trip major data for major which it saw for the longest duration.
        for t in trips:
            start_idx = 0
            end_idx = 0
            sble_idx = 0
            while (sble_idx < user_sble.shape[0]):
                if (user_sble.iloc[sble_idx]["timestamp"] < t.start):
                    sble_idx += 1
                else:
                    break
            
            start_idx = sble_idx

            if t.noFinalCollectingData:
                sble_idx = user_sble.shape[0] - 1
            else:
                while (sble_idx < user_sble.shape[0] - 1):
                    if (user_sble.iloc[sble_idx + 1]["timestamp"] < t.end):
                        sble_idx += 1
                    else:
                        break
        
            end_idx = sble_idx
            t.data = user_sble.iloc[start_idx:end_idx+1, :]

            major_dur = [0]*(t.data["major"].max() + 1)

            # print("unique: " + str(t.data["major"].unique()))
            # print("major_dur: " + str(major_dur))

            major = 0
            freq = 0
            for i in t.data["major"]:
                if i != major:
                    # print("major: " + str(major))
                    # print("freq: " + str(freq))
                    major_dur[major] = max(major_dur[major], freq)
                    major = i
                    freq = 1
                else:
                    freq += 1
            major_dur[major] = max(major_dur[major], freq)
            t.major = major_dur.index(max(major_dur))

        
        trips = group_sort(trips)

        all_trips.extend(trips)

    add_trip_numbers(all_trips)

    return all_trips

"""
Single function to generate trip objects, including other parsing options
"""
def get_trips_quick(clean_majors = True, clean_minors = False, merge = True, include_pretrips = True):
    if merge:
        trips = group_sort(get_trips(get_sble_data(), get_notif_data()))
    else:
        trips = group_sort(get_trips_unmerged(get_sble_data(), get_notif_data()))
    for i in trips:
        if clean_majors:
            i.clean_majors()
        if clean_minors:
            i.clean_minors()
    return trips

"""
Helper function to merge two trip objects that fit the critera
"""
def merge_trips(t1, t2):
    new_trip = trip(t1.user, t1.start)
    
    if (t1.seat == "none") and (t2.seat == "none"):
        new_trip.seat == "none"
    elif (t1.seat == "none"):
        new_trip.seat = t2.seat
        new_trip.seat_time = t2.seat_time
    else:
        new_trip.seat = t1.seat
        new_trip.seat_time = t1.seat_time

    new_trip.on_bus = t1.on_bus
    new_trip.end = t2.end
    new_trip.didNotMarkExit = False
    new_trip.major = t1.major
    new_trip.data = pd.concat([t1.data, t2.data])

    return new_trip

"""
Sorting function to sort array of trip objects
First by user alphabetically, then by chronological order
"""
def group_sort(trips):
    if len(trips) <= 1:
        return trips
    else:
        mid = len(trips)//2
        left = trips[:mid]
        right = trips[mid:]

        left = group_sort(left)
        right = group_sort(right)

        return merge(left, right)

def merge(left, right):
    # print("left: " + str(left))
    # print("right: " + str(right))
    sorted_trips = []

    i = j = 0
    while (i < len(left)) and (j < len(right)):
        if left[i].user < right[j].user:
            sorted_trips.append(left[i])
            i += 1
        elif right[j].user < left[i].user:
            sorted_trips.append(right[j])
            j += 1
        else:
            if left[i].start < right[j].start:
                sorted_trips.append(left[i])
                i += 1
            else:
                sorted_trips.append(right[j])
                j += 1
    
    while i < len(left):
        sorted_trips.append(left[i])
        i += 1

    while j < len(right):
        sorted_trips.append(right[j])
        j += 1
    return sorted_trips

"""
Function to add trip indices to each trip object, based on the total number of trips for each user in the array
"""
def add_trip_numbers(trips):
    trips = group_sort(trips)
    trip_idx = 1
    current_user = ""
    for i in trips:
        if i.user != current_user:
            trip_idx = 1
            current_user = i.user
        else:
            trip_idx += 1
        i.trip_idx = trip_idx

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
Functions to take trip arrays and make them 2D based on user, and vice-versa.
"""

def flatten_trips(trips):
    if isinstance(trips[0], list):
        new_trips = []
        for i in trips:
            for j in i:
                new_trips.append(trips)
        return new_trips
    else:
        return trips

def deepen_trips(trips):
    if not isinstance(trips[0], list):
        new_trips = []
        temp = []
        user = trips[0].user
        for i in trips:
            if user != i.user:
                new_trips.append(temp)
                temp = []
                temp.append(i)
            else:
                temp.append(i)
            user = i.user
        new_trips.append(temp)
        return new_trips
    else:
        return trips

"""
Function to report trip data in a similar manner to the way matlab reports the data
"""
def report_trip_data(trips):
    add_trip_numbers(trips)
    for i in trips:
        data = i.report_duration()
        print("{0} - Trip #{1}, seat: {2}, Pretrip Duration: {3}, Trip Duration: {4}, Actual Trip Duration: {5}, Seconds Recorded: {6}".format(i.user, i.trip_idx, i.seat, round(data[0]), round(data[1]), round(data[2]), round(data[3])))

