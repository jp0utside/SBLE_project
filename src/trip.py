"""
Class to store a recorded trip for a given user.
Params:
    user - the username of the person recording the trip.
    first - time of first recorded sble data point for given trip.
    start - time of collecting_data = true notification.
    on_bus - the time the user reported they were on the bus.
    seat - the seat position reported by the user for the given trip.
    seat_time - the time the user reported that seat.
    end - time of collecting_data = false notification.
    last - the time of last recorded instance of the user on the given trip.
    data - the pandas dataframe of all sble data related to the trip
"""
import pandas as pd

class trip:
    """
    Params:
        user: username of user
        start: first time recorded on a given trip. At initialization it may be incorrect due to including non-majority majors.
        on_bus: time user reported being on the bus
        seat: seat reported by user for the given trip
        seat_time: time that seat was reported
        end: time last recorded on a given trip. At initialization it may be incorrect due to including non-majority majors.
    """
    def __init__(self, user, start, on_bus = -1, seat = "none", seat_time = 0, end = 0):
        self.user = user
        self.start = start
        self.on_bus = on_bus
        self.seat = seat
        self.seat_time = seat_time
        self.data = pd.DataFrame()
        self.end = end
        self.didNotMarkExit = False
        self.major = -1
        self.noFinalCollectingData = False
        self.trip_idx = 0

    def get_duration(self):
        return (self.end - self.start)
    
    def print(self):
        print("User: " + str(self.user))
        print("     start: " + str(self.start))
        print("     on_bus: " + str(self.on_bus))
        print("     seat: " + str(self.seat))
        print("     seat_time: " + str(self.seat_time))
        print("     end: " + str(self.end))
        print("     major: " + str(self.major))
    
    """
    Function to clean all unwanted majors from the dataset, then update the rest of the trip data accordingly
    """
    def clean_majors(self):
        new_data = self.data.loc[self.data["major"] == self.major]
        self.data = new_data
        if self.didNotMarkExit:
            self.end = new_data.iloc[-1]["timestamp"]

    """
    Function to clean all data points for which only one minor was recorded
    """
    def clean_minors(self):
        timestamps = list(self.data["timestamp"].unique())
        new_data = pd.DataFrame()
        for i in timestamps:
            cur = self.data.loc[self.data["timestamp"] == i]
            if (cur.shape[0] > 1) and (cur["minor"].unique().shape[0] > 1):
                new_data = pd.concat([new_data, cur])
        self.data = new_data

    """
    Function to remove all trip data collected during pre-trip.
    """
    def clean_pretrip(self):
        new_data = self.data
        if self.on_bus != -1:
            new_data = new_data.loc[new_data["timestamp"] > self.on_bus]
        self.data = new_data



    def report_duration(self):
        pre_dur = self.on_bus - self.start
        trip_dur = self.end - self.on_bus
        actual_dur = self.end - self.start
        if not self.data.empty:
            # sec_rec = self.data.iloc[-1]["timestamp"] - self.data.iloc[0]["timestamp"]
            sec_rec = len(self.data["timestamp"].unique())
        else:
            sec_rec = 0
        return [pre_dur,trip_dur,actual_dur,sec_rec]
    