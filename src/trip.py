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
import numpy as np

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
        self.notifs = pd.DataFrame()
        self.end = end
        self.didNotMarkExit = False
        self.major = -1
        self.noFinalCollectingData = False
        self.postSeatChange = False
        self.preSeatChange = False
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
        if(self.data.shape[0] > 0):
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
    
    """
    Function to update object attributes using populated sble and notification tables.
    Meant to be used on trips generated from saved files.
    """
    def update_vars(self):
        if self.data.shape[0] > 0 and self.notifs.shape[0] > 0:
            pass


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
    
    def copy(self):
        # Create new trip with basic attributes
        new_trip = trip(
            user=self.user,
            start=self.start,
            on_bus=self.on_bus,
            seat=self.seat,
            seat_time=self.seat_time,
            end=self.end
        )
        
        # Copy the additional attributes
        new_trip.didNotMarkExit = self.didNotMarkExit
        new_trip.major = self.major
        new_trip.noFinalCollectingData = self.noFinalCollectingData
        new_trip.postSeatChange = self.postSeatChange
        new_trip.preSeatChange = self.preSeatChange
        new_trip.trip_idx = self.trip_idx
        
        # Copy the pandas DataFrame
        new_trip.data = self.data.copy()
        
        return new_trip
    
    def __eq__(self, other):
        if not isinstance(other, trip):
            return False

        sble_eq = True
        notif_eq = True

        if (self.data.shape[0] == other.data.shape[0]) and (self.notifs.shape[0] == other.notifs.shape[0]):
            if (self.data.shape[0] > 0) and (other.data.shape[0] > 0) and (self.notifs.shape[0] > 0) and (other.notifs.shape[0] > 0):
                sble_cols = self.data.columns.intersection(other.data.columns)
                notif_cols = self.notifs.columns.intersection(other.notifs.columns)

                if len(sble_cols) == 0 or len(notif_cols) == 0:
                    return False
                else:
                    for col in sble_cols:
                        dtype1 = self.data[col].dtype
                        dtype2 = other.data[col].dtype

                        if np.issubdtype(dtype1, np.number) and np.issubdtype(dtype2, np.number):
                            if not np.allclose(self.data[col].fillna(0), other.data[col].fillna(0),
                                            rtol = 1e-8, atol = 1e-8, equal_nan=True):
                                return False
                        else:
                            if not (self.data[col].reset_index(drop = True) == other.data[col].reset_index(drop = True)).all():
                                return False
                    
                    for col in notif_cols:
                        dtype1 = self.notifs[col].dtype
                        dtype2 = other.notifs[col].dtype

                        if np.issubdtype(dtype1, np.number) and np.issubdtype(dtype2, np.number):
                            if not np.allclose(self.notifs[col].fillna(0), other.notifs[col].fillna(0),
                                            rtol = 1e-8, atol = 1e-8, equal_nan=True):
                                return False
                        else:
                            if not (self.notifs[col].reset_index(drop = True) == other.notifs[col].reset_index(drop = True)).all():
                                return False

            elif (self.data.shape[0] > 0) and (other.data.shape[0] > 0):
                sble_cols = self.data.columns.intersection(other.data.columns)
                if len(sble_cols) == 0:
                    return False
                else:
                    for col in sble_cols:
                        dtype1 = self.data[col].dtype
                        dtype2 = other.data[col].dtype

                        if np.issubdtype(dtype1, np.number) and np.issubdtype(dtype2, np.number):
                            if not np.allclose(self.data[col].fillna(0), other.data[col].fillna(0),
                                            rtol = 1e-8, atol = 1e-8, equal_nan=True):
                                return False
                        else:
                            if not (self.data[col].reset_index(drop = True) == other.data[col].reset_index(drop = True)).all():
                                return False

            elif (self.notifs.shape[0] > 0) and (other.notifs.shape[0] > 0):
                notif_cols = self.notifs.columns.intersection(other.notifs.columns)
                if len(notif_cols) == 0:
                    return False
                else:
                    for col in notif_cols:
                        dtype1 = self.notifs[col].dtype
                        dtype2 = other.notifs[col].dtype

                        if np.issubdtype(dtype1, np.number) and np.issubdtype(dtype2, np.number):
                            if not np.allclose(self.notifs[col].fillna(0), other.notifs[col].fillna(0),
                                            rtol = 1e-8, atol = 1e-8, equal_nan=True):
                                return False
                        else:
                            if not (self.notifs[col].reset_index(drop = True) == other.notifs[col].reset_index(drop = True)).all():
                                return False

            else:
                return False
        else:
            return False
        return True
    