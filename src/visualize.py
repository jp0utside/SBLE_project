import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from mpl_toolkits.basemap import Basemap
from trip import trip
from parse import *
from filter import *
import cartopy.crs as ccrs
import cartopy.feature as cfeature

hex_colors = [
        "#FF5733",  # Bright Red
        "#33FF57",  # Bright Green
        "#3357FF",  # Bright Blue
        "#FF33A8",  # Bright Pink
        "#33FFF6",  # Bright Cyan
        "#FFAF33",  # Bright Orange
        "#8D33FF",  # Bright Purple
        "#A8FF33",  # Bright Lime
        "#FF6F33",  # Bright Coral
        "#33FFAF",  # Bright Mint
        "#5733FF",  # Bright Violet
        "#FFD133",  # Bright Gold
        "#33D1FF",  # Bright Sky Blue
        "#FF33D1",  # Bright Magenta
        "#33FF33",  # Bright Green
        "#D133FF",  # Bright Lavender
        "#FFA833",  # Bright Amber
        "#33FFA8",  # Bright Aqua
        "#6F33FF",  # Bright Indigo
        "#33FFD1",  # Bright Turquoise
        "#FF3383",  # Bright Salmon
        "#33FF83",  # Bright Greenish
        "#3333FF",  # Bright Cobalt
        "#FF338F",  # Bright Pinkish
        "#33FFE1",  # Bright Light Cyan
        "#FF8F33",  # Bright Tangerine
        "#9F33FF",  # Bright Violet
        "#B8FF33",  # Bright Chartreuse
        "#FF7F33",  # Bright Tangerine
        "#33FFCF",  # Bright Teal
        "#6733FF",  # Bright Indigo
        "#FFDF33",  # Bright Mustard
        "#33B1FF",  # Bright Cerulean
        "#FF33F1",  # Bright Fuchsia
        "#33FF63",  # Bright Light Green
        "#FF3357"   # Bright Crimson
    ]

def pandas_format():
    pd.options.display.float_format = '{:.0f}'.format

def graph_trip(trip):
    stop_data = get_stop_data()
    data = trip.data.copy()

    pre_trip = data.loc[data["timestamp"] < trip.on_bus]
    mid_trip = data.loc[data["timestamp"] >= trip.on_bus]

    plt.xlim(min(stop_data["stop_lon"]) - 0.005, max(stop_data["stop_lon"]) + 0.005)
    plt.ylim(min(stop_data["stop_lat"]) - 0.005, max(stop_data["stop_lat"]) + 0.005)

    plt.scatter(stop_data["stop_lon"], stop_data["stop_lat"], color = hex_colors[:stop_data.shape[0]], marker = "^", s = 50)

    plt.scatter(pre_trip["longitude"], pre_trip["latitude"], color="red", s = 2)
    plt.scatter(mid_trip["longitude"], mid_trip["latitude"], color="blue", s = 2)

    plt.show()

def graph_map(trip):
    data = trip.data.copy()
    stop_data = get_stop_data()

    pre_trip = data.loc[data["timestamp"] < trip.on_bus]
    mid_trip = data.loc[data["timestamp"] >= trip.on_bus]

    ax = plt.axes(projection=ccrs.PlateCarree())
    
    ax.add_feature(cfeature.LAND.with_scale('10m'))
    ax.add_feature(cfeature.OCEAN.with_scale('10m'))
    ax.add_feature(cfeature.COASTLINE.with_scale('10m'))
    ax.add_feature(cfeature.BORDERS.with_scale('10m'), linestyle=":")
    ax.add_feature(cfeature.LAKES.with_scale('10m'))
    ax.add_feature(cfeature.RIVERS.with_scale('10m'))

    plt.xlim(min(stop_data["stop_lon"]) - 0.005, max(stop_data["stop_lon"]) + 0.005)
    plt.ylim(min(stop_data["stop_lat"]) - 0.005, max(stop_data["stop_lat"]) + 0.005)

    plt.scatter(stop_data["stop_lon"], stop_data["stop_lat"], color = hex_colors[:stop_data.shape[0]], marker = "^", s = 50)

    plt.scatter(pre_trip["longitude"], pre_trip["latitude"], color="red", s = 2)
    plt.scatter(mid_trip["longitude"], mid_trip["latitude"], color="blue", s = 2)

    plt.show()



    



def graph_rssi(trip):
    data = trip.data.copy()

    first_time = data["timestamp"].min()
    data["timestamp"] = data["timestamp"] - first_time
    on_bus = trip.on_bus - first_time
    
    pre_one = data.loc[(data["minor"] == 1) & (data["timestamp"] < on_bus)]
    pre_two = data.loc[(data["minor"] == 2) & (data["timestamp"] < on_bus)]
    mid_one = data.loc[(data["minor"] == 1) & (data["timestamp"] >= on_bus)]
    mid_two = data.loc[(data["minor"] == 2) & (data["timestamp"] >= on_bus)]

    #Setting rssi = 0 values to -100
    pre_one.loc[pre_one["rssi"] == 0, "rssi"] = -100
    pre_two.loc[pre_two["rssi"] == 0, "rssi"] = -100
    mid_one.loc[mid_one["rssi"] == 0, "rssi"] = -100
    mid_two.loc[mid_two["rssi"] == 0, "rssi"] = -100

    plt.scatter(pre_one["timestamp"], pre_one["rssi"], color="red", marker="x", s = 75, linewidths=.5, label="Minor 1, Pre-trip")
    plt.scatter(pre_two["timestamp"], pre_two["rssi"], color="red", marker="o", facecolors = 'none', s = 75, linewidths=.5, label="Minor 2, Pre-trip")
    plt.scatter(mid_one["timestamp"], mid_one["rssi"], color="gold", marker="x", s = 75, linewidths=.5, label="Minor 1, Mid-trip")
    plt.scatter(mid_two["timestamp"], mid_two["rssi"], color="purple", marker="o", facecolors = 'none', s = 75, linewidths=.5, label="Minor 2, Mid-trip")

    plt.xlabel("Time (Unix)")
    plt.ylabel("RSSI (dBm)")

    plt.title(str(trip.user) + " - Trip #" + str(trip.trip_idx) + "\n Seat: " + str(trip.seat) + ", Major: " + str(trip.major) + ", Exit marked: " + str(not trip.didNotMarkExit))

    plt.legend(bbox_to_anchor=(.9, 1), loc="upper left")
    plt.subplots_adjust(right=0.8)

    plt.show()


"""
Function to graph RSSI of minors 1 and 2
Takes in a dataframe, which should contain a seat location and an rssi for minors one and two in each row.
"""

def graph_minor_rssi(data):
    front = data.loc[data["seat"] == "front"]
    middle = data.loc[data["seat"] == "middle"]
    back = data.loc[data["seat"] == "back"]

    axis_max = max(max(front["rssi_1"].tolist()), max(front["rssi_2"].tolist()))
    axis_max = max(axis_max, max(max(middle["rssi_1"].tolist()), max(middle["rssi_2"].tolist())))
    axis_max = max(axis_max, max(max(back["rssi_1"].tolist()), max(back["rssi_2"].tolist())))
    axis_min = min(min(front["rssi_1"].tolist()), min(front["rssi_2"].tolist()))
    axis_min = min(axis_min, min(min(middle["rssi_1"].tolist()), min(middle["rssi_2"].tolist())))
    axis_min = min(axis_min, min(min(back["rssi_1"].tolist()), min(back["rssi_2"].tolist())))

    axis_max += ((axis_max - axis_min)/10)
    axis_min -= ((axis_max - axis_min)/10)

    plt.scatter(front["rssi_1"].tolist(), front["rssi_2"].tolist(), c="blue", marker = "x", s = 75, linewidths=.5)
    plt.scatter(middle["rssi_1"].tolist(), middle["rssi_2"].tolist(), c="green", marker = "x", s = 75, linewidths=.5)
    plt.scatter(back["rssi_1"].tolist(), back["rssi_2"].tolist(), c="red", marker = "x", s = 75, linewidths=.5)
    plt.xlabel("Minor 1 RSSI (dBm)")
    plt.ylabel("Minor 2 RSSI (dBm)")
    plt.xlim(axis_min, axis_max)
    plt.ylim(axis_min, axis_max)

    plt.show()

"""
Function to graph CDF of each seat using the RSSI difference
"""

def graph_rssi_cdf(data):
    front = data.loc[data["seat"] == "front", "rssi_diff"].tolist()
    middle = data.loc[data["seat"] == "middle", "rssi_diff"].tolist()
    back = data.loc[data["seat"] == "back", "rssi_diff"].tolist()

    min_val = min([min(front), min(middle), min(back)])
    max_val = max([max(front), max(middle), max(back)])

    f_range, f_cdf = get_cdf(front)
    m_range, m_cdf = get_cdf(middle)
    b_range, b_cdf = get_cdf(back)

    plt.plot(f_range, f_cdf, c="blue")
    plt.plot(m_range, m_cdf, c="red")
    plt.plot(b_range, b_cdf, c="green")
    plt.xlim(min_val, max_val)
    plt.ylim(0, 1)

    plt.show()

"""
Function to graph CDF of each seat using the RSSI difference, while removing outliers in the dataset.
"""

def graph_rssi_cdf_no_outliers(data):
    front = data.loc[data["seat"] == "front", "rssi_diff"].tolist()
    middle = data.loc[data["seat"] == "middle", "rssi_diff"].tolist()
    back = data.loc[data["seat"] == "back", "rssi_diff"].tolist()

    front = remove_outliers(front)
    middle = remove_outliers(middle)
    back = remove_outliers(back)

    min_val = min([min(front), min(middle), min(back)])
    max_val = max([max(front), max(middle), max(back)])

    f_range, f_cdf = get_cdf(front)
    m_range, m_cdf = get_cdf(middle)
    b_range, b_cdf = get_cdf(back)

    plt.plot(f_range, f_cdf, c="blue")
    plt.plot(m_range, m_cdf, c="red")
    plt.plot(b_range, b_cdf, c="green")
    plt.xlim(min_val, max_val)
    plt.ylim(0, 1)

    plt.show()

"""
Function to graph CDF of each seat using the RSSI difference;
Excluding zero values and taking the mean of each trip.
"""

def graph_rssi_cdf_means_no_zero(data):
    data = get_avg_rssi_diff(data)

    front = data.loc[data["seat"] == "front", "rssi"].tolist()
    middle = data.loc[data["seat"] == "middle", "rssi"].tolist()
    back = data.loc[data["seat"] == "back", "rssi"].tolist()
    

    min_val = min([min(front), min(middle), min(back)]) - 3
    max_val = max([max(front), max(middle), max(back)]) + 3

    front = sort(front)
    middle = sort(middle)
    back = sort(back)

    f_cdf = [x*(1/len(front)) for x in range(1, len(front)+1)]
    m_cdf = [x*(1/len(middle)) for x in range(1, len(middle)+1)]
    b_cdf = [x*(1/len(back)) for x in range(1, len(back)+1)]

    plt.plot(front, f_cdf, c="blue", marker='o', fillstyle='none', linestyle='-')
    plt.plot(middle, m_cdf, c="green", marker='o', fillstyle='none', linestyle='-')
    plt.plot(back, b_cdf, c="red", marker='o', fillstyle='none', linestyle='-')
    plt.xlim(min_val, max_val)
    plt.ylim(0, 1)

    plt.show()



def plot_data_with_clf(X_train, y_train, clf):
    c = {"front": "blue", "middle":"green", "back":"red"}

    if len(X_train.shape) > 1:
        x_data = X_train[:, 0].tolist()
    else:
        x_data = X_train.tolist()

    x_range = [min(x_data), max(x_data)]
    x_span = np.linspace(x_range[0], x_range[1], len(x_data))

    lines = []
    for i in c.keys():
        coef = clf[i].iloc[0]
        inter = clf[i].iloc[-1]
        y_data = [(x*coef) - inter for x in x_span]
        lines.append(y_data)
    
    y_data = list(y_train)
    
    for i in range(len(lines)):
        plt.plot(x_data, lines[i], c=list(c.values())[i])
    
    for i in range(len(x_data)):
        plt.scatter(x_data[i], y_data[i], c=c[y_data[i]])
    plt.show()





"""
Function to extract cdf values at each integer difference value in the dataset.
"""
def get_cdf(arr):
    min_val = int(min(arr))
    max_val = int(max(arr))

    arr = sort(arr)

    cdf = [0]*(int(max_val - min_val) + 1)
    rng = list(range(min_val, max_val + 1))

    count = 0
    
    for i in range(int(max_val - min_val)):
        count += arr.count(rng[i])
        cdf[i] = count / len(arr)

    
    cdf[-1] = 1

    return rng, cdf

"""
Function to remove outliers from dataset.
"""

def remove_outliers(arr):
    avg = sum(arr)/len(arr)
    std = 0

    for i in arr:
        std += abs(avg - i)
    
    std = std / len(arr)

    lower = avg - (3*std)
    upper = avg + (3*std)

    trimmed = list(filter(lambda x: x > lower and x < upper, arr))

    return trimmed





def sort(list):
    if len(list) < 2:
        return list
    else:
        mid = len(list) // 2
        left = list[:mid]
        right = list[mid:]

        left = sort(left)
        right = sort(right)

        sorted = []
        i = j = 0

        while (i < len(left)) & (j < len(right)):
            if (left[i] < right[j]):
                sorted.append(left[i])
                i += 1
            else:
                sorted.append(right[j])
                j += 1
        
        while i < len(left):
            sorted.append(left[i])
            i += 1
        
        while j < len(right):
            sorted.append(right[j])
            j += 1
        
        return sorted
            











