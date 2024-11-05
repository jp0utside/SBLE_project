import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from mpl_toolkits.basemap import Basemap
from trip import trip
from parse import *
from filter import *
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from scipy.stats import norm, expon

seat_color = {0 : "blue", 1 : "green", 2 : "red"}

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

def quick_plot(x, y, color = "blue", xlabel = "", ylabel = "", title = "", fig = None, ax = None):
    show = False
    if ax is None:
        fig, ax = plt.subplots()
        show = True

    ax.plot(x, y, color=color)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if show:
        plt.show()

def quick_scatter(x, y, color = "blue", xlabel = "", ylabel = "", title = "", fig = None, ax = None):
    show = False
    if ax is None:
        fig, ax = plt.subplots()
        show = True

    ax.scatter(x, y, color=color)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if show:
        plt.show()

def graph_pdf(data, bin_count = 10, label = ""):
    fig, ax = plt.subplots()

    sorted = data.sort_values().to_list()

    bins = [0 for x in range(bin_count)]
    bin_vals = [x for x in np.linspace(math.floor(sorted[0]), math.ceil(sorted[-1]), bin_count)]

    idx = 0
    bidx = 0
    while idx < len(sorted):
        bin = [bin_vals[bidx], bin_vals[bidx+1]]
        print("idx: {}".format(idx))
        print("bidx: {}".format(bidx))
        print("bin: {}".format(bin))
        print()
        try:
            while sorted[idx] <= bin[1]:
                bins[bidx] += 1
                idx += 1
        except:
            break
        bidx += 1

    ax.plot(bin_vals, bins, color="blue")
    plt.show()

def graph_pdf_norm(data):
    fig, ax = plt.subplots()

    sorted = data.sort_values().to_list()

    mu, std = norm.fit(sorted)

    x = np.linspace(mu - 3*std, mu + 3*std, 100)
    y = norm.pdf(x, mu, std)*len(sorted)

    ax.plot(x, y, color = "red")

    bins = [0 for x in range(100)]
    bin_vals = [x for x in np.linspace(math.floor(sorted[0]), math.ceil(sorted[-1]), 100)]

    idx = 0
    bidx = 0
    while idx < len(sorted):
        bin = [bin_vals[bidx], bin_vals[bidx+1]]
        print("idx: {}".format(idx))
        print("bidx: {}".format(bidx))
        print("bin: {}".format(bin))
        print()
        try:
            while sorted[idx] <= bin[1]:
                bins[bidx] += 1
                idx += 1
        except:
            break
        bidx += 1
    ax.plot(bin_vals, bins, color = "blue")
    plt.show()

def graph_pdf_exp(data):
    fig, ax = plt.subplots()

    sorted = data.sort_values().to_list()

    loc, scale = expon.fit(sorted)

    x = np.linspace(min(sorted), max(sorted), 100)
    y = expon.pdf(x, loc, scale)*len(sorted)

    ax.plot(x, y, color = "red")

    bins = [0 for x in range(100)]
    bin_vals = [x for x in np.linspace(math.floor(sorted[0]), math.ceil(sorted[-1]), 100)]

    idx = 0
    bidx = 0
    while idx < len(sorted):
        bin = [bin_vals[bidx], bin_vals[bidx+1]]
        print("idx: {}".format(idx))
        print("bidx: {}".format(bidx))
        print("bin: {}".format(bin))
        print()
        try:
            while sorted[idx] <= bin[1]:
                bins[bidx] += 1
                idx += 1
        except:
            break
        bidx += 1
    ax.plot(bin_vals, bins, color = "blue")
    plt.show()

def graph_correlation(x, y, corr_coef, xlabel = "", ylabel = ""):
    vars = np.polyfit(x, y, 1)
    line = np.poly1d(vars)

    plt.figure(figsize = (10, 10))

    plt.scatter(x, y, color="blue")
    plt.plot(x, line(x), color="red", alpha=0.4)
    plt.text(0.05, 0.95, "Correlation: {}".format(corr_coef), transform=plt.gca().transAxes)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title("Correlation plot for {} and {}".format(xlabel, ylabel))
    plt.show()


def graph_decision_boundaries(X, y, clf, feature_names=None):

    #Plotting data along with decision boundaries
    # plt.subplot(121)

    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    x_grid, y_grid = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))

    # print("Mesh shapes:", x_grid.shape, y_grid.shape)
    # print("Mesh types:", x_grid.dtype, y_grid.dtype)

    Z = clf.predict(np.c_[x_grid.ravel(), y_grid.ravel()])
    Z = Z.astype(np.float64)
    Z = Z.reshape(x_grid.shape)

    # print("Predictions shape:", Z.shape)
    # print("Predictions type:", Z.dtype)

    #Plotting Decision Regions
    plt.contourf(x_grid, y_grid, Z, alpha=0.5, cmap="viridis")
    scatter = plt.scatter(X[:, 0], X[:, 1], c = [seat_color[i] for i in y])
    plt.colorbar(scatter)

    if feature_names:
        plt.xlabel(feature_names[0])
        plt.ylabel(feature_names[1])
    plt.title("Decision boundaries")
    
    plt.show()


"""
Function to plot latitude and longitude data for a given trip.
Pre-trip data plotted in red, mid-trip data plotted in blue.
"""
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

"""
Function to plot latitude and longitude data for a given trip.
Pre-trip data plotted in red, mid-trip data plotted in blue.
"""
def graph_trip_rssi_diff(data):
    stop_data = get_stop_data()

    plt.xlim(min(stop_data["stop_lon"]) - 0.005, max(stop_data["stop_lon"]) + 0.005)
    plt.ylim(min(stop_data["stop_lat"]) - 0.005, max(stop_data["stop_lat"]) + 0.005)

    plt.scatter(stop_data["stop_lon"], stop_data["stop_lat"], color = hex_colors[:stop_data.shape[0]], marker = "^", s = 50)

    norm = plt.Normalize(data["rssi_diff"].min(), data["rssi_diff"].max())

    sc = plt.scatter(data["longitude_1"], data["latitude_1"], c=data["rssi_diff"], cmap="viridis", norm=norm, s=100)

    cbar = plt.colorbar(sc)
    cbar.set_label('Value')

    # plt.figure(figsize=(10,10))

    plt.show()

"""
(Non) Function to graph latitude and longitude data for a trip on top of a map.

"""
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



    


"""
Function to graph RSSI readings over time for a single trip.
Red and blue markers represent pre-trip readings for minors one and two respectively,
gold and purple markers represent the same but for mid-trip readings.
"""
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
    plt.scatter(pre_two["timestamp"], pre_two["rssi"], color="blue", marker="o", facecolors = 'none', s = 75, linewidths=.5, label="Minor 2, Pre-trip")
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
    front = data.loc[data["seat"] == 0]
    middle = data.loc[data["seat"] == 1]
    back = data.loc[data["seat"] == 2]

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
    front = data.loc[data["seat"] == 0, "rssi_diff"].tolist()
    middle = data.loc[data["seat"] == 1, "rssi_diff"].tolist()
    back = data.loc[data["seat"] == 2, "rssi_diff"].tolist()

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
    front = data.loc[data["seat"] == 0, "rssi_diff"].tolist()
    middle = data.loc[data["seat"] == 1, "rssi_diff"].tolist()
    back = data.loc[data["seat"] == 2, "rssi_diff"].tolist()

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

    front = data.loc[data["seat"] == 0, "rssi"].tolist()
    middle = data.loc[data["seat"] == 1, "rssi"].tolist()
    back = data.loc[data["seat"] == 2, "rssi"].tolist()
    

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


"""
Function to plot data used in training models alongside linear boundaries found by those models.
"""
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

def confidence_ellipse(mean, cov, ax, n_std=1.0, facecolor='none', edgecolor='none', linewidth=2, alpha=0.5, **kwargs):
    """
    Create a plot of the covariance confidence ellipse of a 2D Gaussian.
    
    Parameters
    ----------
    mean : array-like, shape (2, )
        Mean of the Gaussian
    cov : array-like, shape (2, 2)
        Covariance matrix of the Gaussian
    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into
    n_std : float
        The number of standard deviations to determine the ellipse's radius
    **kwargs : dict
        Forwarded to matplotlib.patches.Ellipse
    """
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this two-dimensional dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, edgecolor=edgecolor, linewidth=linewidth, alpha=alpha, **kwargs)
    
    # Calculating the standard deviation of x from the square root of the covariance
    scale_x = np.sqrt(cov[0, 0]) * n_std
    scale_y = np.sqrt(cov[1, 1]) * n_std
    
    transf = transforms.Affine2D() \
        .rotate_deg(45 if pearson > 0 else -45) \
        .scale(scale_x, scale_y) \
        .translate(mean[0], mean[1])
    
    ellipse.set_transform(transf + ax.transData)
    return ellipse

def graph_gaussian(X, gmm, fig, ax, color):
    # fig, ax = plt.subplots()

    mean = gmm.means_[0]
    cov = gmm.covariances_[0]

    ax.scatter(mean[0], mean[1], marker='x', color=color, s = 150, linewidths=2)

    for std in range(1,4):
        el = confidence_ellipse(mean, cov, ax, n_std=std, edgecolor=color)
        ax.add_patch(el)
    
    ax.scatter(X["rssi_1"].to_list(), X["rssi_2"].to_list(), color = color, marker = "o", s = 25)
    # plt.show()

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
            











