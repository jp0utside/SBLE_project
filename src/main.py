from parse import *
from visualize import *
from matlab import *
import trip
from train import *
import random

def main():
    stop_data = get_stop_data()
    sble_data = get_SBLE_data()
    notif_data = get_notif_data()
    bad_MM = [2, 8, 21]
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

    # ll = load_struct('LL')
    # ll = unpack(ll)
    # trips = get_trips_quick()
    # data = get_tagged_dataset(trips, include_pretrip=False)
    # graph_rssi_cdf_means_no_zero(data)
    # graph_minor_rssi(data)
    # print("Python: ")
    # basic_linear(data)
    # print("Matlab: ")
    # check_acc()
    # trips = get_trips_quick()
    # data = get_tagged_dataset(trips)
    # linear_classifier_averages(data, random.randint(0,100))


if __name__ == "__main__":
    main()
