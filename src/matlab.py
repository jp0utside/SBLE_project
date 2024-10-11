from parse import *
from filter import *
from visualize import *
from trip import trip
import scipy.io
import numpy

"""
REMEMBER: SUBTRACT MATLAB INDICES BY 1

DATA STRUCTURE DESCRIPTIONS
- LL: indices of SBLE rows for each trip
- LLM: same thingh, but includes a value for the major and splits the trip values up by minor
- LL1: indices of SBLE rows for minor 1 values from each trip
- LL2: indices of SBLE rows for minor 2 values from each trip
- SL: seat location for each trip (1,2, or 3)
- theMM: the major seen the majority of the times recorded for a given trip
- T_LLM: seems to have structure to store LL, LL1, LL2, and theMM in the same struct.
    Has structs for each trip, and each trip has a M value (major), and D1 and D2 structs each holding an array of values.
- 

TRIP PARSING INCONSISTANCIES
- THEORY: all final trips for users draw the "data collection" line at the final notification for that user
- UPDATE:
    - Current trip parsing methods are largely consistent with Matlab parsing.
    - Complexities considered:
        - Matlab indices need to be subtracted by 1
        - Many trip and pretrip index arrays contain a single duplicate value so need to filter those by unique values
    - Outstanding unaligned trips:
        - Trips[7] (8), Liverleaf trip 4
        - Trips[15] (16), Papyrus Plants trip 4
        - Trips[18] (19), Papyrus Plants trip 7
        - Trips[21] (22), Rhododendrons trip 1
        - Trips[43] (44), Zebra trip 2
        - Trips[44] (45), Zebra trip 3
- Trips[15], Papyrus Plant trip 4
    - This one is odd
    - Python and Matlab trips contain the same number of data points
    - Python trip starts and stops recording data at a seemingly arbitrary place (relative to notification data)
    - Matlab trip starts recording data just after first collecting_data = True notification, and stops just after subsequent collecting_data = False
    - MAJORS ARE DIFFERENT
    - Ok weird, SBLE data between collecting_data = True and collecting_data = False notifications contains an EXACTLY EVEN number of major = 3 and major = 15 data points
    - Weirder, the major = 3 data points perfectly surround the major = 15 ones (i.e. it goes major = 3 until major = 15 until major = 3 again)
    - Even weirder, major = 3 data points in the front only have minor = 1, and major = 3 data points at the end only have minor = 2
- Trips[7], Liverleaf trip 4
    - Python data starts right after collecting_data = True, ends seemingly with the last data recorded for Liverleaf since there was no collecting_data = False notification after
    - Matlab data starts at the same place, but ends at the time of seat_location = front notification
    - May be that, since it is the last trip for Liverleaf and there was no concluding collecting_data = False notification, Matlab cuts it off at the last notification
- Trips[18], Papyrus Plants trip 7
    - Python data starts sometime after collecting_data = True notification and stops like 12 seconds before collecting_data = False notification (presumably due to trimmed majors)
    - Matlab data starts almost 80 seconds after Python data, but ends at the same spot
    - Matlab data starts collecting right after the sitting_on_bus = True notification
    - Did I lose the pretrip?
        No it's empty.
- Trips[21], Rhododendrons trip 1
    - Python and Matlab datasets are identical except Python trip has one extra row at the end
    - Don't know why it isn't included in Matlab, it comes before collecting_data = False notification and carries the same major
    - Could be some small detail, not super worried about it
- The last two are supposed to be clipped together, but there may have been an issue with how I did this in processing
    - For some reason, the "clipping" happens in Matlab after the trips are already sorted out
    - So when importing them in they are still separate
    - To manually clip these I tried just grouping the trip and pre-trip data from the two trips to be clipped but this may not be how to go about it

AVERAGE RSSI INCONSISTENCIES
- Most of the trips align, however there is a lot of inconsistency in how the average RSSI difference values are calculated
- And even more surprising an uneven amount of trips in each category (front, middle, back)
    - Front each dataset has 6 points, but the Python iteration has 14 middle points to Matlab's 15, and 16 back data points to Matlab's 14
- Which trips are excluded overall?
    - Should be any with bad majors, as they don't have rows recording RSSI difference (majors 2, 8, 21)
        - These are (one indexed): Cat - 1, Liverleaf - 3, Papyrus Plants - 1, Papyrus Plants - 5, Shrimp - 10, Shrimp - 11, Squid - 2
    - This should leave 39 trips
    - Trips should be left off where the seat isn't marked
        - Only one, trips[26]/Shrimp-5
    - Leaves 38 trips, Python currently has 37 and Matlab 35
- Need to figure out:
    - Which trips Python is setting the seat location different from Matlab
    - Which other trips need to be discounted
    - Why the averages for the correct ones still don't work right
- Seat Locations
    - All Python trip seat locations seem to line up with matlab ones
    - Only possible difference would be if Matlab isn't clipping the last two Zebra trips automatically
- DISCOVERY: seems like RSSIdiff data does not include pretrips
- Tried CDF of means, MUCH CLOSER
- Still some problem datapoints (indexed unordered, thus probably retain original order from trips data):
    - Front: 1st, 2nd (slightly)
    - Middle: 8th (slightly), 13th (slightly), 14th (slightly), 15th (MATLAB has extra one)
    - Back: 7th (significant), 9th (slightly), 11th, 14th, 15th (Python has extra one)
- NOTE: seems like excluding pretrips fixed the quantity inconsistency by 1 for the back datapoints
    - Must have been that one of the back datapoints was all pre-trip
- Problem Trips:
    - trips[3], Guinea Pig 1, Seat: front, Python RSSIdiff contains 303 values to Matlab's 222
        - Trip data excluding pretrip lines up for Python and Matlab trip data
        - SEVERAL data points with the same timestamp. This must be what is creating the extra difference points.
    - trips[7], Liverleaf 4, Seat: front, Python has 171 to Matlab's 6 (Known inconsistencies with this trip)
    - trips[15], Papyrus Plant 4, Seat: back, Python has 10 to Matlab's 0 (Known inconsistencies with this trip) (EXTRA BACK in PYTHON DATA)
    - trips[23], Shrimp 2, Seat: middle, Python has 155 to Matlab's 143
    - trips[25], Shrimp 4, Seat: back, Python has 241 to Matlab's 73
    - trips[30], Shrimp 9, Seat: back, Python has 122 to Matlab's 119
    - trips[35], Shrimp 14, Seat: back, Python has 55 to Matlab's 52
    - trips[39], Shrimp 18, Seat: back, Python has 867 to Matlab's 59
    - trips[43], Zebra 2, Seat: middle, Python has 477 to Matlab's 499
    - trips[44], Zebra 3, Seat: middle, Python has 818 to Matlab's 838
    - Trips[45], Zebra 4, Seat: back, Python has 173 to Matlab's 92
- ISSUE: trips contain multiple rows for the same timestamp AND minor
    - Based on what Matlab code seems to be doing, going to keep the first unique row for values timestamp, minor
- Problem Trips AFTER fix:
    - front: 2nd (minor)
    - middle: 13th (minor), 14th (minor), 15th (STILL AN EXTRA MATLAB)
    - back: 15th (STILL AN EXTRA PYTHON)
    - By trip:
        - trips[7], Liverleaf 4, Seat: front, Python has 168 to Matlab's 6 (Known inconsistencies) (Average Values: -11.304964539007091 -- -11.166666666666666)
        - trips[15], Papyrus Plant 4, Seat: back, Python has 10 to Matlab's 0 (Known inconsistencies) (Average Values: 6.308641975308642 -- NONE)
        - trips[43], Zebra 2, Seat: middle, Python has 477 to Matlab's 499 (Average Values: 10.467415730337079 -- 10.473233404710921)
        - trips[44], Zebra 3, Seat: middle, Python has 818 to Matlab's 838 (Average Values: 5.532818532818533 -- 5.763819095477387)
- EXTRA BACK VALUE IN PYTHON HAS SAME VALUE AS EXTRA MIDDLE VALUE IN MATLAB
    - Fairly certain Papyrus Plants 4 is getting counted as a middle incorrectly
    - Data is split evenly between majors 3 and 15, but 3 data does not have any minor 2 

KEY TAKEAWAYS:
- Subtract all matlab indices by 1
- Common inconsistencies can be caused by:
    - Major confusion (particularly when data range has equal points with 2 different majors)
    - Clipping inconsistencies
"""

"""
NEW DATA ANALYSIS
- Still many inconsistencies, even after proper clipping
- Many "off by 1" or 2-10 errors, now need to look for those and figure out why it is happening.
- Already sorting by 2 columns, but might need another one just to make certain both matlab and python are aligned
- QUESTION: Is LLM the only structure that only includes relevant majors? Could LL also be filtering out incorrect majors?
    - Could double minors be in play for error?
    - Furthermore, do my trips by default clean majors?
- FIXED PARSING FUNCTION
- Much closer matches the matlab trips
- Picking through specific inconsistancies:
    - Trips[30], Cat 31, issue here is a confusing notification sequence
        - collecting_data = True, sitting_on_bus = True, collecting_data = True, sitting_on_bus = True, seat_location = front, collecting_data = False
        - Matlab data starts just after first collecting_data = True, stops right before second collecting_data = True
        - Python data starts just after first collecting_data = True (same spot), stops somewhere between second collecting_data = True and sitting_on_bus = True
        - End time for trip is set to the final collecting_data = False, so must be that there are no data points recorded 
        - THIS IS NOT TRUE, there are several more data points recorded
        - Questions:
            - Why is the python trip not filling with all the data?
        - IN MATLAB
            - notification sequence breaks up into two separate trips and pretrips, one with both occuring between the first two collecting_datas, one occuring after the collecting_data = True and before collecting_data = False
            - But the data isn't kept, seems to be zeroed out for some reason
        - Could this be a majors issue?
            - YES. Most of the data from the second half of the python trip is a different major from the first half.
        - Current Guess:
            - Python processes this sequence as two trips
            - The trips become merged, but a lot of the data is trimmed because it is from a different trip
        - NEED TO ADD: If majority major is only seen in the pre-trip, then cancel the trip
        - FIXED
    - Trips[51, 52, 53], Creeping 1, 2, and 3, Matlab trips merge all of these into one, python has them
        - NEED TO FIX: parsing loop condition was incorrect for looping as long as notification isn't collecting_data AND message ISN'T "answered no"
        - This made trips[52] and trips[53] get parsed as one trip
        - trips[51] is only 2 data points, so matlab and python were only 2 off and it was moved into close
        - Call this good enough for now
    - Trips[78,79], Elk 1 and 2, Matlab has these as two trips of other sizes (the first is very small and the second is very large), python has them with almost even number of points
        - Python logs: Merging Trips 0 and 1
                Merging Trips 8 and 9
        - NEED TO FIX: my helper function to merge trips sets the new_trip.didNotMarkExit = False. Should be new_trip.didNotMarkExit = t2.didNotMarkExit
        - NEED TO FIX: merging process only allows for the merging of TWO trips, and doesn't consider any additional. Need to make it better
        - Now the trips merge for the python data, but still are unmerged for the matlab data. Not sure why
    - THESE FIXES ALMOST FIXED EVERYTHING
    - Current stats
        - len(trips) = 490
        - num matched: python - 467, matlab - 463 (have to figure out why uneven)
        - num unmatched: python - matlab - 14
        - num close: python - matlab - 9

"""

problem_trips = [7, 15, 18, 21, 43, 44]



"""
Function to import matlab data structures into Python.
Data structures need to be saved in Archive/data_structures/, takes file/structure name without extension as input
"""
def load_struct(struct_name):
    path = '../matlab_archive/data_structures/'
    path = path + struct_name + '.mat'
    obj = scipy.io.loadmat(path)
    struct = obj[struct_name]
    arr = mat_to_arr(struct)

    return arr

"""
Function to take in data structure created by importing .mat file and turn it into a multi-tiered array.
"""
def mat_to_arr(struct):
    data = []

    if isinstance(struct, np.ndarray):
        if struct.shape[0] != 0:
            if isinstance(struct[0], np.number):
                data.append(struct.tolist())
            else:
                if len(struct.shape) > 1:
                    if (struct.shape[0] == 1) and (struct.shape[1] == 0):
                        return []
                    
                for i in range(struct.shape[0]):
                    data.append(mat_to_arr(struct[i]))

    elif isinstance(struct, numpy.void):
        for i in range(len(struct)):
            data.append(mat_to_arr(struct[i]))
    
    return data

"""
Function to (attempt) to un-nest superfluous sub-arrays.
"Un-packs" the array if it holds only one item and that item is an array, or if all other arrays at that sublevel only hold one item as well.
"""

def unpack(arr):
    new_arr = arr
    temp = []

    while (len(new_arr) == 1) and isinstance(new_arr[0], list):
        new_arr = new_arr[0]
    
    temp = new_arr.copy()

    depth = 1
    trim_depth = []


    while True:
        max_len = 0
        br = False

        for i in temp:
            try:
                max_len = max(len(i), max_len)
            except:
                br = True
                break

        if br:
            break
        
        orig_len = len(temp)

        for i in range(orig_len):
            elem = temp.pop(0)
            try:
                for i in elem:
                    temp.append(i)
            except:
                br = True
                break
        if br:
            break
        
        if max_len == 1:
            trim_depth.append(depth)
        depth += 1

    for i in trim_depth[::-1]:
        new_arr = trim(new_arr, i)
    
    return new_arr

#Auxilliary function to unpack every subarray at a given depth in the overall array
def trim(arr, depth, level = 1):
    # print("depth: " + str(depth) +", level: " + str(level))
    # print(arr)
    # print(len(arr))
    temp = []
    
    if depth == level:
        for i in arr:
            try:
                temp.append(i[0])
            except:
                temp.append([])
    else:
        for i in arr:
            new = trim(i, depth, level + 1)
            temp.append(new)
    return temp
        

"""
Function to print multi-tiered arrays using new lines and indentation for each level.
"""

def pretty_print(arr, level = 0):
    spacer = str(level) + "   "*level
    print(spacer + "[")

    if len(arr) == 0:
        print(spacer + str(arr))
    else:
        if isinstance(arr[0], list):
            for i in range(len(arr)):
                pretty_print(arr[i], level + 1)
        else:
            string = str(arr)
            print(spacer + string)
    

    print(spacer + "],")

"""
Composite function to compare data generated by python code to data generated by matlab code.
Function parses data using python functions, converts them to indices.
Imports matlab data, modifies both python and matlab data to have the same structure.
First tries to find perfect matches for trips between datasets.
Second tries to find matlab data that may be clipped to match python data.
Then reports un-resolved data.
"""
def compare_data(trips = [], matlab_idxs = [], clip_trips = True, trim_zeros = False):
    if trips == []:
        trips = get_trips_quick()
    
    python_idxs = trips_to_idxs(trips)
    if matlab_idxs == []:
        matlab_idxs = get_matlab_idxs(clip_trips = clip_trips, trim_zeros = trim_zeros)

    python_unmatched, matlab_unmatched = find_unmatched(python_idxs, matlab_idxs)

    python_matched = [[] for i in range(len(python_unmatched))]
    matlab_matched = [[] for i in range(len(matlab_unmatched))]

    for i in range(len(python_idxs)):
        temp = [x for x in python_idxs[i] if x not in python_unmatched[i]]
        python_matched[i] = temp
    
    for i in range(len(matlab_idxs)):
        temp = [x for x in matlab_idxs[i] if x not in matlab_unmatched[i]]
        matlab_matched[i] = temp
    
    print("Unmatched pre-find_close: ")
    py_temp = [len(x) for x in python_unmatched]
    mat_temp = [len(x) for x in matlab_unmatched]
    print("     python: " + str(sum(py_temp)))
    print(py_temp)
    print("     matlab: " + str(sum(mat_temp)))
    print(mat_temp)
    print()
    python_close, matlab_close = find_close(python_unmatched.copy(), matlab_unmatched.copy())


    print("Num Total: ")
    py_temp = [len(x) for x in python_idxs]
    mat_temp = [len(x) for x in matlab_idxs]
    print("     python: " + str(sum(py_temp)))
    print(py_temp)
    print("     matlab: " + str(sum(mat_temp)))
    print(mat_temp)
    print()
    print("Num Matched: ")
    py_temp = [len(x) for x in python_matched]
    mat_temp = [len(x) for x in matlab_matched]
    print("     python: " + str(sum(py_temp)))
    print(py_temp)
    print("     matlab: " + str(sum(mat_temp)))
    print(mat_temp)
    print()
    print("Num Unmatched: ")
    py_temp = [len(x) for x in python_unmatched]
    mat_temp = [len(x) for x in matlab_unmatched]
    print("     python: " + str(sum(py_temp)))
    print(py_temp)
    print("     matlab: " + str(sum(mat_temp)))
    print(mat_temp)
    print()
    print("Num Close: ")
    py_temp = [len(x) for x in python_close]
    mat_temp = [len(x) for x in matlab_close]
    print("     python: " + str(sum(py_temp)))
    print(py_temp)
    print("     matlab: " + str(sum(mat_temp)))
    print(mat_temp)
    print()

    return python_unmatched, matlab_unmatched, python_close, matlab_close


def find_unmatched(python_idxs, matlab_idxs):
    python_unmatched = []
    matlab_unmatched = []
    
    for i in range(len(python_idxs)):
        py_user = python_idxs[i].copy()
        mat_user = matlab_idxs[i].copy()
        py_idx = 0
        mat_idx = 0
        while py_idx < len(py_user):
            py_trip = py_user[py_idx]
            br = False
            for j in range(mat_idx, len(mat_user)):
                mat_trip = mat_user[j]
                if set(py_trip) == set(mat_trip):
                    py_user.remove(py_trip)
                    mat_user.remove(mat_trip)
                    mat_idx = j
                    br = True
                    break
            if not br:
                py_idx += 1
        python_unmatched.append(py_user)
        matlab_unmatched.append(mat_user)
    
    return python_unmatched, matlab_unmatched

"""
Function taking in unmatched python and matlab data and finding close matches, defined by two comperable trips with less than 20 different data points.
"""
def find_close(python_unmatched, matlab_unmatched):
    python_close = [[] for i in range(len(python_unmatched))]
    matlab_close = [[] for i in range(len(matlab_unmatched))]
    for i in range(len(python_unmatched)):
        py_idx = 0
        while py_idx < len(python_unmatched[i]):
            br = False
            for j in range(len(matlab_unmatched[i])):
                py_diff = list(set(python_unmatched[i][py_idx]) - set(matlab_unmatched[i][j]))
                mat_diff = list(set(matlab_unmatched[i][j]) - set(python_unmatched[i][py_idx]))
                if len(py_diff) < 10 and len(mat_diff) < 10:
                    python_close[i].append(python_unmatched[i][py_idx])
                    matlab_close[i].append(matlab_unmatched[i][j])
                    python_unmatched[i].remove(python_unmatched[i][py_idx])
                    matlab_unmatched[i].remove(matlab_unmatched[i][j])
                    br = True
                    break
            if not br:
                py_idx += 1
    return python_close, matlab_close


"""
Function to compare lengths of sets of trips, displaying unique points on each side and differences in length
"""
def compare_trips(py_trips, mat_trips):
    print("py_length -- py_unique, mat_unique -- mat_length")
    for i in range(min(len(py_trips), len(mat_trips))):
        left = list(set(py_trips[i]) - set(mat_trips[i]))
        right = list(set(mat_trips[i]) - set(py_trips[i]))

        print(str(len(py_trips[i])) + " -- " + str(len(left)) + ", " + str(len(right)) + " -- " + str(len(mat_trips[i])))



"""
Function to get a table of relevant rows given a list of indices in the master SBLE sheet.
"""
def idx_to_table(idxs):
    table = pd.DataFrame()
    sble = get_sble_data()
    for i in idxs:
        table = pd.concat([table, sble.iloc[i]], axis = 1)
    return table.transpose(copy = True)

"""
Composite function to load LL and LLpt structs, and merge them to compare Python and Matlab data parsing
"""
def get_matlab_idxs(include_pretrip = True, clip_trips = True, trim_zeros = True, flatten = False):
    ll = unpack(load_struct('LL'))

    if include_pretrip:
        llpt = unpack(load_struct('LLpt'))
        for i in range(len(ll)):
            for j in range(len(ll[i])):
                ll[i][j] += llpt[i][j]
                ll[i][j] = sort(list(set([x-1 for x in sort(ll[i][j])])))
    
    if clip_trips:
        temp = unpack(load_struct("nClips"))
        nClips = []
        for i in temp:
            arr = []
            for j in i:
                if j != []:
                    arr.append(j)
                else:
                    arr.append(0)
            nClips.append(arr)

        ll_new = []
        for x in range(len(ll)):
            user = ll[x]
            temp = []
            idx = 0
            while idx < len(user):
                if nClips[x][idx] == 0:
                    temp.append(user[idx])
                else:
                    clipped = user[idx]
                    for i in range(1, nClips[x][idx]+1):
                        clipped.extend(user[idx + i])
                    temp.append(clipped)
                    idx = idx + nClips[x][idx]
                idx += 1
            ll_new.append(temp)
        ll = ll_new

    if trim_zeros:
        i = 0
        while i < len(ll):
            j = 0
            while j < len(ll[i]):
                if len(ll[i][j]) == 0:
                    ll[i].pop(j)
                else:
                    j += 1
            i += 1


    if flatten:
        aux = []
        for i in ll:
            for j in i:
                aux.append(j)
        ll = aux
    
    return ll

"""
Function to evaluate the equivalency between a table of SBLE data and a list of SBLE indices.
"""
def check_trip_eq(trip, idxs):
    trip_list = trip.data["index"].tolist()
    return trip_list == idxs

"""
Function to turn a trips object into a multi-tiered array similar to how it is stored in Matlab.
"""
def trips_to_idxs(trips):
    names = get_users(get_sble_data())
    names = sort(names)

    arr = [[] for i in range(len(names))]

    for i in trips:
        if i.data.shape[0] > 0:
            data = i.data["index"].tolist()
            arr[names.index(i.user)].append(data)
        else:
            arr[names.index(i.user)].append([])
    
    return arr


"""
Function to try and replicate the prediction procedure from the end of the Matlab code.
"""
def check_acc():
    minacc = -1000000
    ith1 = list(range(-15,0))
    ith2 = list(range(0,15))

    totDiff = unpack(load_struct('totDiff'))
    ba = []
    for th1 in ith1:
        for th2 in ith2:
            th = [-1000000, th1, th2, 1000000]
            a = [[0 for x in range(3)] for y in range(3)]
            acc = 0
            for i in range(3):
                for j in range(3):
                    in_range = [1 if totDiff[i][x] > th[j] and totDiff[i][x] < th[j+1] else 0 for x in range(len(totDiff[i]))]
                    a[i][j] = sum(in_range)/len(totDiff[i])
                acc = acc + a[i][i]*(len(totDiff[i])/sum([len(totDiff[0]), len(totDiff[1]), len(totDiff[2])]))
            if acc > minacc:
                bth1 = th1
                bth2 = th2
                minacc = acc
                ba = a
    
    tot_len = sum([len(x) for x in totDiff])
    rand_acc = (len(totDiff[0])/tot_len)**2 + (len(totDiff[1])/tot_len)**2 + (len(totDiff[2])/tot_len)**2
    best_acc = minacc
    best_range = [bth1, bth2]

    print("Random accuracy: " + str(rand_acc))
    print("Best accuracy: " + str(best_acc))
    print("Best range: " + str(best_range))
    for i in ba:
        print(i)

