import scipy.io as spio
from pathlib import Path 
import matplotlib.pyplot as plt 
import numpy as np
from scipy.signal import find_peaks
import scipy.ndimage

def import_dataset(filepath):
    mat = spio.loadmat(str(filepath), squeeze_me=True)
    # d = mat['d']
    # _index = mat['Index']
    # _class = mat['Class']

    return mat


def plot_data(data, peaks):
    plt.plot(data)
    plt.plot(peaks, data[peaks], "x")
    plt.plot(np.zeros_like(data), "--", color="gray")
    plt.show()



def find_boundaries(peak, data, plot=False):
    """Function to find from the peak of the signal, where the beginning and end of the peak is. 

    Checks cases: 

    > If we are going up instead of down, we are climbing another peak. 
    Stop and find the minima between the peaks and that is either our begin or end 
    index depending if its before or after the peak.

    > If we have a dip in the signal (the signal falls below zero significantly), 
    keep exploring until the signal is zero or above after the dip peak index.

    > If we cross zero and neither of the above apply, that's the end of that index.

    :return: Begin and end index
    :rtype: tuple
    """

    # we shift data up or down depending on the average noise floor
    _range = 50


    dataset = data[np.clip(peak - _range, 0, None): np.clip(peak + _range, None, len(data) - 1)]  # get the local data around the peak 
    # clipping to prevent negative indexes and indexing past max index. 
    mean, std = np.mean(dataset), np.std(dataset)  # find the mean and std of the local data
    
    filtered_dataset = dataset[dataset < (mean + 0.5 * std)]  # exclude peaks and dips to focus on the noise
    filtered_dataset = dataset[dataset > -(mean + 0.5 * std)]  # +- 0.5 std deviations around the mean is excluded
    new_mean = np.mean(filtered_dataset)  # find the mean of just the noise

    if plot:
        print(mean, std, new_mean)
        plt.plot(filtered_dataset)
        plt.show()

    dataset = dataset - new_mean  # adjust for the base noise level and zero the noise. 
    zero_point = new_mean  # treat this value as the new zero. 

    begin, end = peak - 1, peak + 1  # indexes
    ret_begin, ret_end = None, None  # return values
    queue_size = 12  # length of the last checked gradients (seeing if the gradient is up or down)
    last_minimums, last_maximums = [True] * queue_size, [True] * queue_size  # to check if we start climbing another peak 

    msr = 40  # minima search range (finding minima lower than -1)
    dips, _ = find_peaks(dataset[_range - msr:_range + msr], height=0.05, width=5)  # find the dips in the signal


    while True:
        last_minimums.append(data[begin] < data[begin + 1])  # if our current position is less than the previous position
        last_maximums.append(data[end] < data[end - 1])  # if our current position is less than the previous position

        if len(last_maximums) > queue_size: last_maximums.pop(0)  # making sure the queue doesn't exceed limits
        if len(last_minimums) > queue_size: last_minimums.pop(0)

        begin_gradient = sum(last_minimums) > len(last_minimums)//2 
        end_gradient = sum(last_maximums) > len(last_maximums)//2  # if more than half is true, we are going down properly
        
        if begin <= 0:  # if we hit 0 index
            ret_begin = 0
        elif not begin_gradient:
            ret_begin = np.argmin(data[begin:peak]) + begin  
            # if we are going up, stop, find the minimum between the peak and the current index, and that is the begin index.
        elif data[begin] <= zero_point:  # we have crossed zero, stop search
            ret_begin = begin
        else:
            begin -= 1

        
        if end >= len(data) - 1:  # if we exceed the len of the array
            ret_end = len(data) - 1
        elif not end_gradient:
            ret_end = np.argmin(data[peak:end]) + peak     
        elif len(dips) == 0 and data[end] <= zero_point:  # if no dips in the signal, just check for 0
            ret_end = end
        elif len(dips) > 0:  # otherwise wait until the signal is >= 0 after the minima
            if data[end] >= zero_point and end > peak + (dips[0] - msr):
                ret_end = end
            else:
                end += 1
        else:
            end += 1

        # print(begin, end, ret_begin, ret_end, dips, sep="|")

        if ret_begin is not None and ret_end is not None:
            break

    return ret_begin, ret_end


def filter_peaks(peaks, data, plot=False):
    ret = []
    for peak in peaks:
        # print(peak)
        begin, end = find_boundaries(peak, data, plot)
        ret.append([begin, end]) 
        # print(f"begin, end: ", begin, end)
    return ret


def pad_data(indexes, data, shape=(100,), training_data=False):
    
    padded_data = []
    
    # if not training_data:
    #     for begin, end in indexes:
    #         spike = data[begin:end]
    #         output = np.zeros(shape)
    #         insert_idx = (len(output) - len(spike)) // 2 

    #         output[insert_idx : insert_idx + len(spike)] = spike
    #         padded_data.append(output)

    for begin, end, *classification in indexes:
        spike = data[begin:end]
        output = np.zeros(shape)
        insert_idx = (len(output) - len(spike)) // 2 

        output[insert_idx : insert_idx + len(spike)] = spike

        if training_data:
            padded_data.append([output, classification])
        else:
            padded_data.append(output)

    return padded_data


def prepare_data(input_data, params):

    h, w, d = params["peaks_criterion"]
    sig = params["gaussian_sigma"]

    rescaled = input_data / np.max(input_data)  # normalise data to [-1, 1], scaling with the maximum positive peak value
    filtered = scipy.ndimage.gaussian_filter1d(rescaled, sigma=sig)  # gaussian filter to mitigate some noise
    
    # IMPLEMENT A BUTTERWORTH + WAVELET FILTER AND REVIEW SIGNALS 
    
    peaks, _ = find_peaks(filtered, height=h, width=w, distance=d)

    plot_data(filtered, peaks)
    indexes = filter_peaks(peaks, filtered, plot=False)  
    indexes = np.array(pad_data(indexes, filtered, shape=(100,)))

    retdict = {
        "data": indexes,
        "peaks": peaks,
        "filtered_data": filtered
    }

    return retdict



def import_matdata():
    mat_data = {"D1": "coursework_c/datasets/D1.mat",
        "D2": "coursework_c/datasets/D2.mat",
        "D3": "coursework_c/datasets/D3.mat",
        "D4": "coursework_c/datasets/D4.mat",
        "D5": "coursework_c/datasets/D5.mat",
        "D6": "coursework_c/datasets/D6.mat"
        }
    
    for fname, fp in mat_data.items():
        mat_data[fname] = import_dataset(Path(fp).absolute())

    return mat_data


def main():

    #TODO: MEAN FILTER OF KERNEL 3 TO AVERAGE OUT THE NOISE A BIT MAYBE. 

    mat_data = {"D1": "coursework_c/datasets/D1.mat",
            "D2": "coursework_c/datasets/D2.mat",
            "D3": "coursework_c/datasets/D3.mat",
            "D4": "coursework_c/datasets/D4.mat",
            "D5": "coursework_c/datasets/D5.mat",
            "D6": "coursework_c/datasets/D6.mat"
            }
    
    for fname, fp in mat_data.items():
        mat_data[fname] = import_dataset(Path(fp).absolute())
    
    x = mat_data["D1"]["d"]
    # print(np.mean(D1), np.std(D1))

    for d in mat_data:
        print(np.min(mat_data[d]["d"]), np.max(mat_data[d]["d"]))


    rescaled = x / np.max(x)   # scale down so the range goes from -1 to 1.
    
    noise_floor = np.mean(x); print(noise_floor)
    # print(np.min(rescaled), np.max(rescaled))
    
    # weights = np.array([0.15, 0.7, 0.15])# np.full((3,), 1.0/3)
    # x = scipy.ndimage.convolve(x, weights, mode="reflect")


    peaks, _ = find_peaks(rescaled, height=0.1, width=5, distance=10)

    print(len(peaks), len(mat_data["D1"]["Index"]))
    print(mat_data["D1"]["Index"].shape)

    arr = [(x, y) for x, y in zip(mat_data["D1"]["Index"], mat_data["D1"]["Class"])]
    arr = np.array(sorted(arr, key=lambda x: x[0]))
    _indexes = filter_peaks(peaks, x)  # find the 

    print(arr[:20])
    plot_data(rescaled, peaks)
    index_dict = {}

    for idx, classification in arr:
        if idx//10000 not in index_dict:
            index_dict[idx//10000] = [(idx%10000, classification)]
        else:
            index_dict[idx//10000].append((idx%10000, classification))


    for i, idxs in enumerate(_indexes):
        begin, _ = idxs
        floor, mod = begin//10000, begin%10000

        for pk, classification in index_dict[floor]:
            if abs(mod - pk) < 30:
                _indexes[i].append(classification)
    
    
    _indexes = filter(lambda x: len(x) == 3, _indexes)
    _indexes = [_ for _ in _indexes]

    _indexes = pad_data(_indexes, rescaled, shape=(100,), training_data=True)

    print(len([_ for _ in _indexes]))  
    
    train_data = np.array([_data for _data, _class in _indexes])
    train_labels =  np.array([_class[0] - 1 for _data, _class in _indexes])  # We shift the class label from 1-5 to 0-4 for tensorflow

    print(train_data.shape)
    
    # plot_data(rescaled, peaks)

    test_data, test_labels = train_data, train_labels

    from CNN import model

    model.fit(train_data, train_labels, batch_size=32, epochs=10)
    test_loss, test_acc = model.evaluate(test_data,  test_labels, verbose=2)
    print('\nTest accuracy:', test_acc)

    params = {"peaks_criterion": (0.1, 5, 10), "gaussian_sigma": 1}

    
    D2_data = prepare_data(mat_data["D2"]["d"], params)
    test_data_D2, D2_peaks = D2_data["data"], D2_data["peaks"]
    results = model.predict(test_data_D2)

    print(f"results: {np.argmax(results, axis=1) + 1}")
    print(np.argmax(results[:40], axis=1) + 1)

    plot_data(D2_data["filtered_data"], D2_peaks)

if __name__ == "__main__":
    main()