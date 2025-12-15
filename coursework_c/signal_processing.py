import scipy.io as spio
from pathlib import Path 
import matplotlib.pyplot as plt 
import numpy as np
from scipy.signal import find_peaks, savgol_filter, filtfilt, medfilt, resample
from scipy.signal import butter as butter_filter
from filtering import denoise_pipeline, estimate_band_from_psd


def plot_data(data, peaks, title=""):
    plt.plot(data)
    plt.plot(peaks, data[peaks], "x")
    plt.plot(np.zeros_like(data), "--", color="gray")
    plt.title(title)
    plt.show()


def outside_in_search(data, 
                      outer_peak_index, 
                      direction, 
                      zero_point):

    queue_size = 5  # length of the last checked gradients (seeing if the gradient is up or down)
    last_gradients = [True] * queue_size

    current_index = outer_peak_index

    while True:
        
        if direction == "forward": 
            last_gradients.append(data[current_index] < data[current_index + 1])  # if our current position is less than the previous position
        elif direction == "backward":
            last_gradients.append(data[current_index] < data[current_index - 1])  # if our current position is less than the previous position

        if len(last_gradients) > queue_size: last_gradients.pop(0)  # keep len of queue as queue-sized
        current_gradient = sum(last_gradients) > len(last_gradients)//2   # check if the queue is more True or False.
        
        if current_index <= 0:  # if we hit 0 index
            return 0
        elif not current_gradient:  # we are no longer going towards zero.
            if direction == "backward":  # search backwards (peak in front of current peak)
                ret_idx = current_index + np.argmin(data[current_index:outer_peak_index])
            elif direction == "forward":  # search forwards (peak behind current peak)
                # print(outer_peak_index, current_index)
                ret_idx = outer_peak_index + np.argmin(data[outer_peak_index:current_index])   
            return ret_idx
            # if we are going up, stop, find the minimum between the peak and the current index, and that is the begin index.
        elif data[current_index] <= zero_point:  # we have crossed the zero point, stop search
            return current_index
        else:
            if direction == "forward":
                current_index += 1
            elif direction == "backward":
                current_index -= 1  
        

def inside_out_search(data, 
                      peak_index, 
                      direction, 
                      _range,
                      zero_point):

    if direction == "forward":
        
        for i in range(_range):
            if data[peak_index - i] <= zero_point:
                return peak_index - i


        return peak_index - _range  # just return all if nothing is found. 
    
    elif direction == "backward":
        return peak_index + _range

    # TODO: IMPLEMENT THIS, ~BUT TEST THE BUTTERWORTH + WAVELET FILTER FIRST AND SEE ALL THE DATASETS ON IT. TUNE IT. 


def find_boundaries(peak_index, peaks, data, shape=(100,), plot=False):
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

    if peak_index == 0:
        prev_peak = None
    else:
        prev_peak = peaks[peak_index - 1]
    
    if peak_index == len(peaks) - 1:
        next_peak = None
    else:
        next_peak = peaks[peak_index + 1]

    current_peak = peaks[peak_index]  # get the previous, current and next peak
    
    # we shift data up or down depending on the average noise floor
    _range = shape[0] // 2
    extra_peaks = [False, False]

    # if the adjacent peaks are within the range of the data collection, take note. 

    if prev_peak is not None:
        if abs(prev_peak - current_peak) < _range:
            extra_peaks[0] = True
    
    if next_peak is not None:
        if abs(next_peak - current_peak) < _range:
            extra_peaks[1] = True

    dataset = data[np.clip(current_peak - _range, 0, None): np.clip(current_peak + _range, None, len(data) - 1)]  # get the local data around the peak 
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

    if extra_peaks[0]:
        begin = outside_in_search(data, prev_peak, "forward", zero_point) # outside-in search for beginning index
    else:
        begin = inside_out_search(data, current_peak, "forward", _range, zero_point) # inside-out search for beginning index
        
    if extra_peaks[1]:
        end = outside_in_search(data, next_peak, "backward", zero_point) # outside-in search for end index
    else:
        end = inside_out_search(data, current_peak, "backward", _range, zero_point) # inside-out search for end index

    begin = np.clip(begin, 0, None)  # clip begin and end 
    end = np.clip(end, None, len(data) - 1)
    return begin, end, current_peak


def search_peaks(peaks, data, shape=(100,), plot=False):
    ret = []
    for i in range(len(peaks)):
        # print(peak)
        begin, end, centre = find_boundaries(i, peaks, data, shape=shape, plot=plot)
        ret.append([begin, end, centre]) 
        # print(f"begin, end: ", begin, end)
    return ret


def normalise_peak(peak):
    mean, std = np.mean(peak), np.std(peak)
    
    noise = peak.copy()
    noise = peak[peak < (mean + 0.5 * std)]  # exclude peaks to focus on the noise
    new_mean = np.mean(noise)  # find the mean of just the noise

    peak = peak - new_mean  # shift the peak so it sits more on the zero line. 
    
    # plt.plot(peak)
    # plt.show()
    return peak


def random_gain(x, low=0.8, high=1.2, p=0.2):
    if np.random.rand() < p:
        f = np.random.uniform(low, high)
        return x * f
    return x


def time_stretch_resample(x, factor_low=0.92, factor_high=1.08, p=0.2):
    if np.random.rand() < p:
        s = np.random.uniform(factor_low, factor_high)
        new_len = max(2, int(len(x) * s))
        new_len = np.clip(new_len, None, 98)

        stretched = resample(x, new_len)
        # Resample/pad back to out_len
        
        return stretched
    return x


def vary_shape(peak):
    peak = random_gain(peak)
    peak = time_stretch_resample(peak)
    return peak

def pad_data(indexes, data, shape=(100,), training_data=False):
    
    padded_data = []

    if len(shape) == 2:
            shape = (shape[0],)  # if shape is (100, 1), change to (100,)

    for begin, end, centre, *classification in indexes:
        peak = data[begin:end]
        peak = normalise_peak(peak)
    
        if training_data:
            peak = vary_shape(peak)

        output = np.zeros(shape)
        insert_idx = (len(output) // 2) - abs(centre - begin) - 1   # centre the peak's centre in the dataset.
        # insert_idx = (len(output) - len(peak)) // 2
        
        if len(peak) + insert_idx > len(output):
            peak = peak[: -(abs(len(output) - (len(peak) + insert_idx)) + 1)]

        try:
            output[np.clip(insert_idx, 0, None) : np.clip(insert_idx + len(peak), None, len(output))] = peak
        except Exception as err:
            print(err, insert_idx, len(peak))   
            
        # plt.plot(output)
        # print(output.shape)
        # plt.show()

        if training_data:
            padded_data.append([output, classification])
        else:
            padded_data.append(output)

    return padded_data


def flatten_data(input_data, kernel_size=171):
    """Reduce low frequency variation over the whole dataset

    :param input_data: _description_
    :type input_data: _type_
    :return: _description_
    :rtype: _type_
    """
    median = medfilt(input_data, kernel_size=kernel_size)
    return input_data - median


def prepare_data(input_data, params):

    h, w, d = params["peaks_criterion"]
    filter_options = params["filter_options"]

    data_shape = params["shape"]
    rescaled = input_data / np.max(input_data)  # normalise data to [-1, 1], scaling with the maximum positive peak value

    if "do_bandpass" in filter_options: 
        freqs, Pxx_mean, band_energy, band_db = estimate_band_from_psd(rescaled, fs=25e3, plot=False)
        filter_options["bandpass"] = band_db    
    
    filtered = denoise_pipeline(rescaled, **filter_options)
    filtered = flatten_data(filtered)
    
    # IMPLEMENT A BUTTERWORTH + WAVELET FILTER AND REVIEW SIGNALS 

    if "scale" in params:
        filtered = params["scale"] * filtered 
        print("Rescaled.") 
        # due to the way I have filtered the data, I rescale the data before I flatten it, leading to variations in the size.
        # this is to correct the squished data


    peaks, peak_properties = find_peaks(filtered, height=h, width=w, distance=d)
    
    # print(peak_properties, peak_properties["left_bases"], peak_properties["right_bases"])
    # plot_data(filtered, peaks)

    indexes = search_peaks(peaks, filtered, shape=data_shape, plot=False)  
    isolated_peaks = np.array(pad_data(indexes, filtered, shape=data_shape))

    retdict = {
        "data": isolated_peaks,
        "indexes": indexes,
        "peaks": peaks,
        "filtered_data": filtered
    }

    return retdict


def main():
    pass


if __name__ == "__main__":
    main()