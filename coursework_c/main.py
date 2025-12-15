from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as spio
from signal_processing import plot_data, prepare_data
from sklearn.model_selection import train_test_split


def import_dataset(filepath):
    """Import .mat data as a dictionary.

    :param filepath: filepath
    :type filepath: Path or str
    :return: dict of the mat data
    :rtype: dict
    """
    mat = spio.loadmat(str(filepath), squeeze_me=True)
    return mat


def export_dataset(filepath, indexes, labels):
    """Export the data as a .mat file. 

    :param filepath: filepath
    :type filepath: Path or str
    :param indexes: indexes of the peaks
    :type indexes: numpy array
    :param labels: list of classes
    :type labels: list
    """
    matdict = {"Index": np.array([i[0] for i in indexes]), "Class": labels}  # grab the beginning index + classification 
    spio.savemat(str(filepath), matdict)


def plot_all_stages(raw, info, norm, title="D1", n_samples=500, fs=25e3):
    """
    Plot raw, smoothed, and normalised signals for first n_samples.
    """
    raw_seg = raw[:n_samples]
    filt_seg = info["filtered"][:n_samples]
    norm_seg = norm[:n_samples]
 
    t = np.arange(n_samples) / fs
 
    plt.figure(figsize=(12, 8))
 
    # Raw
    plt.subplot(3, 1, 1)
    plt.plot(t, raw_seg)
    plt.title(f"{title} - Stage 0: Raw")
    plt.ylabel("Amplitude")
 
    # Stage 1 (Savitzky–Golay)
    plt.subplot(3, 1, 2)
    plt.plot(t, filt_seg)
    plt.title(f"{title} - Stage 1: Savitzky–Golay Smoothed")
    plt.ylabel("Amplitude")
 
    # Stage 3
    plt.subplot(3, 1, 3)
    plt.plot(t, norm_seg)
    plt.title(f"{title} - Stage 3: Normalised [-1, 1]")
    plt.ylabel("Amplitude")
    plt.xlabel("Time (s)")
 
    plt.tight_layout()
    plt.show()


def evaluate_dataset(model, dataset_name, dataset, plot=False):
    """
    Evaluate a dataset with the trained model. 
    
    :param model: CNN model
    :type model: keras.model.Model
    :param dataset_name: str of the dataset's name
    :type dataset_name: str
    :param dataset: dict of the dataset
    :type dataset: dict
    :param plot: plot a graph of the filtered data, defaults to False
    :type plot: bool, optional
    :return: classification results
    :rtype: numpy ndarray
    """


    probabilities = model.predict(dataset["data"])
    results = np.argmax(probabilities, axis=1) + 1  # shift the answers back from 0-4 to 1-5. 

    print(f"{dataset_name} results: {results}")
    print(np.argmax(probabilities[:50], axis=1) + 1)
    print(f"number of peaks = {len(dataset['peaks'])}")

    if plot: 
        plot_data(dataset["filtered_data"], dataset["peaks"], title=dataset_name)

    print_classes(results)

    return results


def print_classes(results):
    """Print how many peaks of each class is in a dataset's results. 

    :param results: array of classifications
    :type results: numpy ndarray
    """
    # Predict classes for each spike in D2
    if results.shape[0] > 0:
        
        print("\nPredicted class counts:")
        for _class in range(5):
            count = np.sum(results == _class + 1)
            print(f"Class {_class+1}: {count} spikes")
    else:
        print("No valid windows were created for prediction.")
    
    print("="*20)


def process_datasets(data_shape):
    """Preprocess the datasets, import them, and prepare them for training and classification. This includes filtering. 

    :param data_shape: the shape of the input data into the CNN model.
    :type data_shape: tuple
    :return: dict of the mat data
    :rtype: dict
    """

    mat_data = {"D1": "coursework_c/datasets/D1.mat",
                "D2": "coursework_c/datasets/D2.mat",
                "D3": "coursework_c/datasets/D3.mat",
                "D4": "coursework_c/datasets/D4.mat",
                "D5": "coursework_c/datasets/D5.mat",
                "D6": "coursework_c/datasets/D6.mat"}


    params = {"D1": {"peaks_criterion": (0.08, 5, 10), "filter_options": {}},
              "D2": {"peaks_criterion": (0.08, 5, 10), "filter_options": {"sg_poly": 4}},
              "D3": {"peaks_criterion": (0.08, 5, 10), "filter_options": {"sg_poly": 4}},
              "D4": {"peaks_criterion": (0.06*1.5, 5, 10), "scale": 1.5, "filter_options": {"do_wavelet": True, "swt_wavelet": "db4", "sg_poly": 5}},
              "D5": {"peaks_criterion": (0.16, 5, 10), "scale": 1.88, "filter_options": {"do_wavelet": True, "sg_poly": 5}},
              "D6": {"peaks_criterion": (0.2, 5, 10), "scale": 2, "filter_options": {"do_wavelet": True, "swt_wavelet": "coif3", "sg_poly": 4}}}

    # expected num of peaks:
    # D2 = 4092
    # D3 = 3420
    # D4 = 3078
    # D5 = 2957
    # D6 = 4500?

    for fname, fp in mat_data.items():
        mat_data[fname] = import_dataset(Path(fp).absolute())
        params[fname]["shape"] = data_shape
        data_dict = prepare_data(mat_data[fname]["d"], params[fname])
        mat_data[fname].update(data_dict)  # add to the existing dictionary

        print(f"Prepping dataset {fname}, peak len: {len(mat_data[fname]['peaks'])}")
        # plot_data(data_dict["filtered_data"], data_dict["peaks"])


    return mat_data


def prepare_train_data(mat_data):
    """Prepare the training data from D1. 

    :param mat_data: mat data of all the datasets
    :type mat_data: dict
    :return: training data and labels
    :rtype: tuple of arrays
    """

    arr = [(i, j) for i, j in zip(mat_data["D1"]["Index"], mat_data["D1"]["Class"])]
    arr = np.array(sorted(arr, key=lambda x: x[0]))

    index_dict = {}

    for idx, classification in arr:
        if idx//10000 not in index_dict:
            index_dict[idx//10000] = [(idx%10000, classification)]
        else:
            index_dict[idx//10000].append((idx%10000, classification))

    train_data, train_labels = [], []

    for idxs, peak_data in zip(mat_data["D1"]["indexes"], mat_data["D1"]["data"]):
        begin, *_ = idxs
        
        floor, mod = begin//10000, begin%10000

        for pk, classification in index_dict[floor]:
            if abs(mod - pk) < 15:
                train_data.append(peak_data)
                train_labels.append(classification)

    train_data = np.array(train_data)
    train_labels =  np.array(train_labels) - 1 # We shift the class label from 1-5 to 0-4 for tensorflow

    # print(train_data.shape)

    return train_data, train_labels


def main():
    from CNN import model_version_5

    data_shape = (100, 1)  # define the CNN model input size

    mat_data = process_datasets(data_shape)  # import the data and prepare them
    train_data, train_labels = prepare_train_data(mat_data)  # prepare the D1 data for training
    train_data = train_data.reshape(-1, *data_shape)  # reshape just in case

    model = model_version_5(signal_length=data_shape)  # choose the model and define the shape of inputs. 

    train_data, temp_data, train_labels, temp_labels = train_test_split(
    train_data, train_labels, test_size=0.20, random_state=42, shuffle=True, stratify=train_labels)  # split the training data into 80% training and 20% validation datasets

    # Split temporary into 50% val + 50% test
    val_data, test_data, val_labels, test_labels = train_test_split(  
    temp_data, temp_labels, test_size=0.50, random_state=42,
    shuffle=True, stratify=temp_labels)


    print(np.unique(train_labels, return_counts=True))  # check if the data is split correctly. 
    print(np.unique(val_labels, return_counts=True))
    print(np.unique(test_labels, return_counts=True))
    # print(train_data.shape)
    # print(train_labels.shape)

    history = model.fit(
    train_data,
    train_labels,
    batch_size=64,
    epochs=200,
    validation_data=(val_data, val_labels),
    shuffle=True
    )  # train/fit the model to the training data and labels

    test_loss, test_acc = model.evaluate(train_data, train_labels, verbose=2)  # evaluate the model on D1. 

    print('\nTest accuracy:', test_acc)
    

    evaluate_dataset(model, "D1", mat_data["D1"], plot=True)  # classify all the datasets
    D2_labels = evaluate_dataset(model, "D2", mat_data["D2"], plot=False)
    D3_labels = evaluate_dataset(model, "D3", mat_data["D3"], plot=False)
    D4_labels = evaluate_dataset(model, "D4", mat_data["D4"], plot=True)
    D5_labels = evaluate_dataset(model, "D5", mat_data["D5"], plot=True)
    D6_labels = evaluate_dataset(model, "D6", mat_data["D6"], plot=True)

    export_dataset(Path("coursework_c/outputs/D2.mat").absolute(), mat_data["D2"]["indexes"], D2_labels)  # save all the data into .mat files. 
    export_dataset(Path("coursework_c/outputs/D3.mat").absolute(), mat_data["D3"]["indexes"], D3_labels)
    export_dataset(Path("coursework_c/outputs/D4.mat").absolute(), mat_data["D4"]["indexes"], D4_labels)
    export_dataset(Path("coursework_c/outputs/D5.mat").absolute(), mat_data["D5"]["indexes"], D5_labels)
    export_dataset(Path("coursework_c/outputs/D6.mat").absolute(), mat_data["D6"]["indexes"], D6_labels)




if __name__ == "__main__":
    main()

    # TODO: 
    # D5-6,
    # Bandpass -> Wavelet -> MAD? -> CNN to classify if its a peak or not. 