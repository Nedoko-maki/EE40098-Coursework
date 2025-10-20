import numpy as np
import math
import scipy
from ANN import NeuralNetwork
import matplotlib.pyplot as plt 

def train(neural_network, output_nodes, training_data):
     # Load the MNIST 100 training samples CSV file into a list
    
    print("Starting training")

    # Train the neural network on each training sample
    for idx, record in enumerate(training_data):
        # Split the record by the commas
        # if not isinstance(record, (list, np.ndarray)):
        #     all_values = record.split(",")
        # else:
        #     all_values = record

        all_values = record.split(",")
        # Scale and shift the inputs from 0..255 to 0.01..1
        inputs = (np.asarray(all_values[1:], dtype=np.float64) / 255.0 * 0.99) + 0.01
        # Create the target output values (all 0.01, except the desired label which is 0.99)
        targets = np.zeros(output_nodes) + 0.01
        # All_values[0] is the target label for this record
        targets[int(all_values[0])] = 0.99
        # Train the network
        
        if idx % 10000 == 0:
            print(f"Training {idx} out of {len(training_data)}")
        
        neural_network.train(inputs, targets)


def test(neural_network, test_data_list, verbose=False):
    # Scorecard list for how well the network performs, initially empty
    scorecard = []
    incorrect_classifications = []
    # Loop through all of the records in the test data set
    for record in test_data_list:
        # Split the record by the commas
        all_values = record.split(",")
        # The correct label is the first value
        correct_label = int(all_values[0])
        # Scale and shift the inputs
        inputs = (np.asarray(all_values[1:], dtype=np.float64) / 255.0 * 0.99) + 0.01
        # Query the network
        outputs = neural_network.query(inputs)
        # The index of the highest value output corresponds to the label
        label = np.argmax(outputs)

        if verbose:
            print(f"Correct label: {correct_label}")
            print(f"Network label: {label}")

        # Append either a 1 or a 0 to the scorecard list
        if (label == correct_label):
            scorecard.append(1)
        else:
            scorecard.append(0)
            incorrect_classifications.append({"image_data": np.asarray(all_values[1:], dtype=np.float64).reshape((28, 28)),
                                              "correct_label": correct_label,
                                              "network_label": label})


            # Calculate the performance score, the fraction of correct answers


    scorecard_array = np.asarray(scorecard)
    print("Performance = ", (scorecard_array.sum() / scorecard_array.size)*100, "%")

    return incorrect_classifications


def display_images(dict_list):
    total_plots = len(dict_list)
    row = 3
    col = math.ceil(total_plots / row)

    for idx, data in enumerate(dict_list):
        ax = plt.subplot(row, col, idx + 1)
        clabel, mlabel = data["correct_label"], data["network_label"]
        ax.set_title(f"Correct label: {clabel}, Model label: {mlabel}")
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        ax.imshow(data["image_data"])

    plt.show()


def preprocess_data(training_data):
    for idx, record in enumerate(training_data):
        # Split the record by the commas
        all_values = record.split(",")
        # Scale and shift the inputs from 0..255 to 0.01..1
        inputs = (np.asarray(all_values[1:], dtype=np.float64)) # / 255.0 * 0.99) + 0.01
        # blurred = scipy.ndimage.gaussian_filter(inputs, sigma=5)
        # alpha = 30
        # sharp = inputs + alpha * (inputs - scipy.ndimage.gaussian_filter(blurred, 1))

        image = np.reshape(inputs, (28, 28))

        sx = scipy.ndimage.sobel(image, axis=0, mode='constant')
        sy = scipy.ndimage.sobel(image, axis=1, mode='constant')
        image = np.hypot(sx, sy)

        ret = image.flatten()
        #print(sharp.shape, sharp)
        ret = np.concatenate(([all_values[0]], ret))  # prepend with removed value
        training_data[idx] = ret

        if idx % 10000 == 0:
            print(f"Preprocessing {idx} out of {len(training_data)}")
        

    return training_data


def main():
    input_nodes = 784
    hidden_nodes = 250
    output_nodes = 10
    learning_rate = 0.11

    NN = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)
    
    train_fp = "datasets/mnist_train.csv"
    test_fp = "datasets/mnist_test.csv"
    # train_fp = "fashion_mnist_train.csv"
    # test_fp = "fashion_mnist_test.csv"

    with open(train_fp, "r", encoding="utf-8") as fs:
        training_data_list = fs.readlines()

        # Load the MNIST test samples CSV file into a list
    with open(test_fp, "r") as fs:
        test_data_list = fs.readlines()

    # preprocess_data(training_data_list)
    for i in range(5):
        train(NN, output_nodes, training_data_list)
        test(NN, test_data_list)
    
    ret = test(NN, test_data_list)
    display_images(ret[1:12])

    # There is like 1 or 2 pretty much impossible images, the 5 is completely malformed. The rest is difficult but doable,
    # There is simply not enough training data. 

    # oddly enough with the full training set, it got 100% on the MNIST10 data set. 
    # The perf on the full data sets is around 94% with table 5.1 values. More data means more reference points and patterns the model can 
    # interpret and figure out.     

    # i: 784, h: 150: o: 10, l: 0.3 gives 94.81%
    # i: 784, h: 200: o: 10, l: 0.3 gives 95.41% ( took a lot longer )
    # i: 784, h: 100: o: 10, l: 0.5 gives 92.25%
    # i: 784, h: 100: o: 10, l: 0.25 gives 94.9%
    # i: 784, h: 150: o: 10, l: 0.25 gives 95.29%
    # i: 784, h: 150: o: 10, l: 0.2 gives 95.66%
    # i: 784, h: 150: o: 10, l: 0.15 gives 95.81%
    # i: 784, h: 150: o: 10, l: 0.1 gives 95.77%
    # i: 784, h: 200: o: 10, l: 0.125 gives 96.12%
    # i: 784, h: 300: o: 10, l: 0.125 gives 96.09%
    # i: 784, h: 150: o: 10, l: 0.12 gives 97.04% (3 training iterations)
    # i: 784, h: 150: o: 10, l: 0.12 gives 97.23% (5 training iterations)
    # i: 784, h: 200: o: 10, l: 0.1 gives 97.61% (5 training iterations) (ironically the original setup works really well.)



    # Fashion dataset:
    # i: 784, h: 200: o: 10, l: 0.11 gives 83.59%
    # i: 784, h: 150: o: 10, l: 0.3 gives 71.0%
    # i: 784, h: 150: o: 10, l: 0.2 gives 75.41%
    # i: 784, h: 150: o: 10, l: 0.1 gives 82.08%
    # i: 784, h: 150: o: 10, l: 0.08 gives 82.98%
    # i: 784, h: 150: o: 10, l: 0.04 gives 83.93%
    # i: 784, h: 175: o: 10, l: 0.01 gives 83.16%
    # i: 784, h: 40: o: 10, l: 0.01 gives 80.99%
    # i: 784, h: 50: o: 10, l: 0.02 gives 82.38%
    # i: 784, h: 25: o: 10, l: 0.02 gives 82.47%
    # i: 784, h: 32: o: 10, l: 0.03 gives 83.0%
    # i: 784, h: 40: o: 10, l: 0.04 gives 83.4%

    # i: 784, h: 40: o: 10, l: 0.01 gives 80.51%
    # i: 784, h: 50: o: 10, l: 0.05 gives 83.69%
    # i: 784, h: 100: o: 10, l: 0.05 gives 83.85%
    # i: 784, h: 125: o: 10, l: 0.0475 gives 84.50%
    # i: 784, h: 50: o: 10, l: 0.05 gives 84.69% (4 training iterations)
    # i: 784, h: 125: o: 10, l: 0.048 gives 85.54% (4 training iterations)
    # i: 784, h: 125: o: 10, l: 0.04 gives 86.50% (4 training iterations)
    # i: 784, h: 150: o: 10, l: 0.04 gives 86.36% (5 training iterations)


if __name__ == "__main__":
    main()