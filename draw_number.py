# Import numpy for arrays and matplotlib for drawing the numbers
import numpy
import matplotlib.pyplot as plt
# Open the 100 training samples in read mode
data_file = open("mnist_train_100.csv", "r")
# Read all of the lines from the file into memory
data_list = data_file.readlines()
# Close the file (we are done with it)
data_file.close()
# Take the first line (data_list index 0, the first sample), and split it up based on the commas
# all_values now contains a list of [label, pixel 1, pixel 2, pixel 3, ... ,pixel 784]
all_values = data_list[0].split(",")
# Take the long list of pixels (but not the label), and reshape them to a 2D array of pixels
image_array = numpy.asarray(all_values[1:], dtype=numpy.float64).reshape((28, 28))
# Plot this 2D array as an image, use the grey colour map and don’t interpolate
plt.imshow(image_array)
