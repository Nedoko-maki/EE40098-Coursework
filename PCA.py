# Numpy for useful maths
import numpy
# Sklearn contains some useful CI tools
# PCA
from sklearn.decomposition import PCA
# Matplotlib for plotting
import matplotlib as mpl
import matplotlib.pyplot as plt

def plot(components):
    # Plot the explained variance

    cum_list = []
    cum = 0

    for i in pca.explained_variance_ratio_:
        cum += i
        cum_list.append(cum)    

    # plt.axhline(y=max(0.1 * (pca.explained_variance_ratio_)), color='r', linestyle='-')
    bar = plt.bar([x+1 for x in range(components)], pca.explained_variance_ratio_)
    plt.bar_label(bar, labels=[round(x, 2) for x in cum_list],
             padding=8, color='b', fontsize=6)
    plt.xticks([x+1 for x in range(components)])
    plt.title("PCA Variance Explained by Extracted Components - MNIST full")
    plt.ylabel("Variance")
    plt.xlabel("Principal Components")
    plt.show()

# Load the train and test MNIST data
train = numpy.loadtxt("datasets/mnist_train.csv", delimiter=",")
test = numpy.loadtxt("datasets/mnist_test.csv", delimiter=",")
# Separate labels from training data
train_data = train[:, 1:]
train_labels = train[:, 0]
test_data = test[:, 1:]
test_labels = test[:, 0]
# Select number of components to extract

components = 84
pca = PCA(n_components = components)
# Fit to the training data
pca.fit(train_data)
# Determine amount of variance explained by components
print("Total Variance Explained: ", numpy.sum(pca.explained_variance_ratio_))

# plot(components)

# To get all the variance, it needs 98 comps to give 99.99% of the variance
# to get 95% of the variance, it requires 54 components 

# 90% variance is more than sufficient (38 comps), 80% probably is still good (24 comps) 