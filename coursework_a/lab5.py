from PCA import pca, train_data, train_labels, test_data, test_labels
from sklearn.preprocessing import MinMaxScaler
# k Nearest Neighbour
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

def run(k=5, p=2):
    # print("running")
    # Extract the principal components from the training data
    train_ext = pca.fit_transform(train_data)
    # Transform the test data using the same components
    test_ext = pca.transform(test_data)
    # Normalise the data sets
    min_max_scaler = MinMaxScaler()
    train_norm = min_max_scaler.fit_transform(train_ext)
    test_norm = min_max_scaler.fit_transform(test_ext)
    # Create a KNN classification system with k = 5
    # Uses the p2 (Euclidean) norm
    # print("fitting")
    knn = KNeighborsClassifier(n_neighbors=k, p=p)
    knn.fit(train_norm, train_labels)
    # print("predicting")
    # Feed the test data in the classifier to get the predictions
    pred = knn.predict(test_norm)
    # Check how many were correct
    scorecard = []
    for i, sample in enumerate(test_data):
        # if i % 1000 == 0:
            #print(f"testing {i} / {len(test_data)}")

        # Check if the KNN classification was correct
        if round(pred[i]) == test_labels[i]:
            scorecard.append(1)
        else:
            scorecard.append(0)

    # Calculate the performance score, the fraction of correct answers
    scorecard_array = np.asarray(scorecard)
    print("Performance = ", (scorecard_array.sum() / scorecard_array.size) * 100, "%", f"k={k}, p={p}")
    
    return (scorecard_array.sum() / scorecard_array.size)


if __name__ == "__main__":
    
    scorecard = {}

    # for i in range(10):
    #     run(k=5, p=1 + i/10)
    
    run(k=5, p=1.2)

    # runs = 100

    # for p in range(1, 25):
    #     _p = 1.5 + p/100
    #     tally = []

    #     for i in range(runs):
    #         tally.append(run(k=6, p=_p))

    #     scorecard[_p] = sum(tally) / runs

    # # scorecard = sorted(scorecard, key=lambda x: scorecard[x])

    # for k, v in scorecard.items():
    #     print(f"p:{round(k, 3)}, avg: {round(v, 4)}")

# After testing PCA with 100 components, the point of inflection is about 39 principal components. 
# If we set the threshold of variance% we want to keep at 90%, then it is 39 pcs. 
#
# **EDIT after testing a bit manually, it takes 39 to get ~90% variance explained with the sum. Not sure 
# what is wrong with the methodology I used to get the value for the principle component number, but I trust
# numpy.sum more. 
# 
# For effective classification, minimum 8-10, most likely one for each possible outcome. 
# I get 70% with MNIST 100/10 and k=4, p=2
# 70-80% with k=4, p=1 and k=3, p=1

# After testing thoroughly at a PCA of 39, my best values are k=6, p=1.58 or 1.6.
# This nets an average of about 0.77-79 (77-79% accuracy.) (now p=1.66-1.67)

# At a PCA of 10, k=10, p=1.65 gives about 75%.



# For the real dataset, it is pretty decent at k=5, p=2. (85~%?)
# k=5, p=1.2, %
