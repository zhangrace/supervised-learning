#! /usr/bin/env python3
########################################
# CS63: Artificial Intelligence, Lab 7
# Spring 2018, Swarthmore College
########################################
# full name(s): Cindy Li, Grace Zhang
# username(s): cli2, yzhang1
########################################
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from collections import Counter
from statistics import mode

from SupervisedLearning import parse_args

np.seterr(all="raise")

class kNearestNeighbors:
    def __init__(self, k):
        self.k = k

    def fit(self, x_train, y_train):
        """Stores the training set after normalizing each dimension.

        Very little training is required here. The input data is stored in
        a list, but first each dimension is normalized to have values in the
        range 0-1. The normalization factors need to be stored so that they
        can be applied to the test set later.
        """
        # make a copy so we don't modify x_train directly
        x_train = x_train.copy()
        self.normalizing = []
        self.y_train = y_train

        for i in range(len(x_train[0])):
            # normalize columns (features)
            normal = np.amax(x_train[:,i]) - np.amin(x_train[:,i])
            # if max and min are the same, we don't want to try to divide by 0
            if normal == 0:
                normal = 1
            self.normalizing.append(normal)
            x_train[:,i] /= normal

        self.training_set = np.array(x_train)

    def predict(self, X_test):
        """Returns the plurality class over the k closest training points.

        Nearest neighbors are chosen by Euclidean distance (after normalizing
        the test point).
        Ties are broken by repeatedly removing the most-distant element(s)
        until a strict plurality is achieved.
        """
        X_test = X_test.copy()

        # normalize first
        for i in range(len(X_test[0])):
            X_test[:,i] /= self.normalizing[i]

        # pairs of (distance, label)
        neighbors_distances = []
        plurality_classes = []

        for i in range(len(X_test)):
            k_neighbors = []
            # find euclidean distances between a testing point and every point
            # in our training set using linalg.norm
            neighbors_distances = np.linalg.norm(self.training_set - X_test[i], axis=1)
            zipped_distances = list(zip(neighbors_distances, self.y_train))
            zipped_distances.sort()

            for i in range(self.k):
                k_neighbors.append(zipped_distances[i][1])
            # find most common label among k neighbors
            most_common = Counter(k_neighbors).most_common()[0][0]
            plurality_classes.append(most_common)

        return plurality_classes

    def error(self, X_test, Y_test):
        """Finds the fraction of test points on which the predicted
        label is incorrect."""
        # call predict, compare resulting label with Y_test
        num_incorrect = 0
        predicted_labels = self.predict(X_test)
        for i in range(len(X_test)):
            if predicted_labels[i] != Y_test[i]:
                num_incorrect+=1

        return num_incorrect/len(X_test)


def confusion_matrix(predictions, labels):
    """Counts how many times each label/prediction pair occurred.

    The first row & column of the returned matrix will be the y values that
    occur in either predictions or labels. Other entries show how many times
    the prediction was the row when the label should have been the column.
    """
    all_labels = sorted(set(predictions).union(set(labels)))
    m = np.zeros([len(all_labels), len(all_labels)], int)
    for y1, y2 in zip(predictions, labels):
        i1 = all_labels.index(y1)
        i2 = all_labels.index(y2)
        m[i1,i2] += 1
    m = np.append([all_labels], m, axis=0)
    m = np.append([[0]] + [[l] for l in all_labels], m, axis=1)
    return m

def main():
    X_train, X_test, Y_train, Y_test, args = parse_args()
    my_KNN = kNearestNeighbors(args.k)
    my_KNN.fit(X_train, Y_train)
    skl_KNN = KNeighborsClassifier(n_neighbors=args.k, algorithm="brute")
    skl_norm = MinMaxScaler()
    skl_KNN.fit(skl_norm.fit_transform(X_train), Y_train)

    my_predictions = my_KNN.predict(X_test)
    print("confusion matrix for my predictions vs. true labels:")
    print(confusion_matrix(my_predictions, Y_test))
    print("my error rate:", my_KNN.error(X_test, Y_test))

    skl_predictions = skl_KNN.predict(skl_norm.transform(X_test))
    print("confusion matrix for my predictions vs. sklearn predictions:")
    print(confusion_matrix(my_predictions, skl_predictions))
    print("sklearn error rate:", 1 - skl_KNN.score(X_test, Y_test))



if __name__ == "__main__":
    main()
