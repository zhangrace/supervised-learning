#! /usr/bin/env python3
########################################
# CS63: Artificial Intelligence, Lab 7
# Spring 2018, Swarthmore College
########################################

import csv
import numpy as np
from argparse import ArgumentParser

def parse_args():
    "Reads command line and and data file; returns X_train, X_test, Y_train, Y_test."
    parser = ArgumentParser()
    parser.add_argument("data_file", type=str, help="csv file with learning data")
    parser.add_argument("-t", default=0.2, help="fraction of the data to use for testing")
    parser.add_argument("-k", type=int, default=3, help="number of neighbors for KNN.")
    parser.add_argument("-s", type=float, default=0.01, help="step size for linear regression.")
    parser.add_argument("-i", type=int, default=10000, help="iterations for linear regression.")
    args = parser.parse_args()
    X,Y = read_input(args.data_file)
    X_train, X_test, Y_train, Y_test = test_train_split(X, Y, args.t)
    return X_train, X_test, Y_train, Y_test, args

def read_input(filename):
    """Reads a CSV file into X and Y arrays.

    The first line of the file should be a single integer indicating which
    column index is the target (y output).
    """
    with open(filename) as f:
        r = csv.reader(f)
        data = [l for l in r]
    target_index = int(data[0][0])
    X = np.array([[float(v) for v in l[0:target_index] + l[target_index+1:-1]] for l in data[1:]])
    Y = np.array([float(l[target_index]) for l in data[1:]])
    return X,Y

def test_train_split(X_data, Y_data, test_fraction=0.2):
    """Splits the data int a test set and a training set.

    Returns four arrays: X_train, X_test, Y_train, Y_test.
    """
    split_index = int(X_data.shape[0] * test_fraction)
    indices = np.arange(X_data.shape[0])
    np.random.shuffle(indices)
    X_test = X_data[indices[:split_index]]
    X_train = X_data[indices[split_index:]]
    Y_test = Y_data[indices[:split_index]]
    Y_train = Y_data[indices[split_index:]]
    return X_train, X_test, Y_train, Y_test
