#! /usr/bin/env python3
########################################
# CS63: Artificial Intelligence, Lab 7
# Spring 2018, Swarthmore College
########################################
# full name(s): Cindy Li, Grace Zhang
# username(s): cli2, yzhang1
########################################

import numpy as np
import sklearn.linear_model
import matplotlib.pyplot as plt

from SupervisedLearning import parse_args

np.seterr(all="raise")

class LinearRegression:
    def gradient(self, X_train, Y_train, weights):
        new_weights = []

        for j in range(len(weights)):
            sums = 0
            for i in range(len(X_train)):
                sums += X_train[i][j] * (Y_train[i] - (np.dot(weights, X_train[i])))
            sums *= (-2/len(X_train))
            new_weights.append(sums)
        return np.array(new_weights)

    def fit(self, X_train, Y_train, iters=10000, step_size=0.1):
        """Uses gradient descent to find optimal weights.

        steps:
        1) A 1 is appended to each data point in X_train to serve as a bias.
        2) Weights are initialzed to an array of zeros equal to the number of
            dimensions (including bias).
        3) Each iteration, weights are updated by step_size * (-gradient).

        Hint: a helper function to compute gradients is a good idea.

        For turkish_stock_exchange: good parameters are step size: 0.8, iterations: 50000
        For linear_data: good-ish parameters are
        """
        #append 1 to beginning of all data points as bias
        np.insert(X_train, 0, 1)
        self.weights = np.zeros(len(X_train[0]))
        for i in range(iters):
            self.weights = self.weights - (step_size * (self.gradient(X_train, Y_train, self.weights)))

        return self.weights
    def predict(self, X_test):
        """Uses learned weights to predict the y value at each test point.

        steps:
        1) A bias must be appended to match the weights.
        2) For each point, the prediction is the dot product of the data
            vector and the weight vector."""
        np.insert(X_test, 0, 1)
        self.predictions = np.dot(X_test, self.weights)

        return self.predictions
        
    def error(self, X_test, Y_test):
        """Finds the sum of squared errors on test points.

        Squared error is the sum over all test points of the difference
        between the output of predict and the desired output (Y_test), squared.
        """
        sum_error = 0
        predictions = self.predict(X_test)
        for i in range(len(X_test)):
            sum_error += (predictions[i] - Y_test[i])**2

        return sum_error

def plot_predictions(X, Y, *models):
    """Plots predicted values against actual values for each model."""
    for model in models:
        plt.scatter(Y, model.predict(X), edgecolors=(0,0,0))
    plt.plot([Y.min(), Y.max()], [Y.min(), Y.max()], 'k--', lw=4)
    plt.xlabel('Measured')
    plt.ylabel('Predicted')
    plt.show()

def plot_1D(X, Y, *models, dim=0):
    """Makes a scatter plot of X[dim] vs Y, overlayed with each model's line."""
    plt.clf()
    plt.scatter(X[:,dim], Y, edgecolors=(0,0,0))
    x_min = X[:,dim].min()
    x_max = X[:,dim].max()
    regr_x = np.arange(x_min, x_max, (x_max-x_min)/1000)
    regr_x = regr_x.reshape(regr_x.shape[0],1)
    regr_x = np.append(regr_x, np.zeros([regr_x.shape[0], X.shape[1]-1]), 1)
    for model in models:
        plt.plot(regr_x[:,0], model.predict(regr_x), label=str(model))
    plt.show()

def main():
    X_train, X_test, Y_train, Y_test, args = parse_args()
    my_LR = LinearRegression()
    my_LR.fit(X_train, Y_train, args.i, args.s)
    skl_LR = sklearn.linear_model.LinearRegression()
    skl_LR.fit(X_train, Y_train)
    plot_predictions(X_test, Y_test, my_LR, skl_LR)
    plot_1D(X_test, Y_test, my_LR, skl_LR)

if __name__ == "__main__":
    main()
