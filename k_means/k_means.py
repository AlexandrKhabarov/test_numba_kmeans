from numba import njit
import numpy as np
from math import sqrt
import argparse
import time


def kmeans_straight(dots, num_center, num_iter, num_dots, features, init_centroids):
    centroids = init_centroids

    for l in range(num_iter):
        dist = np.array([[sqrt(np.sum((dots[i, :] - centroids[j, :]) ** 2))
                          for j in range(num_center)] for i in range(num_dots)])
        labels = np.array([dist[i, :].argmin() for i in range(num_dots)])

        centroids = np.array([[np.sum(dots[labels == i, j]) / np.sum(labels == i)
                               for j in range(features)] for i in range(num_center)])

    return centroids


@njit
def kmeans_straight(dots, num_center, num_iter, num_dots, features, init_centroids):
    centroids = init_centroids

    for l in range(num_iter):
        dist = np.array([[sqrt(np.sum((dots[i, :] - centroids[j, :]) ** 2))
                          for j in range(num_center)] for i in range(num_dots)])
        labels = np.array([dist[i, :].argmin() for i in range(num_dots)])

        centroids = np.array([[np.sum(dots[labels == i, j]) / np.sum(labels == i)
                               for j in range(features)] for i in range(num_center)])

    return centroids
