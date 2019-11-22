'''
Name: Part 1 for the SKLearn Assignment for FE
Intro: This file should use K-means and the Iris or Wine data set to create a graph that visually displays how the total
 squared distance decreases as the number of clusters increases.
Author: William Long
Date : 11/22/2019
'''

from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def get_inertia(data, n):
    '''
    This should find the inertia (total squared distance inside each cluster) for a K-means clustering of n clusters on
    the data.
    :param data: 2D array
    :param n: int
    :return: float
    '''
    kmeans = KMeans(n_clusters=n).fit(data)
    inert = kmeans.inertia_
    return inert

if __name__ == "__main__":
    iris = load_iris()
    data = iris.data
    inert = [get_inertia(data, i) for i in range(1,6)]

    fig, ax = plt.subplots()
    ax.plot(range(1,6), inert)
    plt.show()

