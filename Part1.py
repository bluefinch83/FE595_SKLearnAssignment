'''
Name: Part 1 for the SKLearn Assignment for FE
Intro: This file should use the Linear Regression class and the Boston housing data set from SKLearn and determine which
element has the most influence on the price of a house in Boston.
Author: William Long
Date : 11/22/2019
'''

from sklearn.datasets import load_boston
from sklearn import linear_model
import numpy as np


def part1():
    boston = load_boston()
    model = linear_model.LinearRegression()
    model.fit(boston.data, boston.target)
    '''Wait, does this model not come with the t-values for the ols model? What am I suppose to do with these coeffs? 
    Anyways, since this is quantitative data, just looking at the coefs many not be the best idea. We need to look at the 
    mean times the coef to get a good idea of an element's impact. Plus, we can find the range of its impact.'''
    coeffs = model.coef_
    impact_avg = []
    impact_ran = []
    data = boston.data
    for i in range(len(data[0])):
        col = data[:,i]
        avg = sum(col)/len(col)
        sd = np.sqrt(np.var(col))
        imp = abs(coeffs[i]*avg)
        imp_sd = abs(coeffs[i]*sd)
        impact_avg.append(imp)
        impact_ran.append(imp_sd)

    return coeffs, impact_avg, impact_ran,  model


if __name__ == "__main__":
    coeffs, impact_avg, impact_ran, model = part1()

