import numpy as np
from scipy.stats import norm, cauchy
import matplotlib.pyplot as plt
import math

if __name__ == '__main__':
    maxval = math.radians(50)
    minval = -maxval
    size = 21
    step = (maxval - minval)/size
    x = np.linspace(minval, maxval-step, size)
    print(x)
    large_x = np.linspace(minval, maxval-step, 1000000)
    sigma = 0.05
    rv = cauchy(0, sigma)   #gennorm, laplace

    large_pdf = rv.pdf(large_x)
    large_pdf /= large_pdf.sum()
    plt.plot(large_x, large_pdf)
    plt.show()

    integral = 0
    division = 1. / size
    intervals = np.zeros(x.shape[0] - 1)
    i = 0
    for val, proba in zip(large_x, large_pdf):
        integral += proba
        if integral >= division:
            intervals[i] = val
            integral = 0
            i += 1
    plt.plot(np.degrees(intervals), np.zeros(intervals.shape), linestyle='--', marker='o', color='b')
    plt.show()
    data = rv.rvs(size=100000)
    print(intervals)
    bins = np.digitize(data, intervals)
    distribution = np.zeros(intervals.shape[0] + 1)
    for bin in bins:
        distribution[bin] += 1
    print(distribution)