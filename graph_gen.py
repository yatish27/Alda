__author__ = 'Shubham'

import numpy as np
import matplotlib.pyplot as plt
from pylab import *

data= np.genfromtxt('C:\Users\Shubham\Downloads\Alda-master\Alda-master\erain50k_16bit - copy.csv', delimiter=',', dtype='f8')
data = data[2:]

for m in range(1, 2):
    Y = [x[m] for x in data]
    plt.hist(Y)
    if m in range(1,14):
        k = 'I%s' % m
        plt.title(k)
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        savefig(k, format='png')

    elif m in range(15,40):
        k = 'C%s' % m
        plt.title(k)
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        savefig(k, format='png')

