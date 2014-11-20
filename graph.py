import numpy as np
import matplotlib.pyplot as plt

data= np.genfromtxt('C:\Users\Shubham\Downloads\eew.train50k.csv', delimiter=',', dtype='f8')
Y = [x[28] for x in data]
plt.hist(Y)
plt.title("Frequency Distribution")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()
