import numpy as np
import matplotlib.pyplot as plt

data= np.genfromtxt('\train50k_16bit - copy.csv', delimiter=',', dtype='f8')
data = data[2:]
Y = [x[28] for x in data]
plt.hist(Y)
plt.title("Frequency Distribution")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()
