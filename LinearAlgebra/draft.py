import numpy as np


x = np.random.rand(10, 6)

x_mean = np.mean(x, axis=0)

print(x)
print('\n-------------\n')
print(x_mean)
print('\n-------------\n')
print(abs(x-x_mean))
print('\n-------------\n')
print(np.sum(abs(x-x_mean), 1))