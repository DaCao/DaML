import numpy as np
from sklearn.preprocessing import binarize, LabelBinarizer, label_binarize




x = [[1,2,3,4,5,6],
     [2, 4, 6, 8, 10, 12],
     [1, 2, 3, 4, 5, 6],
     [2, 4, 6, 8, 10, 12],
     ]

y = [[2],[2],[2],[2],[2],[200]]
y = [2,2,2,2,2,200]


x = np.asarray(x)
y = np.asarray(y)

print(np.dot(x, y))


print(x*y)

exit()

print(label_binarize(y, classes))

exit()

x = np.random.rand(10, 6)

x_mean = np.mean(x, axis=0)

print(x)
print('\n-------------\n')
print(x_mean)
print('\n-------------\n')
print(abs(x-x_mean))
print('\n-------------\n')
print(np.sum(abs(x-x_mean), 1))