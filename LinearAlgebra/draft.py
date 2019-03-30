import numpy as np
from sklearn.preprocessing import binarize, LabelBinarizer, label_binarize


y = [1,6,1,6]
classes = [1,2,4,6,8]
class_counts = [0,0,0,0,0,0,0]

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