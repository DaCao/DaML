# from LinearAlgebra.tools import mat_multiply
import numpy as np


if __name__ == '__main__':

    x = np.asarray([3,6,2,4,7])
    y = np.asarray([1,2,2,3,3])

    print(np.dot(x.dot(y), y))
    exit()


    x = np.asarray([3,6,2,4,7])
    x = x/np.linalg.norm(x)
    print(x)


    a = 23
    ax = a*x

    print(x.dot(ax))

    exit()



    M = [[1,2,6,2],
         [4,4,6,3],
         [6,5,2,6],
         [9,3,9,9]]


    # M = [[0,0,1,0],
    #      [0,1,0,0],
    #      [1,0,0,0],
    #      [0,0,0,1]]

    print(np.linalg.inv(np.asarray(M)))


    M[0], M[2] = M[2], M[0]
    print('-------')
    print(np.asarray(M))
    print(np.linalg.inv(np.asarray(M)))

    # m = [[1,6,3],
    #      [4,5,6],
    #      [7,2,0]]


