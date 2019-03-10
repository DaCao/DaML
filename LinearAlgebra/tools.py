import numpy as np
from functools import reduce


def transpose(m):
    '''

    :param m:
    :return:
    '''
    m_transpose = [[x for x in row] for row in list(zip(*m))]
    return m_transpose


def is_identity(M):
    pass


def is_singular(M):
    pass


def is_permutation(M):
    '''
    Check if matrix M is a permutation matrix

    Inverse of a Permutation matrix is itself
    Transpose of a Permutation matrix is itself

    :param M:
    :return:
    '''

    s = set([i for sublist in M for i in sublist])
    if s != set([0,1]):
        return False

    s = set([sum(args) for args in list(zip(*M))])
    if s == {1}:
        return True
    return False


def inv(M):
    '''
    Compute the inverse of a matrix

    - Inverse of permutation matrix is itself



    :param m: 2d numpy array of dimension n by n
    :return: inverse of m
    '''
    if is_permutation(M):
        return M




def mat_multiply(A, B):
    '''
    computes matrix multiplication a * b
    :param a: 2d numpy array
    :param b: 2d numpy array
    :return: 2d numpy array
    '''
    B_t = list(zip(*B))
    return [[sum(float(a*b) for a, b in zip(a_row, b_row))  for b_row in B_t] for a_row in A]


def pivotize(M):
    '''
    Creates the pivoting matrix for M
    # this is black box magic
    :param M:
    :return:
    '''
    n = len(M)
    ID = [[float(i==j) for j in range(n)] for i in range(n)]
    for j in range(n):
        row = max(range(j, n), key=lambda i:M[i][j])
        if j != row:
            ID[j], ID[row] = ID[row], ID[j]

    return ID



def LUdecomposition(M):
    '''
    Computes the Lower-Upper decomposition of matrix M
    Apply Doolittle algorithm: https://www.geeksforgeeks.org/doolittle-algorithm-lu-decomposition/
    http://mathonline.wikidot.com/the-algorithm-for-doolittle-s-method-for-lu-decompositions
    Doolittle’s method provides an alternative way to factor A into an LU decomposition without going through
    the hassle of Gaussian Elimination.
    :param m:
    :return:
    '''
    n = len(M)
    L = [[0.0]*n for i in range(n)]
    U = [[0.0] * n for i in range(n)]
    P = pivotize(M)
    M2 = mat_multiply(P, M)

    for j in range(n):
        L[j][j] = 1.0
        for i in range(j+1):
            s1 = sum(U[k][j] * L[i][k] for k in range(i))
            U[i][j] = M2[i][j] - s1
        for i in range(j, n):
            s2 = sum(U[k][j] * L[i][k] for k in range(j))
            L[i][j] = (M2[i][j] - s2) / U[j][j]
    return (L, U, P)



def determinant(M):
    '''
    Calculates the determinant of matrix M
    - determinant of an indentity matrix == 1
    - determinant of a singular matrix == 0
    - determinant of a permutation matrix == (-1)^(number of row swaps)

    The LU decomposition computes three matrices such that PA = LU ----> A = inv(P)LU ----> A = PLU
    Thus, det(A) = det(P)det(L)det(U)

    :param m:
    :return:
    '''

    if is_identity(M):
        return 1
    if is_singular(M):
        return 0
    if is_permutation(M):
        return -1 ** sum([1 for i, row in enumerate(M) if row[i] != 1])

    L, U, P = LUdecomposition(M)

    # determinant(P)
    num_row_swaps = len(P) - sum([1 for i, row in enumerate(P) if row[i] == 1])
    det_P = -1**num_row_swaps

    # determinant(L)
    det_L = reduce((lambda x, y: x * y), [L[i][i] for i in range(len(L))])

    # determinant(U)
    det_U = reduce((lambda x, y: x * y), [U[i][i] for i in range(len(U))])

    return det_P * det_L * det_U








if __name__ == '__main__':



    m = [[0.5,0,0,0],
         [0.5,0,0,1],
         [0,1,0,0],
         [0,0,1,0]]

    print(is_permutation(m))
    exit()




    m = [[1,2,6,2],
         [4,4,6,3],
         [6,5,2,6],
         [9,3,9,9]]



    m = [[1,6,3],
         [4,5,6],
         [7,2,0]]

    for row in m:
        print(row)
    print('  ')
    pm = pivotize(m)

    for row in pm:
        print(row)

