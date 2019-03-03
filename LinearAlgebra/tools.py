import numpy as np



def transpose(m):
    '''

    :param m:
    :return:
    '''
    m_transpose = [[x for x in row] for row in zip(*m)]
    return m_transpose



def inv(m):
    '''
    Compute the inverse of a matrix
    :param m: 2d numpy array of dimension n by n
    :return: inverse of m
    '''

    pass



def mat_multiply(A, B):
    '''
    computes matrix multiplication a * b
    :param a: 2d numpy array
    :param b: 2d numpy array
    :return: 2d numpy array
    '''
    B_t = zip(*B)
    return [[sum(a*b for a, b in zip(a_row, b_row))  for b_row in B_t] for a_row in A]


def pivotize(m):
    '''
    Creates the pivoting matrix for m
    :param m:
    :return:
    '''


    pass


def LUdecomposition(m):
    '''
    computes the Lower-Upper decomposition of matrix m
    :param m:
    :return:
    '''


def determinant(m):
    '''

    :param m:
    :return:
    '''

    pass
