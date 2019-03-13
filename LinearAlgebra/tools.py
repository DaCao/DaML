import numpy as np
from functools import reduce




def diag(M):
    '''
    Return the diagonal of n by n matrix as a list
    :param M: n by n 2d list/array
    :return: 1d array of length n
    '''
    return [M[i][i] for i in range(len(M))]



def transpose(m):
    '''
    :param m:
    :return: transpose of m
    '''
    m_transpose = [[x for x in row] for row in list(zip(*m))]
    return m_transpose


def is_identity(M):
    """
    Return True if M is identitu matrix
    :param M:
    :return: Boolean
    """
    # must be square matrix
    if len(M)!=len(M[0]):
        return False

    # diagonal must all be 1
    for i in range(len(M)):
        if float(M[i][i]) != 0.0:
            return False

    # other positions must all be 0
    return set([float(sum(row)) for row in M]) == set([1.0])



def is_singular(M):
    '''
    A matrix is singular iff its determinant is 0
    :param M:
    :return:
    '''
    return determinant(M) == 0


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




def QR_decomposition(M):
    '''
    https://en.wikipedia.org/wiki/QR_decomposition
    :param M:
    :return:
    '''

    pass


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

    :param M: matrix
    :return: determinant of M
    '''

    if is_identity(M):
        return 1
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




def svd(M):
    '''
    Singular Value Decomposition
    :param M:
    :return:
    '''






    return



def Rayleigh_quotient(A, v):
    '''
    Computes eigen-value from given matrix A and eigen-vector v by Rayleigh quotient
    :param A:
    :param v:
    :return:
    '''

    pass







def power_iter(A, max_iter = 1000, thres=0.001):
    '''
    Power Iteration is a method for approximating the dominant Eigenvalues and Eigenvectors of a matrix
    http://mlwiki.org/index.php/Power_Iteration
    :param A: 2d numpy array
    :return: eigenvalue, eigenvector
    '''

    def _eigenvalue(M, v):
        """ Computes eigen-value from given matrix A and unit vector eigenvector v """
        Mv = M.dot(v)
        return v.dot(Mv)

    n, m = A.shape

    vec = np.ones(n)
    vec /= np.linalg.norm(vec)
    val = _eigenvalue(A, vec)

    iter = 0
    while iter < max_iter:
        Avec = A.dot(vec)
        new_vec = Avec / np.linalg.norm(Avec)
        new_val = _eigenvalue(A, vec)

        if np.abs(new_val - val) < thres:
            break

        val, vec = new_val, new_vec
        iter += 1

    return new_val, new_vec



def eigen_decomp_DivideConquer(M):
    pass



def eigen_decomp_QR(M):
    '''
    https://en.wikipedia.org/wiki/Eigenvalue_algorithm
    https://en.wikipedia.org/wiki/List_of_numerical_analysis_topics#Eigenvalue_algorithms
    :param M:
    :return:
    '''




    return


def inv_by_lu_decomp(M):
    '''

    :param M:
    :return:
    '''
    pass


def inv_by_svd(M):
    '''

    :param M:
    :return:
    '''
    pass

def inv_by_gaus(M):
    '''
    Compute the inverse of a matrix

    - Inverse of permutation matrix is itself
    - A singular matrix is a square matrix that does not have a matrix inverse.
    - If any

    Implement 3 methods:  https://www.zhihu.com/question/19584577/answer/359381546
    1. Gaussian Elimination method: https://www.scratchapixel.com/lessons/mathematics-physics-for-computer-graphics/matrix-inverse
    2. todo: LU decomposition method (#todo: use parallel computation!!)
    3. svd method

    :param m: 2d numpy array of dimension n by n
    :return: inverse of m
    '''
    if is_permutation(M):
        return M
    if is_singular(M):
        print('Input matrix is singular and has no inverse! ')
        return False

    n = len(M)
    # append an identity matrix to M such that new dimensions are n by 2n
    id = [[float(i == j) for j in range(n)] for i in range(n)]
    for i in range(n):
        M[i] += id[i]

    # forward part
    for j in range(n):

        # exchange rows when diagonal element is 0
        if M[j][j] == 0:
            if j == n-1:
                print('Singular matrix does not have inverse! ')
                return False
            row_to_swap, max_abs = j, 0
            for i in range(j+1, n):
                if M[i][j] > max_abs:
                    row_to_swap, max_abs = i, abs(M[i][j])

            if row_to_swap == j:
                print('Singular matrix does not have inverse! ')
                return False

            M[j], M[row_to_swap] = M[row_to_swap], M[j]

        pivot = M[j][j]
        for i in range(j, 2*n): # scale row such that the diagonal element is 1.0
            M[j][i] /= pivot
        for i in range(j+1, n): # update rows below the j-th row
            x = M[i][j] # the multiplier that make pivot row ready for subtraction
            for k in range(j, 2*n):
                M[i][k] -= x * M[j][k]

    # backward part
    for j in range(n):
        for i in range(0, j):
            x = M[i][j]
            # do elimination for the i-th row such that its j-th element is 0
            for k in range(j, 2*n):
                M[i][k] -= M[j][k] * x

    # return the second half columns from the n by 2*n matrix M
    return [row[n:] for row in M]







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

