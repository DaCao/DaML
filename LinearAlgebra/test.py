
def pivotize(m):
    """Creates the pivoting matrix for m."""
    n = len(m)
    ID = [[float(i == j) for i in range(n)] for j in range(n)]
    for j in range(n):
        row = max(range(j, n), key=lambda i: abs(m[i][j]))
        print(j, row)
        if j != row:
            ID[j], ID[row] = ID[row], ID[j]
    return ID




if __name__ == '__main__':

    m = [[5,2,6,2],
         [6,7,6,3],
         [2,5,2,6],
         [8,1,1,1]]

    for row in m:
        print(row)
    print('  ')
    pm = pivotize(m)

    for row in pm:
        print(row)

    n = len(m)
    ID = [[float(i == j) for i in range(n)] for j in range(n)]
    print('---------')
    for row in ID:
        print(row)

    TB = zip(*m)
    print(TB)