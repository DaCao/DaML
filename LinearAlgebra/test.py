from LinearAlgebra.tools import mat_multiply



if __name__ == '__main__':




    m = [[1,2,6,2],
         [4,4,6,3],
         [6,5,2,6],
         [9,3,9,9]]



    # m = [[1,6,3],
    #      [4,5,6],
    #      [7,2,0]]

    for row in m:
        print(row)
    print('  ')
    pm = pivotize(m)

    for row in pm:
        print(row)

