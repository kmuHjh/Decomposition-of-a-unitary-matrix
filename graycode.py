import numpy as np
import decomposition_2qubit_complex as d2


def recursive(arr):
    new_arr = []
    size = np.size(arr)
    for i in range(size):
        temp = str(arr[i])
        temp = '0' + temp
        new_arr.append(temp)
    for i in range(size-1,-1,-1):
        temp = str(arr[i])
        temp = '1' + temp
        new_arr.append(temp)

    return new_arr        

#return gray code decimal array
def Graycode(n):
    arr = [0, 1]
    if n >= 2:
        for i in range(n-1):
            arr = recursive(arr)
    for i in range(np.size(arr)):
        temp = int(arr[i],2)
        arr[i] = temp

    return arr
'''
class Graycode:
    n = 0
    arr = [0,1]

    def __init__(self, n):
        self.n = n
        if n >= 2:
            for i in range(n-1):
                self.arr = recursive(self.arr)
        for i in range(np.size(self.arr)):
            temp = int(self.arr[i],2)
            self.arr[i] = temp+1
'''

matrix = (1/2)*np.array([
[1,1,1,1],
[1,-1,1,-1],
[1,1,-1,-1],
[1,-1,-1,1]])
matrix_2 = np.array([
    [0,0,1,0],
    [0,-np.sqrt(3)/2,0,-1/2],
    [np.sqrt(3)/2, -1/4,0, np.sqrt(3)/4],
    [1/2, np.sqrt(3)/4,0,-3/4]])
'''
matrix = np.array([
[0,0,0,0,-1,0,0,0],
[0,0,0,0,0,-1,0,0],
[0,0,0,0,0,0,0,-1],
[0,0,0,0,0,0,-1,0],
[1,0,0,0,0,0,0,0],
[0,1,0,0,0,0,0,0],
[0,0,0,1,0,0,0,0],
[0,0,1,0,0,0,0,0]])
'''
'''
matrix = (1/2)*np.array([
[1,1,1,1],
[1,1j,-1,-1j],
[1,-1,1,-1],
[1,-1j,-1,1j]])
'''
#make (col, pmatrix_col) = 0, use (row, col) p-matrix
'''
def get_pmatrix(matrix, pmatrix_col, row, col): 
    base = np.identity((np.size(matrix[0])))
    x = matrix[row][pmatrix_col]
    y = matrix[col][pmatrix_col]
    r = np.sqrt(x**2 + y**2)
    
    a = x/r
    b = y/r
    base[row][row] = a
    base[row][col] = b
    base[col][row] = -np.conjugate(b)
    base[col][col] = np.conjugate(a)

    return base
'''

def get_pmatrix(matrix, pmatrix_col, row, col): 
    base = np.identity((np.size(matrix[0])), dtype = complex)
    x = matrix[row][pmatrix_col]
    y = matrix[col][pmatrix_col]
    x_1 = np.real(x)
    x_2 = np.imag(x)
    y_1 = np.real(y)
    y_2 = np.imag(y)
    r = np.sqrt(np.abs(x)**2 + np.abs(y)**2)

    A = np.array([
        [x_1, -x_2, y_1, -y_2],
        [x_2, x_1, y_2, y_1],
        [y_1, y_2, -x_1, -x_2],
        [y_2, -y_1, -x_2, x_1]
    ], dtype = complex)
    B = np.transpose(np.array([r,0,0,0]))
    inv_A = np.round(np.linalg.inv(A), decimals=10)
    xy = np.dot(inv_A, B)
    xy = np.round(xy, decimals=10)
    a = xy[0] + 1j*xy[1]
    b = xy[2] + 1j*xy[3]
    
    base[row][row] = a
    base[row][col] = b
    base[col][row] = -np.conjugate(b)
    base[col][col] = np.conjugate(a)
    
    return base

def get_pmatrix_stack(matrix):
    det = (np.linalg.det(matrix)).conjugate()
    stack_matrix = []
    stack_type = []  # CNOT type(qubit number - n, row - graycode[j-1], col - graycode[j])
    n = 0
    cnt = 0
    size = np.size(matrix[0])
    while True:
        size = size/2
        if size<1:
            break
        else:
            n += 1
    graycode = Graycode(n)
    graycode_2 = Graycode(n)
    new = matrix.copy()
    for i in range(np.size(graycode)-1):
        for j in range(np.size(graycode)-1,0,-1):
            x = new[graycode[j-1]][graycode_2[i]]
            y = new[graycode[j]][graycode_2[i]]
            if (np.abs((new[graycode[j]][graycode_2[i]]))==0):
                cnt += 1
                continue
            p_matrix = get_pmatrix(new, graycode_2[i], graycode[j-1], graycode[j])
            stack_type.append([n, graycode[j-1], graycode[j]])
            #print(p_matrix)
            #print(p_matrix.conjugate().transpose())
            #print(np.linalg.det(p_matrix))
            if cnt == 5:
                temp = d2.distinguish(p_matrix)
                if temp == 1:
                    p_matrix[1][0] = det * p_matrix[1][0]
                    p_matrix[1][1] = det * p_matrix[1][1]
                elif temp == 2:
                    p_matrix[2][2] = det * p_matrix[2][2]
                    p_matrix[2][3] = det * p_matrix[2][3]
                elif temp == 4:
                    p_matrix[3][1] = det * p_matrix[3][1]
                    p_matrix[3][3] = det * p_matrix[3][3]
            stack_matrix.append(p_matrix.conjugate().transpose())
            new = np.dot(p_matrix, new) 
            new = np.round(new, decimals=10)
            cnt += 1
        graycode = np.delete(graycode,0)
    
    new = np.round(new, decimals=10)
    return stack_matrix, stack_type

'''
temp, temp_2 = get_pmatrix_stack(matrix)
size = int(np.size(temp)/(np.size(matrix[0]))**2)
'''
'''
for i in range(size):
    print(i)
    print(temp_2[i])
    print(temp[i])
'''



