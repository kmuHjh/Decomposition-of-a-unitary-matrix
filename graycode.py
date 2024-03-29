import numpy as np

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
def get_pmatrix(matrix, pmatrix_col, row, col): 
    base = np.identity((np.size(matrix[0])))
    x = matrix[row][pmatrix_col]
    y = matrix[col][pmatrix_col]
    r = np.sqrt(x**2 + y**2)
    
    a = x/r
    b = y/r
    base[row][row] = np.conjugate(a)
    base[row][col] = np.conjugate(b)
    base[col][row] = -b
    base[col][col] = a

    return base

def get_pmatrix_stack(matrix):
    stack_matrix = []
    stack_type = []  # CNOT type(qubit number - n, row - graycode[j-1], col - graycode[j])
    n = 0
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
            if ((new[graycode[j]][graycode_2[i]])==0):
                continue
            p_matrix = get_pmatrix(new, graycode_2[i], graycode[j-1], graycode[j])
            stack_type.append([n, graycode[j-1], graycode[j]])
            stack_matrix.append(p_matrix.conjugate().transpose())
            new = np.dot(p_matrix, new) 
            new = np.round(new, decimals=10)
        graycode = np.delete(graycode,0)
    new = np.round(new, decimals=10)
    return stack_matrix, stack_type


temp, temp_2 = get_pmatrix_stack(matrix)
size = int(np.size(temp)/(np.size(matrix[0]))**2)

'''
for i in range(size):
    print(i)
    print(temp_2[i])
    print(temp[i])
'''



