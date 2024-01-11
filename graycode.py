#return gray code decimal array

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





