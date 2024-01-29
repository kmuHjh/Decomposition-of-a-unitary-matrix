import numpy as np
from sympy import Symbol, symbols, solve, exp, cos, sin, I

#return control type 0V - 1, 1V - 2, V0 - 3, V1 - 4
def distinguish(matrix):
    if(matrix[2][2] == 1 and matrix[3][3] == 1):
        return 1
    if(matrix[0][0] == 1 and matrix[1][1] == 1):
        return 2
    if(matrix[1][1] == 1 and matrix[3][3] == 1):
        return 3
    if(matrix[0][0] == 1 and matrix[2][2] == 1):
        return 4


def solve_unitary(matrix):
    t = Symbol("t")
    result = solve([cos(t) - matrix[0][0]])
    return result[0][t]

#distinguish return 1 case
def decomposition_0V(matrix):
    u = np.array([[matrix[0][0],matrix[0][1]],[matrix[1][0],matrix[1][1]]])
    t = solve_unitary(u)
    return t

#distinguish return 2 case
def decomposition_1V(matrix):
    u = np.array([[matrix[2][2],matrix[2][3]],[matrix[3][2],matrix[3][3]]])
    t = solve_unitary(u)
    return t

#distinguish return 3 case
def decomposition_V0(matrix):
    u = np.array([[matrix[0][0],matrix[0][2]],[matrix[2][0],matrix[2][2]]])
    t = solve_unitary(u)
    return t

#distinguish return 4 case
def decomposition_V1(matrix):
    u = np.array([[matrix[1][1],matrix[1][3]],[matrix[3][1],matrix[3][3]]])
    t = solve_unitary(u)
    return t


m_0V = np.array([[1/np.sqrt(2),1/np.sqrt(2),0,0],
          [-1/np.sqrt(2),1/np.sqrt(2),0,0],
          [0,0,1,0],
          [0,0,0,1]])

m_1V = np.array([[1,0,0,0],
          [0,1,0,0],
          [0,0,1/np.sqrt(2),1/np.sqrt(2)],
          [0,0,-1/np.sqrt(2),1/np.sqrt(2)]])
m_V0 = np.array([[1/np.sqrt(2),0,1/np.sqrt(2),0],
          [0,1,0,0],
          [-1/np.sqrt(2),0,1/np.sqrt(2),0],
          [0,0,0,1]])
m_V1 = np.array([[1,0,0,0],
          [0,1/np.sqrt(2),0,1/np.sqrt(2)],
          [0,0,1,0],
          [0,-1/np.sqrt(2),0,1/np.sqrt(2)]])

