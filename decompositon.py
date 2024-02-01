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

#return single qubit gate parameter(alpha, beta, gamma), unit : degree
def solve_unitary(matrix):
    x = np.real(matrix[0][0])
    y = np.imag(matrix[0][0])
    a = np.real(matrix[0][1])
    b = np.imag(matrix[0][1])
    
    r = np.sqrt(x**2 + y**2)
    theta = np.arccos(r)
    if ( x==0 ):
        lam = np.pi/2
    else:
        lam = np.arctan(y/x)
    if ( a==0 ):
        mu = np.pi/2
    else:
        mu = np.arctan(b/a)
    
    alpha = lam + mu
    beta = 2*theta
    gamma = lam - mu

    return np.degrees(alpha), np.degrees(beta), np.degrees(gamma)

#distinguish return 1 case
def decomposition_0V(matrix):
    u = np.array([[matrix[0][0],matrix[0][1]],[matrix[1][0],matrix[1][1]]])
    return solve_unitary(u)

#distinguish return 2 case
def decomposition_1V(matrix):
    u = np.array([[matrix[2][2],matrix[2][3]],[matrix[3][2],matrix[3][3]]])
    return solve_unitary(u)

#distinguish return 3 case
def decomposition_V0(matrix):
    u = np.array([[matrix[0][0],matrix[0][2]],[matrix[2][0],matrix[2][2]]])
    return solve_unitary(u)

#distinguish return 4 case
def decomposition_V1(matrix):
    u = np.array([[matrix[1][1],matrix[1][3]],[matrix[3][1],matrix[3][3]]])
    return solve_unitary(u)


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
