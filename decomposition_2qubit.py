import numpy as np
import graycode as gc
import math
from qiskit import QuantumCircuit, execute, Aer
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
    theta = math.atan2(matrix[0][1],matrix[0][0])
    
    alpha = 0
    beta = 2*theta
    gamma = 0

    return (alpha), (beta), (gamma)

def twoqubit_to_single(QuantumCircuit, matrix):
    qc = QuantumCircuit
    stack_matrix, stack_type = gc.get_pmatrix_stack(matrix)
    size = int(np.size(stack_matrix)/(np.size(matrix[0]))**2)
    for i in reversed(range(size)):
        type = distinguish(stack_matrix[i])
        if type == 1:  #0V
            decomposition_0V(qc, stack_matrix[i], 0, 1)

        elif type == 2: #1V
            decomposition_1V(qc, stack_matrix[i], 0, 1)
    
        elif type == 3: #V0
            decomposition_V0(qc, stack_matrix[i], 1, 0)
    
        elif type == 4: #V1
            decomposition_V1(qc, stack_matrix[i], 1, 0)
        
    return 0

#distinguish return 1 case
def decomposition_0V(QuantumCircuit, matrix, control_bit, target_bit):
    qc = QuantumCircuit
    u = np.array([[matrix[0][0],matrix[0][1]],[matrix[1][0],matrix[1][1]]])
    alpha, beta, gamma = solve_unitary(u)
    qc.rz((alpha-gamma)/2, control_bit)
    qc.x(target_bit)
    qc.cx(target_bit, control_bit)
    qc.x(target_bit)
    qc.ry(beta/2, control_bit)
    qc.rz((alpha+gamma)/2, control_bit)
    qc.x(target_bit)
    qc.cx(target_bit, control_bit)
    qc.x(target_bit)
    qc.rz(-alpha, control_bit)
    qc.ry(-beta/2, control_bit)

#distinguish return 2 case
def decomposition_1V(QuantumCircuit, matrix, control_bit, target_bit):
    qc = QuantumCircuit
    u = np.array([[matrix[2][2],matrix[2][3]],[matrix[3][2],matrix[3][3]]])
    alpha, beta, gamma = solve_unitary(u)
    qc.rz((alpha-gamma)/2, control_bit)
    qc.cx(target_bit, control_bit)
    qc.ry(beta/2, control_bit)
    qc.rz((alpha+gamma)/2, control_bit)
    qc.cx(target_bit, control_bit)
    qc.rz(-alpha, control_bit)
    qc.ry(-beta/2, control_bit)
    

#distinguish return 3 case
def decomposition_V0(QuantumCircuit, matrix, control_bit, target_bit):
    qc = QuantumCircuit
    u = np.array([[matrix[0][0],matrix[0][2]],[matrix[2][0],matrix[2][2]]])
    alpha, beta, gamma = solve_unitary(u)
    qc.rz((alpha-gamma)/2, control_bit)
    qc.x(target_bit)
    qc.cx(target_bit, control_bit)
    qc.x(target_bit)
    qc.ry(beta/2, control_bit)
    qc.rz((alpha+gamma)/2, control_bit)
    qc.x(target_bit)
    qc.cx(target_bit, control_bit)
    qc.x(target_bit)
    qc.rz(-alpha, control_bit)
    qc.ry(-beta/2, control_bit)

#distinguish return 4 case
def decomposition_V1(QuantumCircuit, matrix, control_bit, target_bit):
    qc = QuantumCircuit
    u = np.array([[matrix[1][1],matrix[1][3]],[matrix[3][1],matrix[3][3]]])
    alpha, beta, gamma = solve_unitary(u)
    qc.rz((alpha-gamma)/2, control_bit)
    qc.cx(target_bit, control_bit)
    qc.ry(beta/2, control_bit)
    qc.rz((alpha+gamma)/2, control_bit)
    qc.cx(target_bit, control_bit)
    qc.rz(-alpha, control_bit)
    qc.ry(-beta/2, control_bit)

#test set
matrix = (1/2)*np.array([
[1,1,1,1],
[1,-1,1,-1],
[1,1,-1,-1],
[1,-1,-1,1]])
#twoqubit_to_single(matrix)
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
