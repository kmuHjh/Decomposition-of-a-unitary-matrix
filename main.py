%matplotlib inline

import math
import decomposition
import numpy as np
from qiskit import QuantumCircuit, execute, Aer
from qiskit.visualization import plot_histogram
from qiskit.extensions import *
from qiskit.quantum_info import Statevector

#0V
matrix_0V = np.array([[1/np.sqrt(2),1/np.sqrt(2),0,0],
          [-1/np.sqrt(2),1/np.sqrt(2),0,0],
          [0,0,1,0],
          [0,0,0,1]])
#1V
matrix_1V = np.array([[1,0,0,0],
          [0,1,0,0],
          [0,0,1/np.sqrt(2),1/np.sqrt(2)],
          [0,0,-1/np.sqrt(2),1/np.sqrt(2)]])

#V0
matrix_V0 = np.array([[1/np.sqrt(2),0,1/np.sqrt(2),0],
          [0,1,0,0],
          [-1/np.sqrt(2),0,1/np.sqrt(2),0],
           [0,0,0,1]])

#V1
matrix_V1 = np.array([[1,0,0,0],
          [0,1/np.sqrt(2),0,1/np.sqrt(2)],
          [0,0,1,0],
          [0,-1/np.sqrt(2),0,1/np.sqrt(2)]])

matrix = matrix_V0
type = decomposition.distinguish(matrix)

circuit = QuantumCircuit(2,2)
circuit.barrier()

if type == 1:  #0V
    alpha, beta, gamma = decomposition.decomposition_0V(matrix)
    circuit.rz((alpha-gamma)/2,0)
    circuit.x(1)
    circuit.cx(1,0)
    circuit.x(1)
    circuit.ry(beta/2,0)
    circuit.rz((alpha+gamma)/2,0)
    circuit.x(1)
    circuit.cx(1,0)
    circuit.x(1)
    circuit.rz(-alpha,0)
    circuit.ry(-beta/2,0)

elif type == 2: #1V
    alpha, beta, gamma = decomposition.decomposition_1V(matrix)
    circuit.rz((alpha-gamma)/2,0)
    circuit.cx(1,0)
    circuit.ry(beta/2,0)
    circuit.rz((alpha+gamma)/2,0)
    circuit.cx(1,0)
    circuit.rz(-alpha,0)
    circuit.ry(-beta/2,0)
    
elif type == 3: #V0
    alpha, beta, gamma = decomposition.decomposition_V0(matrix)
    circuit.rz((alpha-gamma)/2,1)
    circuit.x(0)
    circuit.cx(0,1)
    circuit.x(0)
    circuit.ry(beta/2,1)
    circuit.rz((alpha+gamma)/2,1)
    circuit.x(0)
    circuit.cx(0,1)
    circuit.x(0)
    circuit.rz(-alpha,1)
    circuit.ry(-beta/2,1)
    
elif type == 4: #V1
    alpha, beta, gamma = decomposition.decomposition_V1(matrix)
    circuit.rz((alpha-gamma)/2,1)
    circuit.cx(0,1)
    circuit.ry(beta/2,1)
    circuit.rz((alpha+gamma)/2,1)
    circuit.cx(0,1)
    circuit.rz(-alpha,1)
    circuit.ry(-beta/2,1)
ket = Statevector(circuit)
ket.draw('latex')

circuit.draw('mpl')