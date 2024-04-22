%matplotlib inline

import decomposition_2qubit as d2
import numpy as np
from qiskit import *
from qiskit.quantum_info import Statevector

#test set
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

#test1
matrix_1 = (1/np.sqrt(3)) * np.array([
    [1, 1, 0, 1],
    [1, 0, 1, -1],
    [0, -1, 1, 1],
    [-1, 1, 1, 0]
])
#test2
matrix_2 = (1/2)*np.array([
        [1, 1, 1, 1],
        [1, 1j, -1, -1j],
        [1, -1, 1, -1],
        [1, -1j, -1, 1j]
    ])
    
qc = QuantumCircuit(2,2)
#qc.x(0)
#qc.x(1)
qc.barrier()

d2.decomposition_2q(qc, matrix)

ket = Statevector(qc)
ket.draw('latex')

qc.draw('mpl')