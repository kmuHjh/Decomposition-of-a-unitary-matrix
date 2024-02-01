%matplotlib inline

import math
import decomposition_2qubit as d2
import numpy as np
from qiskit import QuantumCircuit, execute, Aer
from qiskit.visualization import plot_histogram
from qiskit.extensions import *
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

matrix = matrix_0V
type = d2.distinguish(matrix)

qc = QuantumCircuit(2,2)
qc.barrier()

if type == 1:  #0V
    d2.decomposition_0V(qc, matrix)

elif type == 2: #1V
    d2.decomposition_1V(qc, matrix)
    
elif type == 3: #V0
    d2.decomposition_V0(qc, matrix)
    
elif type == 4: #V1
    d2.decomposition_V1(qc, matrix)

ket = Statevector(qc)
ket.draw('latex')

qc.draw('mpl')