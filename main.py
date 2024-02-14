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

#matrix = matrix_0V
matrix = (1/2)*np.array([
[1,1,1,1],
[1,-1,1,-1],
[1,1,-1,-1],
[1,-1,-1,1]])

qc = QuantumCircuit(2,2)
#qc.x(0)
#qc.x(1)
qc.barrier()

d2.twoqubit_to_single(qc, matrix)

ket = Statevector(qc)
ket.draw('latex')

qc.draw('mpl')