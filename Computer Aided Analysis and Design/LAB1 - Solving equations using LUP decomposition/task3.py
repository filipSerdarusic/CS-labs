from Matrix import *

A = Matrix('./Matrices/C.txt')
b_vector = [12000000.000001, 14000000, 10000000]

A.solveLU(b_vector)

A.solveLUP(b_vector)