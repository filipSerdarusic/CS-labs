from Matrix import *

A = Matrix('./Matrices/E.txt')
b_vector = [9000000000, 15, 0.0000000015]

A.solveLUP(b_vector)