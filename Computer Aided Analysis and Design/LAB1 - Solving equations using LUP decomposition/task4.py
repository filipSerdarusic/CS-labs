from Matrix import *

A = Matrix('./Matrices/D.txt')
b_vector = [6,9,3]

A.solveLUP(b_vector)