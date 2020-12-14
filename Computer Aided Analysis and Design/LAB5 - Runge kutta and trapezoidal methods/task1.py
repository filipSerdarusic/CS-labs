from matrix import *

A = Matrix(file='./matrices/matrix_A.txt')
A.print()

A_inv = A.inverse()

print(A_inv)