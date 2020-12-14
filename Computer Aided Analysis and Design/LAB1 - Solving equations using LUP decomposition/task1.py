from Matrix import *

A = Matrix('./Matrices/A.txt')
b_vector = [12, 12, 1]

try:
	A.solveLU(b_vector)
except ValueError:
	print('=> Equation cannot be solved with LU decomposition.')
	

try:
	A.solveLUP(b_vector)
except ValueError:
	print('=> Equation cannot be solved with LUP decomposition.')