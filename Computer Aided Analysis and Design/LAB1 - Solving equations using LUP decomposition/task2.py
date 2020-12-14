from Matrix import *

A = Matrix('./Matrices/B.txt')
b_vector = [1, 2, 3]

try:
	A.solveLU(b_vector)
except (ValueError, ZeroDivisionError):
	print('=> Equation cannot be solved with LU decomposition.')

try:
	A.solveLUP(b_vector)
except (ValueError, ZeroDivisionError):
	print('=> Equation cannot be solved with LUP decomposition.')