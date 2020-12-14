from methods import *
from counter import *

step = [3,6,14,17]
f1 = Function(f1)

for k in step:
	x_min = simplex(f1, [0.5, 0.5], k)
	y_min = f1(x_min)
	print("\nFunkcija f1, korak = ", k)
	print("X_min:" , x_min)
	print("y_min:", y_min)
	print("Broj evaulacija funkcije:", f1.counter)
	f1.reset()

print(30*'-')

for k in step:
	x_min = simplex(f1, [5,5], k)
	y_min = f1(x_min)
	print("\nFunkcija f1, korak = ", k)
	print("X_min:" , x_min)
	print("y_min:", y_min)
	print("Broj evaulacija funkcije:", f1.counter)
	f1.reset()