from methods import *
from counter import *
from random import uniform

print("Koordinatno trazenje\n" + 30*'-')

for i in range(5):
	x = uniform(-50,50)
	y = uniform(-50,50)
	print("\n%d. Pocetna tocka = (%f,%f)" % (i+1,x,y))
	x_min = koordinatno_trazenje(f5, [x,y])
	y_min = f5(x_min)
	print("X_min:", x_min)
	print("y_min:", y_min)