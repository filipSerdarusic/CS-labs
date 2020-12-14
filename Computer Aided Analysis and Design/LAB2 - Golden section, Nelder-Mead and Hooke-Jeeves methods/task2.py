from methods import *
from counter import *

f1 = Function(f1)
f2 = Function(f2)
f3 = Function(f3)
f4 = Function(f4)

print("\nSimpleks po Nelderu i Meadu\n" + 30*'-')

x_min1 = simplex(f1, [-1.9, 2])
y_min1 = f1(x_min1)
print("Funkcija f1\nX_min:", x_min1)
print("y_min:", y_min1)
print("Broj evaulacija funkcije:", f1.counter, '\n')
f1.reset()

x_min2 = simplex(f2, [0.1, 0.3])
y_min2 = f2(x_min2)
print("Funkcija f2\nX_min:", x_min2)
print("y_min:", y_min2)
print("Broj evaulacija funkcije:", f2.counter, '\n')
f2.reset()

x_min3 = simplex(f3, [0,0,0,0,0])
y_min3 = f3(x_min3)
print("Funkcija f3\nX_min:", x_min3)
print("y_min:", y_min3)
print("Broj evaulacija funkcije:", f3.counter, '\n')
f3.reset()

x_min4 = simplex(f4, [5.1, 1.1])
y_min4 = f4(x_min4)
print("Funkcija f4\nX_min:", x_min4)
print("y_min:", y_min4)
print("Broj evaulacija funkcije:", f4.counter, '\n')
f4.reset()


print("Hooke-Jeeves\n" + 30*'-')

x_min1 = Hook_Jeeves(f1, [-1.9, 2])
y_min1 = f1(x_min1)
print("Funkcija f1\nX_min:", x_min1)
print("y_min:", y_min1)
print("Broj evaulacija funkcije:", f1.counter, '\n')
f1.reset()

x_min2 = Hook_Jeeves(f2, [0.1, 0.3])
y_min2 = f2(x_min2)
print("Funkcija f2\nX_min:", x_min2)
print("y_min:", y_min2)
print("Broj evaulacija funkcije:", f2.counter, '\n')
f2.reset()

x_min3 = Hook_Jeeves(f3, [0,0,0,0,0])
y_min3 = f3(x_min3)
print("Funkcija f3\nX_min:", x_min3)
print("y_min:", y_min3)
print("Broj evaulacija funkcije:", f3.counter, '\n')
f3.reset()

x_min4 = Hook_Jeeves(f4, [5.1, 1.1])
y_min4 = f4(x_min4)
print("Funkcija f4\nX_min:", x_min4)
print("y_min:", y_min4)
print("Broj evaulacija funkcije:", f4.counter, '\n')
f4.reset()


print("Koordinatno trazenje\n" + 30*'-')

x_min1 = koordinatno_trazenje(f1, [-1.9, 2])
y_min1 = f1(x_min1)
print("Funkcija f1\nX_min:", x_min1)
print("y_min:", y_min1)
print("Broj evaulacija funkcije:", f1.counter, '\n')
f1.reset()

x_min2 = koordinatno_trazenje(f2, [0.1, 0.3])
y_min2 = f2(x_min2)
print("Funkcija f2\nX_min:", x_min2)
print("y_min:", y_min2)
print("Broj evaulacija funkcije:", f2.counter, '\n')
f2.reset()

x_min3 = koordinatno_trazenje(f3, [0,0,0,0,0])
y_min3 = f3(x_min3)
print("Funkcija f3\nX_min:", x_min3)
print("y_min:", y_min3)
print("Broj evaulacija funkcije:", f3.counter, '\n')
f3.reset()

x_min4 = koordinatno_trazenje(f4, [5.1, 1.1])
y_min4 = f4(x_min4)
print("Funkcija f4\nX_min:", x_min4)
print("y_min:", y_min4)
print("Broj evaulacija funkcije:", f4.counter, '\n')
f4.reset()