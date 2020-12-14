from methods import *

print("\nSimpleks po Nelderu i Meadu\n" + 30*'-')
x_min4 = simplex(f4, [5,5])
y_min4 = f4(x_min4)
print("Funkcija f4\nX_min:", x_min4)
print("y_min:", y_min4)

print("\nHooke-Jeeves postupak\n" + 30*'-')
x_min4 = Hook_Jeeves(f4, [5,5])
y_min4 = f4(x_min4)
print("Funkcija f4\nX_min:", x_min4)
print("y_min:", y_min4)