from rungeKutta_trapezoid import *
import matplotlib.pyplot as plt
import numpy as np
import sys

A = Matrix(array=[[0, 1],[-1, 0]])
x = Matrix(array=[[0],[1]])

trapese(A, x, 0.1, 800)

runge_kutta(A, x, 0.1, 800)

plt.show()