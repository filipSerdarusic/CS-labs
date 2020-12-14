from rungeKutta_trapezoid import *
import matplotlib.pyplot as plt
import numpy as np
import sys


A = Matrix(array=[[0, 1],[-200, -102]])
x = Matrix(array=[[1],[-2]])

#trapese(A, x, 0.1, 10)

runge_kutta(A, x, 0.1, 10)

plt.show()