from matrix import *
import matplotlib.pyplot as plt
import numpy as np
import sys


def runge_kutta(A, x, T, t_max):
    x_k = Matrix.clone(x)
    t = 0
    ts = []
    x1s = []
    x2s = []
    while t <= t_max:
        m1 = A * x_k
        m2 = A * (x_k + m1 * T * 0.5)
        m3 = A * (x_k + m2 * T * 0.5)
        m4 = A * (x_k + m3 * T)
        x_k = x_k + (m1 + m2*2 + m3*2 + m4) * (T / 6)
        t += T
        print (t, x_k)
        ts.append(t)
        x1s.append(x_k.matrix[0][0])
        x2s.append(x_k.matrix[1][0])

    plt.plot(ts, x1s, label='x1')
    plt.plot(ts, x2s, label='x2')
    plt.legend(loc='best')
    plt.grid()
    plt.show()
    return x_k


def trapese(A, x, T, t_max):

    U = Matrix(array=[[1, 0], [0, 1]])
    R = (U - A * T * 0.5).inverse() * (U + A * T * 0.5)

    x_k = Matrix.clone(x)
    t = 0
    ts = []
    x1s = []
    x2s = []
    while t <= t_max:
        x_k = R * x_k
        t += T
        print (t, x_k)
        ts.append(t)
        x1s.append(x_k.matrix[0][0])
        x2s.append(x_k.matrix[1][0])

    plt.plot(ts, x1s, label='x1')
    plt.plot(ts, x2s, label='x2')
    plt.legend(loc='best')
    plt.grid()
    plt.show()
    return x_k

