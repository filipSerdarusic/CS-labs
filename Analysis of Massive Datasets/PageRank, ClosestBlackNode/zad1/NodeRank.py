import sys
import numpy as np
from scipy.sparse import csr_matrix

'''
Implementation of PageRank algorithm.
The goal is to calculate the rank of every node in the given directed graph.
'''

# Reading input params
n, beta = sys.stdin.readline().split(' ')
n = int(n)
beta = float(beta)

rows = []
cols = []
probs = []

for i in range(n):

	row = list(map(int, sys.stdin.readline().split(' ')))
	row_len = len(row)

	rows.extend(row)
	cols.extend(row_len*[i])
	probs.extend(row_len*[1/row_len])

# Building adjacency matrix M
M = csr_matrix((probs, (rows, cols)), shape=(n,n))

# NodeRank
r = np.repeat(1/n, n).reshape(-1, 1)

stop = False
r_history = [r]

iter = 0
while(not stop):

	r_old = r

	r = beta * M @ r + (1 - beta)/n
	
	r_new = r
	stop = np.equal(r_old, r_new).all() or iter >= 100
	r_history.append(r)
	
	iter += 1

# Reading queries
q = int(sys.stdin.readline())

for _ in range(q):
	node, i = sys.stdin.readline().split(' ')
	node = int(node)
	i = int(i)
	
	if i < len(r_history):
		s = np.format_float_positional(r_history[i][node], precision=10, unique=False, trim='k')
		print(s)

	else:
		s = np.format_float_positional(r_history[-1][node], precision=10, unique=False, trim='k')
		print(s)