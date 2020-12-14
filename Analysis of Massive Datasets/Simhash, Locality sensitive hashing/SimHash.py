import sys
import hashlib
import numpy as np

# SimHash algorithm
def simhash(input, num_bits=128, hexadecimal=True):
	sh = 128*[0]
	for x in input.split(' '):
		hash_x = hashlib.md5(x.encode()).hexdigest()

		hash_x = bin(int(hash_x, 16))[2:]
		hash_x = hash_x.zfill(num_bits)
		
		for i in range(num_bits):
			if hash_x[i] == '1':
				sh[i] += 1
			else:
				sh[i] -= 1
		
	sh = ''.join(list(map(lambda  x: str(1*(x>=0)), sh)))

	if hexadecimal:
		return hex(int(sh, 2))[2:]
	else:
		return sh

# Reading from the input file
num_docs = int(sys.stdin.readline())
documents = []
for _ in range(num_docs):
	doc = sys.stdin.readline().rstrip()
	doc = simhash(doc, num_bits=128, hexadecimal=False)
	documents.append(np.array(list(doc)))

num_queries = int(sys.stdin.readline())
queries = []
for _ in range(num_queries):
	i, k = sys.stdin.readline().rstrip().split(' ')
	queries.append((int(i), int(k)))

# Sequential search for similar documents
for i, k in queries:
	a = documents[i]
	similar_docs = 0
	for j in range(num_docs):
		if i == j:
			continue
		b = documents[j]
		diff = np.count_nonzero(a != b)
		if diff <= int(k):
			similar_docs += 1
	print(similar_docs)