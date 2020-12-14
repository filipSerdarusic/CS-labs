import sys
import numpy as np

'''

Implementation of Dijkstra's shortest path algorithm.
In an undirected graph there are white and black nodes. The task is to find the nearest black node for every node in a graph.

'''

def dijkstra(graph, source):

	inf = float('inf')
	nodes = [i for i in range(n)]
	distances = [0 if i is source else inf for i in nodes]

	while nodes:		
		current = min(nodes, key=lambda x : distances[x])
		for node in graph[current]:
			d = distances[current] + 1
			if d < distances[node]:
				distances[node] = d
		nodes.remove(current)

	return np.array(distances)

# Reading the input graph
n, e = list(map(int, sys.stdin.readline().split(' ')))

black_nodes = []
for i in range(n):
	color = int(sys.stdin.readline())
	if color is 1:
		black_nodes.append(i)

graph = {}
for i in range(e):
	s, d = list(map(int, sys.stdin.readline().split(' ')))
	if s in graph.keys():
		graph[s].append(d)
	else:
		graph[s] = [d]
	
	if d in graph.keys():
		graph[d].append(s)
	else:
		graph[d] = [s]

	graph[s].sort()
	graph[d].sort()


# Closest black node
for i in range(n):

	dist = dijkstra(graph, i)

	node = black_nodes[np.argmin(dist[black_nodes])]
	distance = dist[node]

	output = str(node) + ' ' + str(distance)

	print(output)