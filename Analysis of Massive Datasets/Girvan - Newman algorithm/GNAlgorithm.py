'''

Girvan - Newman's algorithm

The Girvan-Newman algorithm detects communitites by progressively removing edges with highest betweenness from the original network (graph).
The connected components of the remaining network are the communities.

'''

import sys
import copy
import random
import numpy as np


#  Returns list of neighbouring nodes of node_i
def get_neighbour_nodes(i, graph):
	neighbours = []
	for x,y in graph:
		if i not in (x,y):
			continue
		neighbours.append(x if y == i else y)
	return neighbours


# Find shortest paths between the source node and every other node in the graph
def find_shortest_paths(source, nodes, graph):

	inf = float('inf')
	distances = {node : 0 if node == source else inf for node in nodes}
	visited_nodes = []

	paths = {node : [] for node in nodes}
	paths[source] = [[source]]

	while nodes:

		current = min(nodes, key=lambda x : distances[x])
		visited_nodes.append(current)

		for node in get_neighbour_nodes(current, graph):

			if node in visited_nodes:
				continue

			if (current, node) in graph:
				w = graph[(current, node)]
			else:
				w = graph[(node, current)]

			d = distances[current] + w

			if d <= distances[node]:
				
				# Multiple shortest paths
				if d == distances[node]:
					
					for path in paths[current]:
						paths[node].append(path + [node])

				# Single shortest path (new one)
				else:
					
					paths[node] = [path + [node] for path in paths[current]]
					distances[node] = d

		nodes.remove(current)

	return paths


# Calculates edge betweenness - if the edge is on the shortest path between the
# two nodes, its edge betweenness increases.
def calculate_betweenness(graph, nodes):

	betweenness = {x : 0 for x in graph}

	for src_node in nodes:

		paths = find_shortest_paths(src_node, nodes[:], graph)

		for dst_node, path in paths.items():

			# Check if src_node and dst_node are the same or there's not path
			if src_node == dst_node or not path:
				continue

			N = 1/len(path)

			for p in path:

				for i in range(len(p)-1):
					if (p[i], p[i+1]) in graph:
						betweenness[(p[i], p[i+1])] += N
					else:
						betweenness[(p[i+1], p[i])] += N

	# Divide every edge betweenness with 2 (because the graph is undirected so
	#  every path is traversed two times)
	for edge in betweenness:
		betweenness[edge] /= 2
	
	return betweenness


# Depth-first search algorithm for finding reachable nodes
def dfs(visited, graph, node):
	if node not in visited:
		visited.add(node)
		for neighbour in get_neighbour_nodes(node, graph):
			dfs(visited, graph, neighbour)
	return visited


# Finds network communities - returns number of independent components in graph
def find_communities(graph, nodes):
	checked_nodes = set()
	communities = []

	while checked_nodes != set(nodes):
		visited = dfs(set(), graph, random.choice(nodes))
		if visited not in communities:
			communities.append(visited)

		checked_nodes.update(visited)

	return [sorted(list(comm)) for comm in communities]


# Calculates modularity of graph partitions
def calculate_modularity(graph, nodes, communities):

	Q = 0
	m = sum(graph.values())

	for u in nodes:
		
		k_u = sum([graph[x] for x in graph if u in x])

		for v in nodes:
			
			if (u,v) in graph:
				A_uv = graph[(u,v)]
			elif (v,u) in graph:
				A_uv = graph[(v,u)]
			else:
				A_uv = 0

			k_v = sum([graph[x] for x in graph if v in x])

			delta = False
			for comm in communities:
				if (u in comm) and (v in comm):
					delta = True
			
			if delta:
				t = (k_u * k_v)/(2 * m)
				Q += A_uv - t
		
	Q /= (2*m)
	return Q


# Create unweighted undirected graph 
graph = dict()
nodes = set()

for line in sys.stdin:
	if line == '\r\n' or line =='\n':
		break

	line = list(map(int, line.split(' ')))
	line.sort()
	v1, v2 = line

	nodes.add(v1)
	nodes.add(v2)
	graph[(v1, v2)] = 1


# Read vector properties from input file
properties = dict()
for line in sys.stdin:

	id, *vector = line.rstrip().split(' ')
	properties[int(id)] = np.array(vector).astype(int)

	# If node (id) has no edges to other nodes (i.e. didn't appear earlier), we need to add it to the list of nodes
	nodes.add(int(id))

nodes = sorted(list(nodes))


# Add weighted edges to the graph
weights = dict()

max_sim = len(vector)

for v1, v2 in graph:
	sim = np.count_nonzero(properties[v1] == properties[v2])
	dist = max_sim - (sim-1)

	graph[(v1,v2)] = dist


# Girvan - Newman algorithm
g = copy.deepcopy(graph)

communities = [nodes]
current_communities = [nodes]
modularities = []

while g:
	
	# Calculate graph modularity
	Q = round(calculate_modularity(g, nodes, current_communities), 4)
	if Q < 1e-5:
		Q = 0
	modularities.append(Q)

	betweenness = calculate_betweenness(g, nodes)

	# Find and remove edges with heighest betweenness
	max_value = max(betweenness.values())
	edges_to_remove = [x for x in betweenness if betweenness[x] == max_value]

	for edge in sorted(edges_to_remove):
		print(edge[0], edge[1])
		del g[edge]

	# Find communities
	current_communities = find_communities(g, nodes)
	
	if g:
		communities.append(current_communities)


# Optimal communites
i, max_Q = max(enumerate(modularities), key=lambda x: x[1])
best = communities[i]


# Make desired output
output = []
for comm in sorted(best, key=lambda x : (len(x), x[0])):
	out = '-'.join([str(x) for x in comm])
	output.append(out)

print(' '.join(output))