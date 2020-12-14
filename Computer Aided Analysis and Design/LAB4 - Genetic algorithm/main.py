import math
from genetic_algorithm import *

def f1(x):
    y = 100 * math.pow((x[1] - math.pow(x[0],2)),2) + math.pow((1 - x[0]),2)
    return float(y)

def f2(x):
    y = math.pow((x[0] - 4), 2) + 4 * math.pow((x[1] - 2), 2)
    return y

def f3(x):
    y = 0
    for i in range(len(x)):
        y += math.pow((x[i] - (i+1)),2)
    return float(y)

def f4(x):
    y = abs((x[0]-x[1]) * (x[0]+x[1])) + math.pow( (math.pow(x[0],2) + math.pow(x[1],2)), 0.5)
    return float(y)

def f5(x):
    sum = 0
    for i in x:
        sum += i**2
    y = 0.5 + (pow(math.sin(math.sqrt(sum)), 2) - 0.5) / pow(1 + 0.001 * sum, 2)
    return float(y)

def f7(x):
	sum = 0
	for x_ in x:
		sum += x_**2
	return sum**0.25 * (1 + math.sin(50*(sum)**0.1)**2)


GA = genetic_algorithm(function=f2,
						chromosomeShape=2,
						chromosomeRange=(-20,20),
						binary = False,
						selection='tournament',
						elitism=1,
						populationSize=50,
						mutationProbability=0.3,
						mutationScale=1,
						numIterations=10000
						)

stop = False
while not stop:

	stop, i, best = GA.step()
	print("Iteration: " , i , "\tbest chromosome:\t" , best[0] , "\tfitness:\t" , best[1])

print("DONE")
print("Number of iterations:" , i)
print("Best chromosome:", best[0] , "\tfitness:" , best[1])