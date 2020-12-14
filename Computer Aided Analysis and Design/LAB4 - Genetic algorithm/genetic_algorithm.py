import numpy as np
import random
import math

class genetic_algorithm():

	def __init__(	self,
					function,
					chromosomeShape,
					chromosomeRange=(-20,20),
					binary = False,
					numOfBits = 32,
					selection = 'roulette', 
					elitism = 1,
					populationSize = 40,
					mutationProbability = .1,
					mutationScale = 0.5,
					numIterations = 1000,
					fitnessTreshold = 10**6
					):
		
		self.f = function
		self.chromosomeShape = chromosomeShape
		self.range = chromosomeRange
		self.populationSize = populationSize
		self.selection = selection
		self.keep = elitism
		self.p = mutationProbability
		self.k = mutationScale
		self.binary = binary
		self.numOfBits = numOfBits
		self.numIter = numIterations
		self.fitnessTreshold = fitnessTreshold
		self.i = 0

		self.population = []

		if binary is True:
			for _ in range(populationSize):

				chromosome = [np.random.randint(2, size=self.numOfBits) for _ in range(chromosomeShape)]

				fitness = self.calculateFitness(chromosome)
				self.population.append((chromosome, fitness))

		else:
			for _ in range(populationSize):

				chromosome = [random.uniform(self.range[0], self.range[1]) for _ in range(chromosomeShape)]

				fitness = self.calculateFitness(chromosome)
				self.population.append((chromosome, fitness))

		self.population = sorted(self.population, key = lambda t: -t[1])


	def calculateFitness(self, chromosome):
		if self.binary is True:
			chromosome = self.binary_to_float(chromosome)

		fitness = 1./self.f(chromosome)
		return fitness


	def best(self):
		return self.population[0]
		

	def bestN(self, n):
		return [self.population[i] for i in range(n)]


	def rouletteParents(self, numParents=2):
		"""
			Choosing parents for crossover via Roulette wheel selection
		"""
		total_fitness = 0
		upper_margin = {}
		current_margin = 0
		margins = []

		for c, fit in self.population:
			total_fitness =+ fit
		
		for c, f in self.population:
			x = current_margin + float(f) / total_fitness
			upper_margin[x] = c
			margins.append(x)
			current_margin = x

		rand1 = random.random()
		rand2 = random.random()

		p1 = None
		p2 = None
		for k in margins:
			if (rand1 < k and p1 is None):
				p1 = np.array(upper_margin[k])
			
			if (rand2 < k and p2 is None):
				p2 = np.array(upper_margin[k])
		
		return p1, p2


	def tournamentParents(self, numParents=3):
		"""
			Choosing parents for crossover via Tournament selection
		"""
		tournament = []
		for _ in range(numParents):
	
			tournament.append(self.population[random.randint(0, self.populationSize - 1)])

		tournament = sorted(tournament, key = lambda t : -t[1])

		return tournament[0][0], tournament[1][0]


	def step(self):

		self.i += 1

		new_population = []
		new_population[:] = self.bestN(self.keep)

		while((len(new_population)) < self.populationSize):

			if self.selection is 'roulette':
				p1, p2 = self.rouletteParents()

			else:
				p1, p2 = self.tournamentParents()

			if self.binary is True:
				child = self.binary_crossover(p1,p2)
				child = self.binary_mutation(child)

			else:
				child = self.crossover(p1, p2)
				child = self.mutate(child)

			new_population.append((child, self.calculateFitness(child)))

		self.population = new_population
		self.population = sorted(self.population, key = lambda t : -t[1])

		best = self.best()
		stop = best[1] >= self.fitnessTreshold or self.numIter == self.i
		return (stop, self.i, best)


	def crossover(self, p1, p2):

		new = []
		for x,y in zip(p1, p2):
			new.append((x+y)/2)
		
		return np.array(new)


	def mutate(self, chromosome):

		mutated = []
		for x in chromosome:
			if random.random() < self.p:
				mutated.append(x + random.gauss(0, self.k))
			else:
				mutated.append(x)
		return np.array(mutated)


	def binary_to_float(self, chromosome):

		float_representation = []
		dg = self.range[0]
		gg = self.range[1]
		for binary in chromosome:
			b = 0
			for i in range(self.numOfBits):
				b += binary[i] * math.pow(2,i)

			x = dg + (float) (b / (math.pow(2 , self.numOfBits) - 1))*(gg-dg)

			float_representation.append(x)
		return np.array(float_representation)


	def binary_crossover(self,p1, p2):

		new = []
		rand = random.choice([True, False])

		if rand:
			i = random.randint(1,self.numOfBits-1)
			new[:i] = p1[:i]
			new[i:] = p2[i:]
			return new

		else:
			for a,b in zip(p1,p2):
				r = np.random.randint(2, size=self.numOfBits)

				x = np.bitwise_or(np.bitwise_and(a,b), np.bitwise_and(r,np.bitwise_xor(a,b)))

				new.append(x)
			return new


	def binary_mutation(self, chromosome):

		mutated = []
		for bin_vec in chromosome:
			mutated_bin_vec = len(bin_vec) * [0]

			for i in range(len(bin_vec)):

				if random.random() < self.p:
					mutated_bin_vec[i] = 1 if bin_vec[i] == 0 else 0
				else:
					mutated_bin_vec[i] = bin_vec[i]

			mutated.append(np.array(mutated_bin_vec))
		return mutated


	def printPopulation(self):
		for p in self.population:
			print("chromosome:\t", p[0] , "\tfitness:\t", p[1])