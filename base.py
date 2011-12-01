##################################################
#	Non-contiguous Graph Solver
#	Assignment 4 - CS 440 @ UH Hilo
#
#	Professor 	: Dr. M. Peterson
#	Students	: Cunnyngham, I.
#			: Perkins, J.
#			: Wessels, A.
#
#	Python implementaion of Dr. Peterson's 
# template for this project.  Most comments stolen 
# verbatum.  Statistical functions in Fxrandom are a
# Python implementation of Dr. Peterson's Java 
# implementation of a C++ random generator written by 
# Dr. Mateen Rizki of the Department of Computer Science 
# & Engineering, Wright State University - Dayton, Ohio, U.S.A.
##################################################

#!/usr/bin/python

import sys
import datetime
import math

#throwaway test class
class test:
	def __init__(self):
		for y in range(1,10):
			cnn = y/10.0
			for x in range(2,30):	
				test = Graph(x, cnn)
				solution = test.greedySolution()
				sol = solution.set.count(True)/float(len(solution.set))
				print "%i vertices with %.2f connectivity, solution percentage: %.2f"%(x, cnn, sol)

### Vertex Set Class ###

class VSet:
	""" A class representing a list of vertices in a graph """
	
	def __init__(self, size, random=None, percentage=.2):
		if type(size) is list: #allows me to initialize a VSet with a list...
			self.set = size
		else:
			""" Constructor for vertex set, accepts size, Fxrandom object (opt), and True/False density as percentage (opt) """
			if random and random.boolBernoulli:
				#rand = Fxrandom(seed)
				self.set = [random.boolBernoulli(percentage) for i in range(size)]
			else:
				""" Initializes all vertices to False """
				self.set = [False for i in range(size)]
		self.fitness = -1.0 #means "unknown"
	
	def toggleVertex(self, i):
		""" Toggle whether a vertex at the given index is included in the set or not """
		self.set[i] = not self.set[i]
	
	def total(self):
		""" Returns the number of vertices in this set """
		return self.set.count(True)
	
	def pagePrint(self):
		""" Prints the members of this set as 1s(included) and 0s(excluded) grouped in sets of 50 """
		for i, b in enumerate(self.set):
			sys.stdout.write("1") if self.set[i] else sys.stdout.write("0")
			if ((i+1)%50==0):
				print "" 
	
	def __repr__(self):
		return "Vertex set: %s\nFitness: %.3f"%(self.set, self.fitness)
		
	def __getitem__(self, key):
		return self.set[key]
	
	def __setitem__(self, key, value):
		self.set[key] = value
	
	@classmethod
	def lexSet(cls, lexIndex, size):
		""" Generate's the [lexIndex]th lexicographical set of vertices of size [size] """
		# Check to make sure the value passed is a valid lexicographical index
		if lexIndex < 0 or lexIndex > (2**size)-1:
			raise ValueError("Lexicographical index must be between 0 & (2**size)-1")
		
		# Generate the set by converting a number to binary, filling it to the 
		# correct size, and converting the resultant bits into boolean values
		s = cls(size)
		s.set = [ bool(int(x)) for x in bin(lexIndex).split('b')[1].zfill(size)]
		return s

### Graph Class ###	

class Graph:
	""" A class representing a undirected graph with random connectivity """
	
	def __init__(self, size, cnn, seed=0):
		"""  This is the public constructor for the graph class.  
		
		  Size is the number
		  of vertices in the graph, cnn in the connectivity of the graph (the
		  probability of there being an edge between two arbitrary nodes); cnn must
		  be between 0.0 * 1.0 inclusive; and seed is the seed value for the
		  random number generator.  By default, use 0 for a random seed based on 
		  the system clock.
		  
		  size -- Number of vertices in the graph.
		  cnn  -- Connectivity of the graph.
		  seed -- Random number generator seed (default 0)."""
		
		# Perform error checking
		if size <= 0:
			raise ValueError("Graph Init: size must be > 0.")
		if cnn < 0.0 or cnn > 1.0:
			raise ValueError("Graph Init: connectivity must be between 0.0 & 1.0.")
		if seed < 0:
			raise ValueError("Graph Init: seed must be >= 0.")
		
		# Define class variables
		self.sizeN = size
		self.rand = Fxrandom(seed)
		print "Initial seed: %s"%(self.rand.seed) #This can be useful if we want to be able reproduce the same results.
		
		# Instantiate the class's adjacency matrix to all False values
		self.adjMatrix = [ [False for j in range(self.sizeN) ] for i in range(self.sizeN) ]
		
		# Initialize edges according to a weighted coin flip.
        # if cnn = 0.1, there will be a 10% chance of creating an edge
        # between vertices i and j.  Note that there will never be an
        # edge between a vertex and itself.
		for i in range(self.sizeN):
			for j in range(i+1, self.sizeN):
				self.adjMatrix[i][j] = self.adjMatrix[j][i] = self.rand.boolBernoulli(cnn)
			# Debug code
			#print self.adjMatrix[i]
	
	def checkEdgePresent(self, v1, v2):
		""" Returns true if an edge exists between the two vertices, false otherwise or if invalid """
		# Check legality of vertices
		if (v1 < 0) or (v2 < 0) or (v1 >= self.sizeN) or (v2 >= self.sizeN): 
			return False
		return self.adjMatrix[v1][v2]
	
	def edgePresent(self, v1, v2):
		""" Same as checkEdgePresent, minus the validity checks """
		return self.adjMatrix[v1][v2]
	
	def greedySolution(self):
		""" Finds an independent set using a greedy solution """
		# Initialize an empty set
		vs = VSet(self.sizeN)
		
		# Rank the vertexes by their connectedness
		vrank = []
		for i, row in enumerate(self.adjMatrix):
			vrank.append([i, row.count(True)])
		vrank = sorted(vrank, key=lambda v: v[1])
		
		# Try to add each vector from lowest connectedness up.  
		# If the vector breaks the set, toggle it back.
		for v in vrank:
			vs.toggleVertex(v[0])
			if self.evaluateSet(vs) == -1:
				vs.toggleVertex(v[0])
		self.setFitness(vs)
		return vs
	
	def exhaustiveSolution(self):
		""" Generate the biggest possible independent set of vertices by testing all possibilities """
		maxScore = 0
		maxIndex = -1
		for i in range(1, (2**self.sizeN)):
			curScore = self.evaluateSet( VSet.lexSet(i, self.sizeN) )
			if curScore > maxScore:
				maxIndex = i
		return VSet.lexSet(maxIndex, self.sizeN)

	def rouletteSelection(self, popsel, popnumber):
		"""	Stochastic Sampling (Roulette wheel) method of selecting parents
			Source: http://www.cse.unr.edu/~banerjee/selection.htm """
		choice = [ ]
		k=0
		while (k < popnumber):
		    partsum = 0
		    parent = 0
		    randnum = self.rand.uniform(0.0,1.0)
		    for i in range(0, popnumber):
		        partsum += popsel[i].rank
		        #print "%s %s"%(i, popsel[i].rank)
		        if partsum >= randnum:
		            parent = i
		            break
		    k+=1
		    choice.append(parent)
		return choice
		
	def combine(self, group1, group2):
		return VSet([group1[i] if self.rand.boolBernoulli(.5) else group2[i] for i,v in enumerate(group1)])
		
	def GASolution(self, popsize=100, generations=50, percentage=.10):
		""" Finds an independent set using a Genetic Algorithm """
		population = []
		total = 0.0
		#initialize the population
		for i in range(popsize):
			s = VSet(self.sizeN, random=self.rand, percentage=percentage)
			self.setFitness(s)
			total += s.fitness
			population.append(s)
		
		for p in population:
			p.rank = p.fitness/total
			
		print "Average before: %.2f"%(total/popsize)
		
		for g in range(generations):
			total = 0.0
			#population = sorted(population, key=lambda s: s.rank, reverse=True)
			#maxrank = float(max(population, key=lambda s:s.rank).rank)
			
			males = self.rouletteSelection(population, popsize)
			females = self.rouletteSelection(population, popsize)
			for i in range(popsize):
				s = self.combine(population[males[i]], population[females[i]])
				self.setFitness(s)
				total += s.fitness
				population[i] = s
			
			for p in population:
				p.rank = p.fitness/total
		
			#population = [VSet(self.sizeN, random=self.rand) for i in range(popsize)]
			#for i, s in enumerate(siblings):
			#	s.rank = self.evaluateSet(s)
		
		
		
		#for i,s in enumerate(population):
		#	print "%i: %.8f"%(i,s.rank)
			#print "%i: %.1f %s"%(i,s.rank, s)
		
		print "Average after: %.2f"%(total/popsize)
		return max(population, key=lambda s:s.rank)
	
	
	def setFitness(self, vset):
		""" Test the fitness of a passed set: fitness= [set size]^2 - [connections]^2 """
		# Skip error test and assume len(set) == sizeN for quickness of algorithm
		set = vset.set
		
		setSize = connections = 0
		for i in range(self.sizeN):
			if set[i] :
				for j in range(i+1, self.sizeN):
					if set[j] and self.adjMatrix[i][j]:
						connections+=1
				setSize+=1
		fitness = (setSize*setSize)-(connections*connections)
		vset.fitness = fitness
		return fitness
	
	def evaluateSet(self, vset):
		""" Test to see if a passed set is independent, if yes, size of set is returned, -1 elsewise """
		# Skip error test and assume len(set) == sizeN for quickness of algorithm
		set = vset.set
		
		setSize = 0
		independent = True
		for i in range(self.sizeN):
			if set[i] :
				for j in range(i+1, self.sizeN):
					if set[j] and self.adjMatrix[i][j]:
						independent = False
						break
				if not independent: 
					break
				setSize+=1
		if independent:
			return setSize
		return -1
		
	def __repr__(self):
		ret = "Adjacency Matrix: \n"
		for i,l in enumerate(self.adjMatrix):
			for j,v in enumerate(self.adjMatrix[i]):
				ret += "%i    "%self.adjMatrix[i][j]
			ret += "\n\n"
		return ret #str(self.adjMatrix)
		

### Statistics class ###

class Fxrandom:
	""" A class with some useful statistical functions """
	
	def __init__(self, initSeed=0):
		""" Constructor: set class variables, set seed as specified or by system clock if 0(default) """
		self.modulus = 0x7fffffff;
		self.multiplier = 630360016;
		self.setSeed(initSeed)
	
	def setSeed(self, initSeed=0):
		""" Reset seed: If initSeed=0 (default) set seed to clock miliseconds, else use passed seed """
		if initSeed==0:
			self.seed = datetime.datetime.now().microsecond/1000
		else:
			self.seed = initSeed
	
	def random(self):
		""" Returns a random integer 0...0x7fffffff """
		self.seed = ((self.multiplier * self.seed) % self.modulus)
		return math.fabs(self.seed)
	
	def uniform(self, min, max):
		""" Returns a random double between min & max """
		val=0.0
		if min >= max:
			raise ValueError("min must be less than max.")
		val = (min + (max - min) * (self.random())/(self.modulus))
		return val
	
	def boolBernoulli(self, probability):
		""" Return an Bernoulli distributed random variable with given probability """
		num = self.uniform(0.0, 1.0)
		if(num <= probability):
			return True
		return False
