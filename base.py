##################################################
#	Non-contiguous Graph Solver
#	Assignment 4 - CS 440 @ UH Hilo
#
#	Professor 	: Dr. M. Peterson
#	Students	: Cunnyngham, I.
#				: Perkins, J.
#				: Wessels, A.
#
#	Python implementaion of Dr. Peterson's 
#		template for this project.  Most comments stolen 
#		verbatum.  Statistical functions in Fxrandom are a
#   Python implementation of Dr. Peterson's Java 
#		implementation of a C++ random generator written by 
#		Dr. Mateen Rizki of the Department of Computer Science 
#		& Engineering, Wright State University - Dayton, Ohio, U.S.A.
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
				count = 0.0;
				for i,b in enumerate(solution.set):
					if (solution.set[i]):
						count += 1.0
				sol = count/len(solution.set)
				#if sol > .5:
				#	print test
				#	print solution.set
				print "%i verticies with %.2f connectivity, solution percentage: %.2f"%(x, cnn, sol)
				#print "Solution percentage: %.2f"%()
	
	def realConnectivity(self, vector):
		count = 0.0;
		for i in vector:
			if (vector[i]):
				count += 1.0
		return count/len(vector)

### Vertex Set Class ###

class VSet:
	""" A class representing a list of vertices in a graph """
	
	def __init__(self, size):
		""" Constructor for vertex set, accepts size and initializes all vertices to False """
		self.sizeN = size
		self.set = [False for i in range(size)]
		
	def toggleVertex(self, i):
		""" Toggle whether a vertex at the given index is included in the set or not """
		self.set[i] = not self.set[i]
	
	def total(self):
		""" Returns the number of verticies in this set """
		return self.set.count(True)
	
	def pagePrint(self):
		""" Prints the members of this set as 1s(included) and 0s(excluded) grouped in sets of 50 """
		for i, b in enumerate(self.set):
			sys.stdout.write("1") if self.set[i] else sys.stdout.write("0")
			if ((i+1)%50==0):
				print "" 
	
	def __repr__(self):
		return "Vertex set: \n" + str(self.set)
		
	def __getitem__(self, key):
		return self.set[key]
	
	def __setitem__(self, key, value):
		self.set[key] = value
	
	def randomSolution(self, cnn = .2):
		rand = Fxrandom(seed)
		for i in range(self.sizeN):
			self.set[i] = rand.boolBernoulli(cnn)


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
		""" Returns true if an edge exists between the two verticies, false otherwise or if invalid """
		# Check legality of verticies
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
		
		return vs
	
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
		return (setSize*setSize)-(connections*connections)
	
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
	
	def Genetic(self, population=100, generations=50):
		siblings = [VSet(self.sizeN).randomSolution() for i in range(population)]
		print siblings
		#for (g in range(generations)):
		
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
