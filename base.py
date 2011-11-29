##################################################
#	Non-contiguous Graph Solver
#	Assignment 4 - CS 440 @ UH Hilo
#
#	Professor 	: Dr. M. Peterson
#	Students	: Cunnyngham, I.
#				: Perkins, J.
#				: Wessels, A.
#
#		Python implementaion of Dr. Peterson's 
#	template for this project.  Most comments stolen 
#	verbatum.  Statistical functions in Fxrandom are a
#   Python implementation of Dr. Peterson's Java 
#	implementation of a C++ random generator written by 
#	Dr. Mateen Rizki of the Department of Computer Science 
#	& Engineering, Wright State University - Dayton, Ohio, U.S.A.
##################################################

#!/usr/bin/python

import sys
import datetime
import math


### Statistics class ###

# Global Constants
PI = 3.1415926535897;
ONEOVERPI = 1.0/PI;
TWOPI = 2.0*PI;

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
			print self.adjMatrix[i]
	
	def total(bitSet=[]):
		""" Given an array of boolean values, returns number of True-valued members """
		return bitSet.count(True)
	
	def checkEdgePresent(self, v1, v2):
		""" Returns true if an edge exists between the two verticies, false otherwise or if invalid """
		# Check legality of verticies
		if (v1 < 0) or (v2 < 0) or (v1 >= self.sizeN) or (v2 >= self.sizeN): 
			return False
		return self.adjMatrix[v1][v2]
	
	def edgePresent(self, v1, v2):
		""" Same as checkEdgePresent, minus the validity checks """
		return self.adjMatrix[v1][v2]
	
	def printBitset(self, bitSet):
		""" Prints given bool array as 1s and 0s grouped in sets of 50 """
		for i, b in enumerate(bitSet):
			sys.stdout.write("1") if bitSet[i] else sys.stdout.write("0")
			if ((i+1)%50==0):
				print "" 
	
	def greedySolutaion(self):
		""" Finds an independent set using a greedy solution """
		#Initialize an empty set
		set = [False for i in range(self.sizeN)]
		
		#TODO: implement greedy algorithm
		
		return set
	
	def evaluateSet(self, set):
		""" Test to see if a passed set is independent, if yes, size of set is returned, -1 elsewise """
		# Skip error test and assume len(set) == sizeN for quickness of algorithm
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

