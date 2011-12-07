##################################################
#   I.ndependent V.ertex S.et P.roblem Solver
#   Assignment 4 - CS 440 @ UH Hilo
#
#   Professor   : Dr. M. Peterson
#   Students    : Cunnyngham, I.
#               : Perkins, J.
#               : Wessels, A.
#
#   Python implementaion of Dr. Peterson's 
# template for this project.  Most comments stolen 
# verbatum.  Statistical functions in Fxrandom are a
# Python implementation of Dr. Peterson's Java 
# implementation of a C++ random generator written by 
# Dr. Mateen Rizki of the Department of Computer Science 
# & Engineering, Wright State University - Dayton, Ohio, U.S.A.
#
##################################################

#!/usr/bin/python

import sys
import time
import datetime
import math
import random

### Vertex Set Class ###

class VSet:
    """ A class representing a list of vertices in a graph """
    
    def __init__(self, set=[]):
        """ Constructor for VSet, accepts an array of bools as default value of self.set """
        self.set = [ i for i in set ]
        self.fitness = -1
    
    def toggleVertex(self, i):
        """ Toggle whether a vertex at the given index is included in the set or not """
        self.set[i] = not self.set[i]
    
    def total(self):
        """ Returns the number of vertices in this set """
        return self.set.count(True)
    
    def density(self):
        """ Returns the percentage of true values in the set """
        return self.total()/float(len(self.set))
    
    def pagePrint(self):
        """ Prints the members of this set as 1s(included) and 0s(excluded) grouped in sets of 50 """
        for i, b in enumerate(self.set):
            sys.stdout.write("1") if self.set[i] else sys.stdout.write("0")
            if ((i+1)%50==0):
                print "" 
    
    def __repr__(self):
        rep = "VSet(%.3f): "%self.fitness
        for i,v in enumerate(self.set):
            rep += "%i"%v
        return rep+"\n"
        
    def __getitem__(self, key):
        return self.set[key]
    
    def __setitem__(self, key, value):
        self.set[key] = value
        
    def __len__(self):
        return len(self.set)
    
    @classmethod
    def emptySet(cls, size):
        """ Returns an empty set (all False valued) of [size] """
        return cls( [ False for i in range(size) ] )
    
    @classmethod
    def randomSet(cls, size, random, density=.2):
        """ Generates a random set, where individual vertex inclusion has [density] probability """
        return cls( [ random.boolBernoulli(density) for i in range(size) ] )
        
### Vertex Set Iterator Class ###

class VSetIter:
    """ A Class that iterates through all permutations of a vertex set """
    
    def __init__(self, size):
        """ Constructor which takes the size of the vertex sets to be generated """
        self.sizeN = size
        self.vs = VSet.emptySet( size )
    
    def __iter__(self):
        """ Make class callable by iterator functions """
        return self
    
    def next(self):
        """ Find the next perumutation from the current vertex. Covers all possibilties until set is full """
        size = k = len( self.vs.set )
        if self.vs.set.count(True) == size:
            raise StopIteration
        while k>=1 and self.vs.set[k-1]:
            k-=1
        
        if k>=1:
            self.vs.set[k-1]=True
            for i in range(k, size ):
                self.vs.set[i] = False
        return self.vs


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
          seed -- Random number generator seed (default 0).   testcase -- If you want the test case graph, set this to true.
          testcase -- Set to true for the 4x4 static test case
          fitfunc -- Set the fitness function for the GA and annealing solutions to use

        """
        # Perform error checking
        if size <= 0:
            raise ValueError("Graph Init: size must be > 0.")
        if cnn < 0.0 or cnn > 1.0:
            raise ValueError("Graph Init: connectivity must be between 0.0 & 1.0.")
        if seed < 0:
            raise ValueError("Graph Init: seed must be >= 0.")
        
        # Instantiate class variables
        self.cnn = cnn
        self.sizeN = size
        self.rand = Fxrandom(seed)
        
        fitfunc = self.goodFitness #define a global fit function to use by default...
        
        # Instantiate the class's adjacency matrix to all False values
        self.adjMatrix = [ [False for j in range(self.sizeN) ] for i in range(self.sizeN) ]
        
        # Initialize edges according to a weighted coin flip.
        # if cnn = 0.1, there will be a 10% chance of creating an edge
        # between vertices i and j.  Note that there will never be an
        # edge between a vertex and itself.
        for i in range(self.sizeN):
            for j in range(i+1, self.sizeN):
                self.adjMatrix[i][j] = self.adjMatrix[j][i] = self.rand.boolBernoulli(cnn)
    
    def greedySolution(self):
        """ Finds an independent set using a greedy solution """
        s = VSet.emptySet( self.sizeN )
        
        # Rank the vertexes by their connectedness
        vrank = []
        for i, row in enumerate(self.adjMatrix):
            vrank.append( [i, row.count(True) ])
        vrank = sorted( vrank, key=lambda v: v[1] )
                
        # Try to add each vector from lowest connectedness up.  
        # If the vector breaks the set, toggle it back.
        for v in vrank:
            s.toggleVertex( v[0] )
            if self.evaluateSet( s ) == -1:
                s.toggleVertex( v[0] )
        s.fitness = self.fitfunc( s )
        return s
    
    def shallowAnnealing(self, coolStep = 0.90):
        """ Generate the biggest set using the simulated annealing algorithm """
        bestSet = VSet.emptySet( self.sizeN )
        bestScore = 0
        for j in range(self.sizeN*25):
            frozen = False
            T = self.sizeN
            curSet = VSet.randomSet( self.sizeN, self.rand, self.cnn/2)
            newSet = VSet( curSet )
            while not frozen:
                newSet.toggleVertex( random.randrange(self.sizeN) )
                newSet.toggleVertex( random.randrange(self.sizeN) )
                dE = self.fitfunc(curSet) - self.fitfunc(newSet)
                if dE<0:
                    curSet = VSet( newSet )
                elif self.rand.boolBernoulli( math.exp( - dE / T ) ):
                    curSet = VSet( newSet )
                T *= coolStep
                if T < self.sizeN/10:
                    frozen = True
            score = self.evaluateSet( curSet )
            if score > bestScore:
                bestScore = score
                bestSet = VSet( curSet )
        bestSet.fitness = self.fitfunc(bestSet)
        return bestSet 
    
    def exhaustiveSolution(self):
        """ Generate the biggest possible independent set of vertices by testing all possibilities """
        maxScore = 0
        for s in VSetIter( self.sizeN ):
            curScore = self.evaluateSet( s )
            if curScore > maxScore:
                maxScore = curScore
                maxSet = VSet( s )
        maxSet.fitness = self.fitfunc(maxSet)
        return maxSet
    
    def mutate(self, value, rate=None):
        """ Mutates a given bit with [rate] liklihood """
        if rate == None:
            return value
        return not value if self.rand.boolBernoulli(rate) else value
    
    def combine(self, group1, group2, mutation):
        """ Given two parent vertex sets, produce an offspring vertex set by randomly picking between them and mutating """
        #vset = VSet( [self.mutate(group1[i] if self.rand.boolBernoulli(.5) else group2[i], mutation) for i in range(self.sizeN)] )
        vset = VSet( [group1[i] if self.rand.boolBernoulli(.5) else group2[i] for i in range(self.sizeN)] )
        if mutation and self.rand.boolBernoulli(mutation):
            idx = int(self.rand.uniform(0,self.sizeN))
            vset[idx] = not vset[idx]
        return vset
        
    def rouletteSelection(self, population, popsize, total_fitness):
        """    Stochastic Sampling (Roulette wheel) method of selecting parents
            This method requires some extra preperation of the data, and fails when fitness is negative...
            Source: http://www.cse.unr.edu/~banerjee/selection.htm """
        choice = [ ]
        choice_fitness = 0
        return_population = popsize
        #combined_size = popsize*2
        k=0
        while (k < return_population):
            partsum = 0
            parent = 0
            #print total_fitness
            randnum = self.rand.uniform(0.0, total_fitness)
            for i in range(0, popsize):
                partsum += population[i].fitness
                if partsum >= randnum:
                    parent = i
                    break
            k+=1
            fitness = population[parent].fitness
            #print "%i minus %f"%(parent, fitness)
            choice.append(population[parent])
            choice_fitness += fitness
        return (choice, choice_fitness)
        
    def GASolution(self, popsize=100, generations=300, density=None, mutation=0.01, preserve=0, fitfunc=None):
        """ Finds an independent set using a Genetic Algorithm """
        if fitfunc == None:
            fitfunc = self.fitfunc
        if not density:
            """ if no density is set, run greedy and just use a fraction of the density of the set returned"""
            density = self.greedySolution().density()*.5
            # print "Using density: %.5f"%density
        
        pop_range = range(popsize) #avoid remaking this list a bunch of times
        population = [None for x in pop_range] #initialize the list so i don't have to append a million times
        children = [None for x in pop_range]
        combined = []
        population_fitness = 0.0 #stores the current population's fitness total, used in roulette selection
        #lastindex = popsize-1
        
        """initialize the population"""
        for i in pop_range:
            s = VSet.randomSet(self.sizeN, self.rand, density)
            s.fitness = fitfunc(s)
            population_fitness += s.fitness
            population[i] = s
        
        # print "Average fitness before: %.2f"%(population_fitness/popsize)
        for g in range(generations):
            children_fitness = 0 #stores the total fitness for the children genomes
            for i in pop_range:
                rand1 = int(self.rand.uniform(0, popsize))
                rand2 = int(self.rand.uniform(0, popsize))
                while rand1 == rand2:
                    rand2 = int(self.rand.uniform(0, popsize))
                
                #set1 = population[rand1]
                #set2 = population[rand2]
                s = self.combine(population[rand1], population[rand2], mutation=mutation)
                s.fitness = fitfunc(s)
                children_fitness += s.fitness
                children[i] = s
                
            combined = population + children
            
            #this is more shitty!
            random.shuffle(combined)
            
            combined_len = popsize*2
            for i in pop_range:
                other = i+popsize
                if combined[i].fitness > combined[other].fitness:
                    population[i] = combined[i]
                else:
                    population[i] = combined[other]
            
            #(population, population_fitness) = self.rouletteSelection(combined, popsize, population_fitness+children_fitness)
            #print population
        
        best = population[0] 
        for i,t in enumerate(population):
            if self.evaluateSet(t) > 0 and (not best or best.fitness < t.fitness):
                best = t
                break
        #print "Average fitness after: %.2f"%(population_fitness/popsize)
        #print "Highest fitness: %.2f"%(sorted_population[0].fitness)
        #print "Lowest fitness: %.2f"%(sorted_population[-1].fitness)
        #print "Best: %s"%best
        #print population
        return best       
    
    def triangleFitness(self, s):
        """ Fitness= for i vertexes: sum( [tri] || - 1.5*[tri]).  [tri] = triangle number of ith vertex """
        # Skip error test and assume len(s) == sizeN for quickness of algorithm
        setSize = fitness = 0.0
        for i in range(self.sizeN):
            if s[i] :
                ind = True
                for j in range(i+1, self.sizeN):
                    if s[j] and self.adjMatrix[i][j]:
                        ind = False
                        break
                if ind:
                    fitness+=setSize
                else:
                    fitness-=1.5*setSize
                setSize+=1
        if fitness < 0: #i need the fitness to remain in the positives...
            fitness = -1.0/fitness
        return fitness
    
    def goodFitness(self, s):
        # Skip error test and assume len(s) == sizeN for quickness of algorithm
        setSize = fitness = 0.0
        valid = invalid = missingValids = 0.0
        for i in range(self.sizeN):
            if s[i] :
                ind = True
                for j in range(i+1, self.sizeN):
                    if s[j] and self.adjMatrix[i][j]:
                        ind = False
                        break
                if ind:
                    valid += 1
                else:
                    invalid += 1
                setSize+=1
            else:
                missingValids +=1
        return (( setSize )**2) / ( (invalid+1)**2)  # works at low numbers
        # return ( setSize ) / (invalid+1) # works at low numbers
        #return ( valid ) / (invalid+1)
     
    def connFitness(self, s):
        # Skip error test and assume len(s) == sizeN for quickness of algorithm
        setSize = conn =  0.0
        for i in range(self.sizeN):
            if s[i] :
                for j in range(i+1, self.sizeN):
                    if s[j] and self.adjMatrix[i][j]:
                        conn += 1
                setSize+=1
        return (( setSize )**2) / ( (conn+1)**2) 
    
    def evaluateSet(self, vset):
        """ Test to see if a passed set is independent, if yes, size of set is returned, -1 elsewise """
        # Skip error test and assume len(set) == sizeN for quickness of algorithm
        setSize = 0
        independent = True
        for i in range(self.sizeN):
            try:
                if vset[i] :
                    for j in range(i+1, self.sizeN):
                        if vset[j] and self.adjMatrix[i][j]:
                            independent = False
                            break
                    if not independent: 
                        break
                    setSize+=1
            except TypeError, e:
                print "EVALUATE SET BROKEN: {0}".format(e)
                # print "Graph: {0}".format(self)
                exit(1)
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
        """ Return an Bernoulli distributed random bool with given probability """
        num = self.uniform(0.0, 1.0)
        if(num <= probability):
            return True
        return False

if __name__=="__main__":
    import doctest
    doctest.testmod()
