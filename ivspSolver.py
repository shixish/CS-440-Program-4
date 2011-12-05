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
        self.set = [i for i in set]
        self.fitness = -1.0
    
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
        return "Vertex set: %s\nFitness: %.3f"%(self.set, self.fitness)
        
    def __getitem__(self, key):
        return self.set[key]
    
    def __setitem__(self, key, value):
        self.set[key] = value
    
    @classmethod
    def emptySet(cls, size):
        """ Returns an empty set (all False valued) of [size] """
        return cls( [False for i in range(size)] )
    
    @classmethod
    def randomSet(cls, size, random, density=.2):
        """ Generates a random set, where individual vertex inclusion has [density] probability """
        return cls( [random.boolBernoulli(density) for i in range(size)] )


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
    
    def __init__(self, size, cnn, seed=0, testcase=False):
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
        >>> g = Graph(4,1,1,True) # test case graph generation
        
        >>> g
        Adjacency Matrix: 
        0    0    1    0    
        <BLANKLINE>
        0    0    1    0    
        <BLANKLINE>
        1    1    0    1    
        <BLANKLINE>
        0    0    1    0    
        <BLANKLINE>
        <BLANKLINE>
        
        
        """
        # Perform error checking
        if testcase == True:
            self.adjMatrix = [ [ False for j in range(4) ] for i in range(4) ]
            self.adjMatrix[0][0] = False 
            self.adjMatrix[0][1] = False
            self.adjMatrix[0][2] = True
            self.adjMatrix[0][3] = False
            self.adjMatrix[1][0] = False
            self.adjMatrix[1][1] = False
            self.adjMatrix[1][2] = True
            self.adjMatrix[1][3] = False
            self.adjMatrix[2][0] = True
            self.adjMatrix[2][1] = True
            self.adjMatrix[2][2] = False
            self.adjMatrix[2][3] = True
            self.adjMatrix[3][0] = False
            self.adjMatrix[3][1] = False
            self.adjMatrix[3][2] = True
            self.adjMatrix[3][3] = False
            self.rand = Fxrandom(seed)
            self.sizeN = 4
            
        # Perform error checking
        if size <= 0:
            raise ValueError("Graph Init: size must be > 0.")
        if cnn < 0.0 or cnn > 1.0:
            raise ValueError("Graph Init: connectivity must be between 0.0 & 1.0.")
        if seed < 0:
            raise ValueError("Graph Init: seed must be >= 0.")
        
        # Define class variables
        if not testcase:
            self.cnn = cnn
            self.sizeN = size
            self.rand = Fxrandom(seed)
            # print "Initial seed: %s"%(self.rand.seed) #This can be useful if we want to be able reproduce the same results.
        
        # Instantiate the class's adjacency matrix to all False values
        if not testcase:
            self.adjMatrix = [ [False for j in range(self.sizeN) ] for i in range(self.sizeN) ]
        
        # Initialize edges according to a weighted coin flip.
        # if cnn = 0.1, there will be a 10% chance of creating an edge
        # between vertices i and j.  Note that there will never be an
        # edge between a vertex and itself.
        if not testcase:
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
        """ 
        Finds an independent set using a greedy solution 
        
        >>> g = Graph(4,1,1,True)

        >>> g.greedySolution()
        Vertex set: [True, True, False, True]
        Fitness: 9.000
        """
        # Initialize an empty set
        vs = VSet.emptySet(self.sizeN)
        
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


    def annealedSolution(self):
        """ Generate the biggest set using the simulated annealing algorithm 
        -- Start at some initial "temperature" T
        -- Define a "cooling schedule T(x)
        -- Define an energy function Energy(set)
        -- Define a current_set initial state (vertex set)

        Pseudocode:

        while (not_converged):
            new_set = (random)
            Delta_s = Energy(new_set) - Energy(current_set)
            if (Delta_s < 0):
                current_set = new_set
            else with probability P=e^(-Delta_s/T):
                current_set = new_set
            T = alpha T
               
        >>> g = Graph(4,1,1,True)

        >>> g.annealSolution()
        Vertex set: [True, True, False, True]
        Fitness: 9.000

        """
        # initialize the empty set
        loud = False # set to True to see debugging output
        vs = VSet.randomSet(self.sizeN, self.rand)
        new_set = VSet(vs)
        not_converged = True
        T = self.sizeN # temperature
        i = 1 # iteration count
        if T < 1000:
            T = 1000
        while (not_converged):
            if loud: print "(1) Iteration: {0}".format(i)
            if loud: time.sleep(1)
            new_set = VSet(vs.set)
            if loud:
                print "(2) Current set: {0}".format(vs)
                print "(3) new_set: {0}".format(new_set)
            tog = random.randrange(self.sizeN)
            if loud: 
                print "(4) Toggling new_set vertex: {0}".format(tog)
                print "(5) new_set now {0}".format(new_set)
            new_set.toggleVertex(tog)
            if loud: print "(6) Setting fitness values..."
            self.setFitness(new_set)
            if loud: print "(7) new_set done..."
            self.setFitness(vs)
            if loud:
                print "(8) old set done..."
                time.sleep(1)
                print "(9) new set fitness: {0}".format(new_set.fitness)
                print "(10) current fitness: {0}".format(vs.fitness)
            Delta_s = new_set.fitness - vs.fitness
            if loud: print "(11) Delta_s set to {0}".format(Delta_s)
            if (Delta_s < 0):
                vs = new_set
            P = math.e**(-Delta_s/T)
            if (P > 0.5): 
                vs = new_set
            T -= 1
            if (T == 0):
                if loud: print "(**) Set has converged."
                not_converged = False   
            i += 1
            if loud: time.sleep(0.5)
        return vs
    
    def shallowAnnealing(self):
        """ Generate the biggest set using the simulated annealing algorithm 
        -- Start at some initial "temperature" T
        -- Define a "cooling schedule T(x)
        -- Define an energy function Energy(set)
        -- Define a current_set initial state (vertex set)

        Pseudocode:

        while (not_converged):
            new_set = (random)
            Delta_s = Energy(new_set) - Energy(current_set)
            if (Delta_s < 0):
                current_set = new_set
            else with probability P=e^(-Delta_s/T):
                current_set = new_set
            T = alpha T
               
        >>> g = Graph(4,1,1,True)

        >>> g.annealSolution()
        Vertex set: [True, True, False, True]
        Fitness: 9.000

        """
        bestScore = 0
        for j in range(self.sizeN*2):
            T = self.sizeN
            curSet = VSet.randomSet(self.sizeN, self.rand)
            
            not_converged = True
            while (not_converged):
                newSet = VSet( curSet.set )
                newSet.toggleVertex( random.randrange(self.sizeN) )
                Delta_s = self.triangleFitness(curSet) - self.triangleFitness(newSet)
                if (Delta_s < 0):
                    curSet = newSet
                P = math.e**(-Delta_s/T)
                if (self.rand.boolBernoulli( P )): 
                    curSet = newSet
                T -= .1
                if T <= 1.0:
                    not_converged = False
            endScore = self.triangleFitness( curSet )
            if endScore > bestScore:
                bestSet = VSet( curSet )
                bestScore = endScore
        return bestSet
    
    def exhaustiveSolution(self):
        """ Generate the biggest possible independent set of vertices by testing all possibilities 
        
        >>> g = Graph(4,1,1,True)

        >>> g.exhaustiveSolution()
        Vertex set: [True, True, False, True]
        Fitness: 9.000
        """
        maxScore = 0
        for s in VSetIter( self.sizeN ):
            curScore = self.evaluateSet( s )
            if curScore > maxScore:
                maxScore = curScore
                maxSet = VSet( s )
        return maxSet
    
    def rouletteSelection(self, popsel, popnumber):
        """    Stochastic Sampling (Roulette wheel) method of selecting parents
            This method requires some extra preperation of the data, and fails when fitness is negative...
            Source: http://www.cse.unr.edu/~banerjee/selection.htm """
        choice = [ ]
        k=0
        while (k < popnumber):
            partsum = 0
            parent = 0
            randnum = self.rand.uniform(0.0, 1.0)
            for i in range(0, popnumber):
                partsum += popsel[i].rank
                if partsum >= randnum:
                    parent = i
                    break
            k+=1
            choice.append(parent)
        return choice
        
    def mutate(self, value, rate=None):
        if rate == None:
            return value
        return not value if self.rand.boolBernoulli(rate) else value
    
    def combine(self, group1, group2, mutation):
        return VSet( [self.mutate(group1[i] if self.rand.boolBernoulli(.5) else group2[i], mutation) for i,v in enumerate(group1)] )
    
    def GASolution(self, popsize=100, generations=50, density=None, mutation=None, preserve=0, fitFunc=None):
        """ Finds an independent set using a Genetic Algorithm """
        if fitFunc == None:
            fitFunc = self.setFitness
        if not density:
            density = self.greedySolution().density()*.25
            # print "Using density: %.5f"%density
        population = []
        total = 0.0
        #initialize the population
        for i in range(popsize):
            s = VSet.randomSet(self.sizeN, self.rand, density)
            s.fitness = fitFunc(s)
            total += s.fitness
            population.append(s)
        
        if total == 0:
            print "Failure! Total fitness is zero... Something went wrong here."
            return
        
        avg = total/popsize
        # print "Average before: %.2f"%(avg)
        # sys.stdout.write("Epoc: ")
        for g in range(generations):
            
            for p in population:
                p.rank = p.fitness/total #produces a percentage for weighting the roulette...
                
            if preserve > 0:
                population = sorted(population, key=lambda s: s.fitness, reverse=True)
            
            randpop = popsize-preserve
            males = self.rouletteSelection(population, randpop)
            females = self.rouletteSelection(population, randpop)
            
            total = 0.0
            for i in range(randpop):
                s = self.combine(population[males[i]], population[females[i]], mutation=mutation)
                s.fitness = fitFunc(s)
                total += s.fitness
                population[preserve+i] = s
        
            #population = [VSet.randomSet(self.sizeN, self.rand) for i in range(popsize)]
            #for i, s in enumerate(siblings):
            #    s.rank = self.evaluateSet(s)
            
            # sys.stdout.write("%i "%g)
            # sys.stdout.flush()
        # sys.stdout.write("\n")
        
        #for i,s in enumerate(population):
        #    print "%i: %.8f"%(i,s.rank)
            #print "%i: %.1f %s"%(i,s.rank, s)
        best = "No valid solution found!"
        sorted_population = sorted(population, key=lambda s: s.fitness, reverse=True)
        for t in sorted(population, key=lambda s: s.fitness, reverse=True):
            if self.evaluateSet(t) > 0:
                best = t
        #best = max(population, key=lambda s:s.fitness)
        #print "Average fitness before: %.2f"%(avg)
        #print "Average fitness after: %.2f"%(total/popsize)
        #print "Highest fitness: %.2f"%(sorted_population[0].fitness)
        #print "Lowest fitness: %.2f"%(sorted_population[-1].fitness)
        #print "Best: %s"%best
        return best
    
    def setFitness(self, vset_obj):
        """ Test the fitness of a passed set: fitness= [set size]^2 - [connections]^2 
        >>> g = Graph(4,1,1,True)

        >>> vs = VSet([1,1,0,1])
        something_happened!

        """
        # Skip error test and assume len(set) == sizeN for quickness of algorithm
        setSize = connections = 0
        for i in range(self.sizeN):
            if vset_obj[i] :
                for j in range(i+1, self.sizeN):
                    if vset_obj[j] and self.adjMatrix[i][j]:
                        connections+=1
                setSize+=1
        fitness = float(setSize*setSize)-(connections*connections)
        if fitness < 0: #i need the fitness to remain in the positives...
            fitness = -1/fitness
        vset_obj.fitness = fitness
        return fitness
    
    def setFitness4(self, set):
        """ Test the fitness of a passed set: fitness= [set size] - 4*[connections] 
        
        These are just fake values for what maybe we might want
        
        Algorithm idea: fitness could start at 1 (empty set) then
        1st unconnected: +2 (3)
        2nd: +3 (6)
        3rd: +4 (10)
        4th: +5 (15) etc...
        and when the member is found to be connected, score penalty of Nth + (Nth * 1.5)


        >>> g = Graph(4,1,1,True)

        >>> g.setFitness4([1,1,0,1]) # "perfect" score: so fit, so clean
        10

        >>> g.setFitness4([1,1,1,1]) # one-too-many connected (nearly fit)
        8

        >>> g.setFitness4([1,0,0,1]) # could be better
        6

        >>> g.setFitness4([1,1,0,0]) # same as above
        6

        >>> g.setFitness4([0,0,0,1]) # so wrong
        3

        >>> g.setFitness4([0,0,0,0]) # wtf
        1

        """
        # Skip error test and assume len(set) == sizeN for quickness of algorithm
        setSize = connections = 0
        for i in range(self.sizeN):
            if set[i] :
                for j in range(i+1, self.sizeN):
                    if set[j] and self.adjMatrix[i][j]:
                        connections+=1
                setSize+=1
        fitness = float(setSize)-(4*connections)
        if fitness < 0: #i need the fitness to remain in the positives...
            fitness = -1/fitness
        return fitness
    
    def triangleFitness(self, set):
        """ Fitness= for i vertexes: sum( [tri] || - 1.5*[tri]).  [tri] = triangle number of ith vertex """
        # Skip error test and assume len(set) == sizeN for quickness of algorithm
        setSize = fitness = 0
        ind = True
        for i in range(self.sizeN):
            if set[i] :
                for j in range(i+1, self.sizeN):
                    if set[j] and self.adjMatrix[i][j]:
                        ind = False
                        break
                if ind:
                    fitness+=setSize
                else:
                    fitness-=1.5*setSize
                setSize+=1
        if fitness < 0: #i need the fitness to remain in the positives...
            fitness = -1/fitness
        return fitness
    
    def evaluateSet(self, vset):
        """ Test to see if a passed set is independent, if yes, size of set is returned, -1 elsewise """
        # Skip error test and assume len(set) == sizeN for quickness of algorithm
        setSize = 0
        independent = True
        for i in range(self.sizeN):
            if vset[i] :
                for j in range(i+1, self.sizeN):
                    if vset[j] and self.adjMatrix[i][j]:
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

if __name__=="__main__":
    import doctest
    doctest.testmod()
