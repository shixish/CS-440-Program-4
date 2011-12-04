#!/usr/bin/python

import datetime
import ivspSolver
from ivspSolver import *

for i in range(2, 27):
    print "Vertex Set Size: ", i
    g = Graph(i, .15)
    
    # Exhaustive 
    before = datetime.datetime.now()

    vs = g.exhaustiveSolution()
    eScore = g.evaluateSet(vs)
    
    after = datetime.datetime.now()
    print "Exhaustive score:", eScore, "in", str(after-before)
    
    
    # Greedy 
    before = datetime.datetime.now()
    
    vs = g.greedySolution()
    gScore = g.evaluateSet(vs)
    
    after = datetime.datetime.now()
    print "Greedy score:", gScore, "in", str(after-before)
    
    
    # GA 
    before = datetime.datetime.now()
    
    vs = g.GASolution(fitFunc=g.triangleFitness)
    gaScore = g.evaluateSet(vs)
    
    after = datetime.datetime.now()
    print "GA score:", gaScore, "in", str(after-before)
    
    # Annealing 
    vs = g.annealedSolution()
    aScore = g.evaluateSet(vs)
    
    after = datetime.datetime.now()
    print "Anlealing score:", aScore, "in", str(after-before)
    print 