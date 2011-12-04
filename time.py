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
    print "Exhaustive score:".rjust(20), str(eScore).rjust(3), " in", str(after-before)
    
    
    # Greedy 
    before = datetime.datetime.now()
    
    vs = g.greedySolution()
    gScore = g.evaluateSet(vs)
    
    after = datetime.datetime.now()
    print "Greedy score:".rjust(20), str(gScore).rjust(3), " in", str(after-before)
    
    
    # GA 
    before = datetime.datetime.now()
    
    vs = g.GASolution(fitFunc=g.triangleFitness)
    gaScore = g.evaluateSet(vs)
    
    after = datetime.datetime.now()
    print "GA score:".rjust(20), str(gaScore).rjust(3), " in", str(after-before)
    
    before = datetime.datetime.now()

    # Annealing 
    vs = g.annealedSolution()
    aScore = g.evaluateSet(vs)
    
    after = datetime.datetime.now()
    print "Anlealing score:".rjust(20), str(aScore).rjust(3), " in", str(after-before)
    print 
