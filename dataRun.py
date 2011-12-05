#!/usr/bin/python

import datetime
import ivspSolver
from ivspSolver import *

grlog = galog = anlog = comblog = ""
for cnn in [.05, .1, .15, .2]:
    g = Graph(100, cnn)
    #g.fitfunc = g.tobyFitness
    
    # Greedy 
    before = datetime.datetime.now()
    vs = g.greedySolution()
    gScore = g.evaluateSet(vs)
    after = datetime.datetime.now()
    grlog += str(cnn) + "\t" + str(gScore) + "\t" + str((after-before).total_seconds()) + "\n"
    
    # GA 
    before = datetime.datetime.now()
    vs = g.GASolution(popsize=50, generations=500, mutation=.02)
    gaScore = g.evaluateSet(vs)
    after = datetime.datetime.now()
    galog += str(cnn) + "\t" + str(gaScore) + "\t" + str((after-before).total_seconds()) + "\n"
    
    # Annealing 
    before = datetime.datetime.now()
    vs = g.shallowAnnealing(coolStep=1)
    aScore = g.evaluateSet(vs)
    after = datetime.datetime.now()
    anlog += str(cnn) + "\t" + str(aScore) + "\t" + str((after-before).total_seconds()) + "\n"
    
    comblog += str(cnn) + "\t" + str(gScore) + "\t" + str(gaScore) + "\t" + str(aScore) + "\n"

f = open("greedy.log", "w")
f.write( grlog )
f.close()

f = open("ga.log", "w")
f.write( galog )
f.close()

f = open("an.log", "w")
f.write( anlog )
f.close()

f = open("comb.log", "w")
f.write( comblog )
f.close()