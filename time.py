#!/usr/bin/python

import datetime
import ivspSolver
from ivspSolver import *

for i in range(1, 24):
    before = datetime.datetime.now()
    
    g = Graph(i, .15)
    vs = g.exhaustiveSolution()
    eScore = g.evaluateSet(vs)
    
    after = datetime.datetime.now()
    print "V Size: ", i, "score:", eScore, "in", str(after-before)