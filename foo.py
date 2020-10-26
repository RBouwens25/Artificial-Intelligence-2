import mdp
m1 = mdp.makeRNProblem()
m1.valueIteration()
m1.printValues()

m2 = mdp.makeRNProblem()
m2.policyIteration()
m2.printActions()

m3 = mdp.make2DProblem()
m3.valueIteration()
m3.printValues()

m4 = mdp.make2DProblem()
m4.policyIteration()
m4.printActions()