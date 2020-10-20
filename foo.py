import mdp
m1 = mdp.makeRNProblem()
m1.valueIteration()
m1.printValues()

m2 = mdp.makeRNProblem()
m2.policyIteration()
m2.printActions()

