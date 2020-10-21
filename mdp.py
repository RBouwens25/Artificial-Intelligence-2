### code for representing/solving an MDP
##Benjamin Dinh         s3427145
## Rachelle Bouwens     s3661393
## Janneke van Hulten   s3658384

import random
import numpy
from problem_utils import *
from operator import itemgetter
numpy.warnings.filterwarnings('ignore')


class State:

    def __init__(self):
        self.utility = 0.0
        self.reward = 0.0
        ### an action maps to a list of probability/state pairs
        self.transitions = {}
        self.actions = []
        self.policy = None
        self.coords = 0
        self.isGoal = False
        self.isWall = False
        self.id = 0

    def computeEU(self, action):
        return sum([trans[0] * trans[1].utility \
                    for trans in self.transitions[action]])

    def selectBestAction(self):
        best = max([(self.computeEU(a), a) for a in self.actions])
        return best[1]


class Map:
    def __init__(self):
        self.states = {}
        self.stop_crit = 0.01
        self.gamma = 0.8
        self.n_rows = 0
        self.n_cols = 0

    class PrintType:
        ACTIONS = 0
        VALUES = 1

    ### you write this method
    def valueIteration(self):
        for p in self.states.values():
            ## 'Set each non-goal state’s utility to 0.'
            if not p.isGoal:
                p.utility = 0.0

        ## initialize list which will hold all the differences in utilities
        util_dif = []

        while(1):
            for i in self.states.values():
                current_state = i
                ## current state is not a goal state
                if not current_state.isGoal:
                    max1 = []
                    for a in current_state.actions:
                        ## find transitions of actions from current state
                        trans = current_state.transitions[a]
                        sum = 0
                        ## loop through transitions, which consist of p and the state it will go to
                        for x in trans:
                            tr, st = x
                            ## add the multiplication of that probability and its utility to the sum
                            sum = sum + tr * st.utility
                        ## add the sum for all the different transitions for the one action to max1
                        max1.append(sum)
                    ## get the maximal sum, which indicates which of the four actions is best
                    max_ult = max(max1)
                    ## find the new utility of the current state, using the max_ult
                    util = current_state.reward + self.gamma * max_ult
                    ## find the difference between the old utility and the new utility
                    dif = abs(util - current_state.utility)
                    ## add this difference to util_dif
                    util_dif.append(dif)
                    ## set the utility of the current_state to the newly found utility
                    current_state.utility = util
            ## find the maximal difference of all the utilities
            max_dif = max(util_dif)
            ## if this maximal difference is smaller than the stop criterium, break
            if (max_dif < self.stop_crit):
                break
            util_dif = []



    ### you write this method
    def policyIteration(self):
        for p in self.states.values():
            if not p.isGoal:
                ## "Initialize the policy by chosing a random action to be performed in each non-goal state"
                p.policy = random.choice(p.actions)

        while(1):
            ## "Calculate the utility estimates of all non-goal states under the current policy "
            self.calculateUtilitiesLinear()
            ## Use boolean change to check if policy has been changed when iterated over all the states
            change = False
            for i in self.states.values():
                if not i.isGoal:
                    max1 = []
                    ## get transitions of current policy of current state
                    trans2 = i.transitions[i.policy]
                    sum2 = 0
                    ## loop over those transitions and find the sum of all multiplications of probabilities and utilities of surrounding states
                    for y in trans2:
                        tr2, st2 = y
                        sum2 = sum2 + tr2 * st2.utility
                    ## loop over all actions to find the maximum sum, and its corresponding action
                    for a in i.actions:
                        ## find transitions of actions from current state
                        trans = i.transitions[a]
                        sum = 0
                        ## loop through transitions, which consist of p and the state it will go to
                        for x in trans:
                            tr, st = x
                            ## add the multiplication of that probability and its utility to the sum
                            sum = sum + tr * st.utility
                        ## add that action and sum to max1
                        max1.append((a,sum))
                    max_ult = max(max1, key=itemgetter(1))[1]
                    ## find if newly found max sum is bigger than sum using the action from the orinigal
                    if max_ult > sum2:
                        ## if so, set policy to this new action
                        i.policy = max(max1, key=itemgetter(1))[0]
                        ## set change to True, so the while-loop continues
                        change = True
            ## if the policy didn't change at all, break from loop
            if change == False:
                break


    def calculateUtilitiesLinear(self):
        n_states = len(self.states)
        coeffs = numpy.zeros((n_states, n_states))
        ordinate = numpy.zeros((n_states, 1))
        for s in self.states.values():
            row = s.id
            ordinate[row, 0] = s.reward
            coeffs[row, row] += 1.0
            if not s.isGoal:
                probs = s.transitions[s.policy]
                for p in probs:
                    col = p[1].id
                    coeffs[row, col] += -self.gamma * p[0]
        solution, _, _, _ = numpy.linalg.lstsq(coeffs, ordinate)
        for s in self.states.values():
            if not s.isGoal:
                s.utility = solution[s.id, 0]

    def printActions(self):
        self.printMaze(self.PrintType.ACTIONS)

    def printValues(self):
        self.printMaze(self.PrintType.VALUES)

    def printMaze(self, print_type):
        to_print = ":"
        for c in range(self.n_cols):
            to_print = to_print + "--------:"
        to_print = to_print + '\n'
        for r in range(self.n_rows):
            to_print = to_print + "|"
            for c in range(self.n_cols):
                if self.states[(c, r)].isWall:
                    to_print = to_print + "        "
                else:
                    to_print = to_print + ' '
                    if self.states[(c, r)].isGoal:
                        to_print = to_print + \
                                   "  {0: d}  ".format(int(self.states[(c, r)].utility))
                    else:
                        if print_type == self.PrintType.VALUES:
                            to_print = to_print + \
                                       "{0: .3f}".format(self.states[(c, r)].utility)
                        elif print_type == self.PrintType.ACTIONS:
                            a = self.states[(c, r)].selectBestAction()
                            to_print = to_print + "  "
                            if a == 'left':
                                to_print = to_print + "<<"
                            elif a == 'right':
                                to_print = to_print + ">>"
                            elif a == 'up':
                                to_print = to_print + "/\\"
                            elif a == 'down':
                                to_print = to_print + "\\/"
                            to_print = to_print + "  "
                    to_print = to_print + ' '
                to_print = to_print + "|"
            to_print = to_print + '\n'
            to_print = to_print + ":"
            for c in range(self.n_cols):
                to_print = to_print + "--------:"
            to_print = to_print + '\n'
        print(to_print)


def makeRNProblem():
    """
    Creates the maze defined in Russell & Norvig. Utilizes functions defined
    in the problem_utils module.
    """

    walls = [(1, 1)]
    actions = ['left', 'right', 'up', 'down']
    cols = 4
    rows = 3

    def filterState(oldState, newState):
        if (newState[0] < 0 or newState[1] < 0 or newState[0] > cols - 1 or
                newState[1] > rows - 1 or newState in walls):
            return oldState
        else:
            return newState

    m = Map()
    m.n_cols = cols
    m.n_rows = rows
    for i in range(m.n_cols):
        for j in range(m.n_rows):
            m.states[(i, j)] = State()
            m.states[(i, j)].coords = (i, j)
            m.states[(i, j)].isGoal = False
            m.states[(i, j)].actions = actions
            m.states[(i, j)].id = j * m.n_cols + i
            m.states[(i, j)].reward = -0.04

    m.states[(3, 0)].isGoal = True
    m.states[(3, 1)].isGoal = True

    m.states[(3, 0)].utility = 1.0
    m.states[(3, 1)].utility = -1.0

    m.states[(3, 0)].reward = 1.0
    m.states[(3, 1)].reward = -1.0

    for t in walls:
        m.states[t].isGoal = True
        m.states[t].isWall = True
        m.states[t].reward = 0.0
        m.states[t].utility = 0.0

    for s in m.states.items():
        for a in actions:
            s[1].transitions[a] = [ \
                (0.8, m.states[filterState(s[0], getSuccessor(s[0], a))]),
                (0.1, m.states[filterState(s[0], getSuccessor(s[0], left(a)))]),
                (0.1, m.states[filterState(s[0], getSuccessor(s[0], right(a)))])]
    return m


def make2DProblem():
    """
    Creates the larger maze described in the exercise. Utilizes functions 
    defined in the problem_utils module.
    """

    walls = [(1, 1), (4, 1), (5, 1), (6, 1), (7, 1), (1, 2), (7, 2), (1, 3), (5, 3),
             (7, 3), (1, 4), (5, 4), (7, 4), (1, 5), (5, 5), (7, 5), (1, 6), (5, 6),
             (7, 6), (1, 7), (5, 7), (7, 7), (1, 8), (3, 8), (4, 8), (5, 8),
             (7, 8), (1, 9)]
    actions = ['left', 'right', 'up', 'down']

    def filterState(oldState, newState):
        if (newState[0] < 0 or newState[1] < 0 or newState[0] > 9 or
                newState[1] > 9 or newState in walls):
            return oldState
        else:
            return newState

    m = Map()
    m.n_cols = 10
    m.n_rows = 10
    for i in range(m.n_cols):
        for j in range(m.n_rows):
            m.states[(i, j)] = State()
            m.states[(i, j)].coords = (i, j)
            m.states[(i, j)].isGoal = False
            m.states[(i, j)].actions = actions
            m.states[(i, j)].id = j * 10 + i
            m.states[(i, j)].reward = -0.04

    m.states[(0, 9)].isGoal = True
    m.states[(9, 9)].isGoal = True
    m.states[(9, 0)].isGoal = True

    m.states[(0, 9)].utility = 1.0
    m.states[(9, 9)].utility = -1.0
    m.states[(9, 0)].utility = 1.0

    m.states[(0, 9)].reward = 1.0
    m.states[(9, 9)].reward = -1.0
    m.states[(9, 0)].reward = 1.0

    for t in walls:
        m.states[t].isGoal = True
        m.states[t].isWall = True
        m.states[t].utility = 0.0
        m.states[t].reward = 0.0

    for s in m.states.items():
        for a in actions:
            s[1].transitions[a] = [ \
                (0.7, m.states[filterState(s[0], getSuccessor(s[0], a))]),
                (0.1, m.states[filterState(s[0], getSuccessor(s[0], opposite(a)))]),
                (0.1, m.states[filterState(s[0], getSuccessor(s[0], left(a)))]),
                (0.1, m.states[filterState(s[0], getSuccessor(s[0], right(a)))])]

    return m
