# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]


def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    # Stack implementation with visited set optimization
    # node structure ([x, y], ['... Action'])

    # SEED START STATE
    frontierStack = util.Stack()
    frontierStack.push((problem.getStartState(), []))
    explored = set()

    while not frontierStack.isEmpty():
        # HANDLE CURRENT STATE
        position, path = frontierStack.pop()
        explored.add(position)

        # GOAL TEST
        if problem.isGoalState(position):
            return path  # SOLUTION

        # EXPAND FRINGE  # new fringe is path + [action] the path the next node at the end
        successorStates = problem.getSuccessors(position)
        for successorPosition, action, _ in successorStates:
            if successorPosition not in explored:
                frontierStack.push((successorPosition, path + [action]))

    return []  # RETURN FAILURE


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    # Queue implementation with visited set optimization
    # node structure ([x, y], ['... Action'])

    # SEED START STATE
    # frontierQueue = util.Queue()
    # root = (problem.getStartState(), [], 0)
    # frontierQueue.push(root)
    # explored = set()
    # generated = set()
    #
    # while not frontierQueue.isEmpty():
    #     # HANDLE CURRENT STATE
    #     state, path, currCost = frontierQueue.pop()
    #     explored.add(state)
    #
    #     # GOAL TEST
    #     if problem.isGoalState(state):
    #         return path  # SOLUTION
    #
    #     # EXPAND FRINGE
    #     successorStates = problem.getSuccessors(state)
    #     for successorPosition, action, cost in successorStates:
    #         if successorPosition not in explored and successorPosition not in generated:
    #             generated.add(successorPosition)
    #             frontierQueue.push((successorPosition, path + [action], cost))
    #
    # return []  # RETURN FAILURE

    frontierQueue = util.Queue()
    root = (problem.getStartState(), [], 0)
    frontierQueue.push(root)
    # explored = set()
    explored = []

    while not frontierQueue.isEmpty():
        # HANDLE CURRENT STATE
        state, path, currCost = frontierQueue.pop()

        if state not in explored:
            # explored.add(state)
            explored.append(state)

            # GOAL TEST
            if problem.isGoalState(state):
                return path  # SOLUTION

            # EXPAND FRINGE
            successorStates = problem.getSuccessors(state)
            for state, action, cost in successorStates:
                if state not in explored:
                    frontierQueue.push((state, path + [action], cost))

    return []  # RETURN FAILURE


def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    # node structure ((item), priority)) => ((position, path), priority))

    # SEED START STATE
    frontierPQ = util.PriorityQueue()
    start = problem.getStartState()
    frontierPQ.push((start, []), problem.getCostOfActions([]))
    explored = set()
    generated = set(start)

    while not frontierPQ.isEmpty():
        # HANDLE CURRENT STATE
        position, path = frontierPQ.pop()

        if position not in explored:
            explored.add(position)

            # GOAL TEST
            if problem.isGoalState(position):
                return path

            # EXPAND FRINGE
            successorStates = problem.getSuccessors(position)
            for successorPosition, action, _ in successorStates:
                if successorPosition not in explored and successorPosition not in generated:
                    generated.add(successorPosition)
                    frontierPQ.push((successorPosition, path + [action]),
                                    problem.getCostOfActions(path + [action]))
                elif successorPosition in generated:
                    frontierPQ.update((successorPosition, path + [action]),
                                      problem.getCostOfActions(path + [action]))
    return []  # RETURN FAILURE


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    frontierPQ = util.PriorityQueue()
    start = problem.getStartState()
    fn, hn = problem.getCostOfActions([]), heuristic(start, problem)
    an = fn + hn
    frontierPQ.push((start, [], an), an)
    print("start state: ", start)
    explored = dict()

    while not frontierPQ.isEmpty():
        # HANDLE CURRENT STATE
        state, path, an = frontierPQ.pop()

        if state not in explored:
            explored[state] = an

            # GOAL TEST
            if problem.isGoalState(state):
                return path

            # EXPAND FRINGE
            successorStates = problem.getSuccessors(state)
            for state, action, _ in successorStates:
                fn = problem.getCostOfActions(path + [action])
                hn = heuristic(state, problem)
                an = fn + hn
                successor = (state, path + [action], an)

                if state not in explored:
                    frontierPQ.push(successor, an)
                elif state in explored and an < explored[state]:
                    explored[state] = an
                    frontierPQ.update(successor, an)
                elif state in explored and an >= explored[state]:
                    continue

    return []  # RETURN FAILURE


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
