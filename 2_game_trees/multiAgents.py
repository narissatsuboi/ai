# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent


def manhattan_distance(startPos, endPos):
    start_x, start_y = startPos
    end_x, end_y = endPos
    return abs(start_x - end_x) + abs(start_y - end_y)


def manhattan_distance_nearest(start_pos, coordinates):
    md = float('inf')
    for coordinate in coordinates:
        md = min(md, manhattan_distance(start_pos, coordinate))
    return md


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()
        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if
                       scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        # newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"

        # def manhattan_distance(startPos, endPos):
        #     start_x, start_y = startPos
        #     end_x, end_y = endPos
        #     return abs(start_x - end_x) + abs(start_y - end_y)
        #
        # def manhattan_distance_nearest(start_pos, coordinates):
        #     md = float('inf')
        #     for coordinate in coordinates:
        #         md = min(md, manhattan_distance(start_pos, coordinate))
        #     return md

        fn = 0

        # distance to the nearest food pellet
        food_coordinates = currentGameState.getFood().asList()
        food_md_nearest = manhattan_distance_nearest(newPos, food_coordinates)
        pos_holds_food = food_md_nearest == 0
        # update fn based on nearest food
        if pos_holds_food:
            fn += 10
        else:
            fn += 1 / food_md_nearest

        # distance to the nearest ghost
        ghost_md_nearest = float('inf')
        ghost_nearest = None

        # find the nearest ghost and its md
        for state in newGhostStates:
            md = manhattan_distance(newPos, state.getPosition())
            if md < ghost_md_nearest:
                ghost_md_nearest = md
                ghost_nearest = state
        # update fn based on nearest ghost state
        if ghost_md_nearest <= 1 and ghost_nearest.scaredTimer == 0:
            fn -= 10  # run from adjacent ghosts
        elif ghost_md_nearest <= 5 and ghost_nearest.scaredTimer >= 5:
            fn += 10  # chase nearby scared ghost

        # distance to nearest capsule
        newCapsules = currentGameState.getCapsules()
        capsule_md_nearest = manhattan_distance_nearest(newPos, newCapsules)
        pos_holds_capsule = capsule_md_nearest == 0
        # update fn based on nearest capsule
        if pos_holds_capsule:
            fn += 10
        else:
            fn += (1 / capsule_md_nearest) * 3

        return fn


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

    def update_agent_idx(self, agent_idx, gamestate):
        if agent_idx + 1 == gamestate.getNumAgents():
            return 0
        return agent_idx + 1

    def terminal_state_test(self, agent_depth, gamestate):
        max_depth = agent_depth == self.depth
        gameOver = gamestate.isLose() or gamestate.isWin()
        return gameOver or max_depth


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gamestate):
        """
        """
        root_utilities = []
        root_actions = gamestate.getLegalActions(0)
        for root_action in root_actions:
            gamestate_next = gamestate.generateSuccessor(0, root_action)
            root_utilities.append(self.minimax(0, 0, gamestate_next))
        max_utility = max(root_utilities)
        max_index = root_utilities.index(max_utility)
        max_action = root_actions[max_index]
        return max_action

    def minimax(self, agent_idx, agent_depth, gamestate):

        agent_idx_next = self.update_agent_idx(agent_idx, gamestate)
        if agent_idx_next == 0:
            agent_depth += 1
            return self.maximizer(agent_idx_next, agent_depth, gamestate)
        return self.minimizer(agent_idx_next, agent_depth, gamestate)

    def minimizer(self, agent_idx, agent_depth, gamestate):
        if self.terminal_state_test(agent_depth, gamestate):
            return self.evaluationFunction(gamestate)

        minimizer_utilities = []
        minimizer_actions = gamestate.getLegalActions(agent_idx)
        for action in minimizer_actions:
            gamestate_next = gamestate.generateSuccessor(agent_idx, action)
            minimizer_utilities.append(self.minimax(agent_idx, agent_depth, gamestate_next))

        return min(minimizer_utilities)

    def maximizer(self, agent_idx, agent_depth, gamestate):
        if self.terminal_state_test(agent_depth, gamestate):
            return self.evaluationFunction(gamestate)

        maximizer_utilities = []
        maximizer_actions = gamestate.getLegalActions(agent_idx)
        for action in maximizer_actions:
            gamestate_next = gamestate.generateSuccessor(agent_idx, action)
            maximizer_utilities.append((self.minimax(agent_idx, agent_depth, gamestate_next)))

        return max(maximizer_utilities)


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        root_best_value = float('-inf')
        root_best_action = None

        root_actions = gameState.getLegalActions(0)
        for action in root_actions:
            gamestate_next = gameState.generateSuccessor(0, action)
            v = self.minimax(0, 0, gamestate_next, root_best_value, float('inf'))
            if v > root_best_value:
                root_best_value = v
                root_best_action = action
        return root_best_action

    def minimax(self, agent_idx, agent_depth, gamestate, alpha, beta):

        agent_idx_next = self.update_agent_idx(agent_idx, gamestate)
        if agent_idx_next == 0:
            agent_depth += 1
            return self.maximizer(agent_idx_next, agent_depth, gamestate, alpha, beta)
        return self.minimizer(agent_idx_next, agent_depth, gamestate, alpha, beta)

    def minimizer(self, agent_idx, agent_depth, gamestate, alpha, beta):
        if self.terminal_state_test(agent_depth, gamestate):
            return self.evaluationFunction(gamestate)

        v = float('inf')
        minimizer_actions = gamestate.getLegalActions(agent_idx)
        for action in minimizer_actions:
            gamestate_next = gamestate.generateSuccessor(agent_idx, action)
            v = min(v, self.minimax(agent_idx, agent_depth, gamestate_next, alpha, beta))

            if v < alpha:
                return v
            beta = min(beta, v)
        return v

    def maximizer(self, agent_idx, agent_depth, gamestate, alpha, beta):
        if self.terminal_state_test(agent_depth, gamestate):
            return self.evaluationFunction(gamestate)

        v = float('-inf')
        maximizer_actions = gamestate.getLegalActions(agent_idx)
        for action in maximizer_actions:
            gamestate_next = gamestate.generateSuccessor(agent_idx, action)
            v = max(v, self.minimax(agent_idx, agent_depth, gamestate_next, alpha, beta))
            if v > beta:
                return v
            alpha = max(alpha, v)

        return v


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        root_utilities = []
        root_actions = gameState.getLegalActions(0)
        for root_action in root_actions:
            gamestate_next = gameState.generateSuccessor(0, root_action)
            root_utilities.append(self.expectimax(0, 0, gamestate_next))
        max_utility = max(root_utilities)
        max_index = root_utilities.index(max_utility)
        max_action = root_actions[max_index]
        return max_action

    def expectimax(self, agent_idx, agent_depth, gamestate):
        agent_idx_next = self.update_agent_idx(agent_idx, gamestate)
        if agent_idx_next == 0:
            agent_depth += 1
            return self.maximizer(agent_idx_next, agent_depth, gamestate)
        return self.expectimizer(agent_idx_next, agent_depth, gamestate)

    def maximizer(self, agent_idx, agent_depth, gamestate):
        if self.terminal_state_test(agent_depth, gamestate):
            return self.evaluationFunction(gamestate)

        maximizer_utilities = []
        maximizer_actions = gamestate.getLegalActions(agent_idx)
        for action in maximizer_actions:
            gamestate_next = gamestate.generateSuccessor(agent_idx, action)
            maximizer_utilities.append((self.expectimax(agent_idx, agent_depth, gamestate_next)))

        return max(maximizer_utilities)

    def expectimizer(self, agent_idx, agent_depth, gamestate):
        if self.terminal_state_test(agent_depth, gamestate):
            return self.evaluationFunction(gamestate)

        expectimizer_utilities = []
        minimizer_actions = gamestate.getLegalActions(agent_idx)
        for action in minimizer_actions:
            gamestate_next = gamestate.generateSuccessor(agent_idx, action)
            expectimizer_utilities.append(self.expectimax(agent_idx, agent_depth, gamestate_next))

        expectimax_value = sum(expectimizer_utilities) / (1.0 * len(expectimizer_utilities))
        return expectimax_value


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """

    def closest_food_feature(ghost_flag, importance_factor):
        factor = 100 * importance_factor

        if not food_coordinates:
            return 0  # no more food

        if ghost_flag:
            return 1 / float('inf')  # run for your life!

        distance = manhattan_distance_nearest(current_position, food_coordinates)
        return factor * (1 / distance)

    def nearest_ghost():
        # distance to the nearest ghost
        ghost_md_nearest = float('inf')
        ghost = None

        # find the nearest ghost
        for state in ghost_states:
            md = manhattan_distance(current_position, state.getPosition())
            if md < ghost_md_nearest:
                ghost_md_nearest = md
                ghost = state
        return ghost

    def ghost_too_close():
        distance = manhattan_distance(current_position, ghost_nearest.getPosition())
        if distance <= 1 and ghost_nearest.scaredTimer == 0:
            return True
        return False

    def food_feature(importance_factor):
        factor = current_score * importance_factor
        NO_FOOD_BONUS = 10
        if not food_count:
            return NO_FOOD_BONUS
        return factor * (-1.0 / food_count)

    def capsule_feature():
        if not capsule_count:
            return 0
        return -1.0 / capsule_count

    def ghost_timer_feature():
        if not ghost_timer:
            return 0
        return 50

    current_position = currentGameState.getPacmanPosition()
    current_score = currentGameState.getScore()

    food_coordinates = currentGameState.getFood().asList()
    food_count = currentGameState.getNumFood()
    capsule_count = len(currentGameState.getCapsules())

    ghost_states = currentGameState.getGhostStates()
    ghost_nearest = nearest_ghost()
    ghost_timer = ghost_nearest.scaredTimer
    is_ghost_too_close = ghost_too_close()

    score_feature = current_score

    eval_value = 0
    eval_value += score_feature
    eval_value += closest_food_feature(is_ghost_too_close, .08)
    eval_value += food_feature(0.0025)
    eval_value += capsule_feature()
    eval_value += ghost_timer_feature()

    return eval_value


# Abbreviation
better = betterEvaluationFunction
