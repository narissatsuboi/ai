# valueIterationAgents.py
# -----------------------
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
from collections import defaultdict

# valueIterationAgents.py
# -----------------------
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


import util

from learningAgents import ValueEstimationAgent


class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """

    def __init__(self, mdp, discount=0.9, iterations=100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter()  # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        """
        Follow Sutton & Barton batch version of value iteration per project specification!!!
        https://lcalem.github.io/blog/2018/09/24/sutton-chap04-dp#41-policy-evaluation-prediction
        """

        for i in range(self.iterations):  # iteration loop

            # create new table row for lookahead values
            values_lookahead = util.Counter()

            # get states at this iteration
            states = self.mdp.getStates()

            # store the max actions at each state to the lookahead row
            for state in states:
                max_action = self.getAction(state)
                if max_action:
                    values_lookahead[state] = self.getQValue(state, max_action)

            # update lookahead row to the current values row for next iteration to use
            self.values = values_lookahead

    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"

        # get transition state info: ((x,y), prob)
        transition_model = self.mdp.getTransitionStatesAndProbs(state, action)

        # compute Q(s,a) and sum for the action
        q_value_for_state_action = 0.0

        for item in transition_model:
            next_state = item[0]  # (x, y) or 'TERMINAL STATE'
            probability = item[1]  # 0 <= P <= 1.0
            reward = self.mdp.getReward(state, action, next_state)
            next_state_value = self.getValue(next_state)

            # Q(s,a) = (T * [R + gamma V']
            q_value_for_state_action += probability * (reward + self.discount * next_state_value)

        return q_value_for_state_action

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """

        # no max action at terminal state!
        if self.mdp.isTerminal(state):
            return None

        # get all actions from current state
        actions = self.mdp.getPossibleActions(state)

        # store (q_value, action) pairs
        q_value_action_pairs = []
        for action in actions:
            q_value_action_pairs.append((self.getQValue(state, action), action))

        # extract max action
        max_q_value, max_action = max(q_value_action_pairs, key=lambda x: x[0])

        return max_action

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)


class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """

    def __init__(self, mdp, discount=0.9, iterations=1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        """
        Each iteration only updates a single state at a time based on list of states
        generated at before iteration begins.
        """
        "*** YOUR CODE HERE ***"

        states_list = self.mdp.getStates()

        current_state_idx, n = 0, len(states_list)

        for i in range(self.iterations):

            # if last iteration was last state, reset idx
            if current_state_idx == n:
                current_state_idx = 0

            # calculate the action yielding max Q value at current_state
            # then update its value
            current_state = states_list[current_state_idx]
            max_action = self.getAction(current_state)
            if max_action:
                self.values[current_state] = self.getQValue(current_state, max_action)

            current_state_idx += 1  # go to next state on next iteration


class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """

    def __init__(self, mdp, discount=0.9, iterations=100, theta=1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        """
        http://incompleteideas.net/book/ebook/node98.html
        """
        pq = util.PriorityQueue()
        states = self.mdp.getStates()
        pr_states = self.get_predecessors_map(states)

        for state in states:
            if not self.mdp.isTerminal(state):
                diff = abs(self.getValue(state) - self.compute_maxQvalue(state))
                pq.push(state, -1.0 * diff)

        for i in range(0, self.iterations):
            if pq.isEmpty():
                return

            state = pq.pop()

            if not self.mdp.isTerminal(state):
                self.values[state] = self.compute_maxQvalue(state)

            for pr in pr_states[state]:
                diff = abs(self.getValue(pr) - self.compute_maxQvalue(pr))
                if diff > self.theta:
                    pq.update(pr, -1.0 * diff)

    def get_predecessors_map(self, states):
        """
        Predecessors <-> successors. For each state check the probability of all of
        their actions. Store in predecessors map.
        """
        predecessors = {}

        for state in states:
            actions = self.mdp.getPossibleActions(state)

            for action in actions:
                transition_model = self.mdp.getTransitionStatesAndProbs(state, action)

                for item in transition_model:
                    next_state, probability = item
                    # chance of this state reaching current state
                    if probability > 0:
                        if next_state not in predecessors:
                            predecessors[next_state] = set()
                        predecessors[next_state].add(state)
        return predecessors

    def compute_maxQvalue(self, state):
        """
        Computes the maximum q value for all possible  actions given a state.
        """
        # q_values = []
        # actions = self.mdp.getPossibleActions(state)
        # for action in actions:
        #     q_values.append(self.getQValue(state, action))
        # q_max = max(q_values) if q_values else float('-inf')
        # return q_max

        max_q_value = float('-inf')
        actions = self.mdp.getPossibleActions(state)
        for action in actions:
            q_value = self.getQValue(state, action)
            max_q_value = max(max_q_value, q_value)
        return max_q_value
