# qlearningAgents.py
# ------------------
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


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random, util, math


class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """

    def __init__(self, **args):
        ReinforcementAgent.__init__(self, **args)
        self.q_values = util.Counter()  # q values for each (state, action). { (s,a) : 12.3, (s,a) : 0.0, ... }

    def no_legal_actions(self, state):
        """ Returns true if there are no legal actions from a given state."""
        return not self.getLegalActions(state)

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        key = (state, action)
        if key in self.q_values:
            return self.q_values[key]
        return 0.0

    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        if self.no_legal_actions(state):
            return 0.0
        max_action = self.getPolicy(state)
        max_action_q_value = self.getQValue(state, max_action)
        return max_action_q_value  # Q(s, max_action)

    def computeActionFromQValues(self, state):
        """
        Compute the best action to take in a state.  Note that if there
        are no legal actions, which is the case at the terminal state,
        you should return None.
        """
        actions = self.getLegalActions(state)
        # state is TERMINAL_STATE
        if not actions:
            return None
        # find best action by calculating all qvalues
        max_q_value = float('-inf')
        max_action = None
        for action in actions:
            q_value = self.getQValue(state, action)
            if q_value > max_q_value:
                max_q_value = q_value
                max_action = action
        return max_action

    def getAction(self, state):
        """
        Compute the action to take in the current state.  With
        probability self.epsilon, we should take a random action and
        take the best policy action otherwise.  Note that if there are
        no legal actions, which is the case at the terminal state, you
        should choose None as the action.
        """
        legalActions = self.getLegalActions(state)
        action = None
        # state is TERMINAL_STATE
        if self.no_legal_actions(state):
            return action
        # choose random action based on e-greedy
        take_random_action = util.flipCoin(self.epsilon)
        if take_random_action:
            action = random.choice(legalActions)
        else:  # choose optimal action (pi*)
            action = self.getPolicy(state)
        return action

    def update(self, state, action, nextState, reward):
        """
        The parent class calls this to observe a
        state = action => nextState and reward transition.
        You should do your Q-Value update here

        NOTE: You should never call this function,
        it will be called on your behalf
        """
        # qsample = R(s,a,s') + discount * q_max_action
        q_sample = reward + self.discount * self.getValue(nextState)
        # Q(s,a) current from self.q_values
        q_sa = self.getQValue(state, action)
        #               Q(s,a) update <-  (1 - alpha) * Q(s,a)     + ( alpha * q_sample)
        self.q_values[(state, action)] = ((1 - self.alpha) * q_sa) + (self.alpha * q_sample)

    def getPolicy(self, state):
        """ Return optimal action to take from state. """
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        """ Return Q value from max-action. Q(s,a)_max_action """
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05, gamma=0.8, alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self, state)
        self.doAction(state, action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """

    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getFeaturesMap(self, state, action):
        """
        Get an updated features map from the IdentityExtractor.
        """
        return self.featExtractor.getFeatures(state, action)

    def getQValue(self, state, action):
        """
        Should return Q(state,action) = w * featureVector
        where * is the dotProduct operator
        """
        features = self.getFeaturesMap(state, action)
        q_sa = 0.0
        # each feature description as key
        for feature_key in features:
            feature_value = features[feature_key]  # corresponding value
            q_sa += feature_value * self.weights[feature_key]
        return q_sa

    def update(self, state, action, nextState, reward):
        """
        Should update your weights based on transition
        """
        features = self.getFeaturesMap(state, action)
        diff = (reward + self.discount * self.getValue(nextState)) - self.getQValue(state, action)
        for feature_key in features:
            feature_value = features[feature_key]
            # update weight vector
            self.weights[feature_key] += self.alpha * diff * feature_value

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)
        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            # print(self.getWeights())
