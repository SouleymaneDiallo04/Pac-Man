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
    """

    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)
        self.qvalues = util.Counter()  # dict with default 0

    def getQValue(self, state, action):
        """
        Returns Q(state,action), 0.0 if unseen.
        """
        return self.qvalues[(state, action)]

    def computeValueFromQValues(self, state):
        """
        Returns max_action Q(state,action) over legal actions.
        Returns 0.0 if there are no legal actions (terminal state).
        """
        actions = self.getLegalActions(state)
        if not actions:
            return 0.0
        return max(self.getQValue(state, a) for a in actions)

    def computeActionFromQValues(self, state):
        """
        Returns the best action according to Q-values.
        Returns None if there are no legal actions.
        """
        actions = self.getLegalActions(state)
        if not actions:
            return None
        best_value = self.computeValueFromQValues(state)
        best_actions = [a for a in actions if self.getQValue(state, a) == best_value]
        return random.choice(best_actions)

    def getAction(self, state):
        """
        Epsilon-greedy action selection.
        With probability epsilon choose a random action,
        otherwise choose the best one.
        """
        legalActions = self.getLegalActions(state)
        if not legalActions:
            return None
        if util.flipCoin(self.epsilon):
            return random.choice(legalActions)
        else:
            return self.computeActionFromQValues(state)

    def update(self, state, action, nextState, reward):
        """
        Performs the Q-learning update:
        Q(s,a) <- (1 - alpha)*Q(s,a) + alpha*(reward + discount*max_a' Q(s',a'))
        """
        sample = reward + self.discount * self.computeValueFromQValues(nextState)
        self.qvalues[(state, action)] = (1 - self.alpha) * self.getQValue(state, action) + self.alpha * sample

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Same as QLearningAgent, but with different default parameters."

    def __init__(self, epsilon=0.05, gamma=0.8, alpha=0.2, numTraining=0, **args):
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # Pacman is always agent index 0
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Just calls QLearningAgent.getAction and informs parent of action.
        """
        action = QLearningAgent.getAction(self, state)
        self.doAction(state, action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
    ApproximateQLearningAgent
    You should only have to overwrite getQValue and update.
    """

    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
        Q(state,action) = w * featureVector
        """
        features = self.featExtractor.getFeatures(state, action)
        return sum(self.weights[f] * features[f] for f in features)

    def update(self, state, action, nextState, reward):
        """
        Update weights based on transition.
        """
        features = self.featExtractor.getFeatures(state, action)
        difference = (reward + self.discount * self.computeValueFromQValues(nextState)) - self.getQValue(state, action)
        for f in features:
            self.weights[f] += self.alpha * difference * features[f]

    def final(self, state):
        """Called at the end of each game."""
        PacmanQAgent.final(self, state)
        if self.episodesSoFar == self.numTraining:
            # print("Weights:", self.weights)
            pass
