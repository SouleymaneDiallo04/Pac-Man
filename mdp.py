# mdp.py
# ------
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


from abc import ABC, abstractmethod

class MarkovDecisionProcess(ABC):  # Hérite de ABC
    @abstractmethod
    def getStates(self):
        pass

    @abstractmethod
    def getStartState(self):
        pass

    @abstractmethod
    def getPossibleActions(self, state):
        pass

    @abstractmethod
    def getTransitionStatesAndProbs(self, state, action):
        pass

    @abstractmethod
    def getReward(self, state, action, nextState):
        pass

    @abstractmethod
    def isTerminal(self, state):
        pass
