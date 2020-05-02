# Team member: Mukun Guo, Siqi Fu

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
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

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
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newGhostPositions = successorGameState.getGhostPositions()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        # print("successorGameState: ", str(successorGameState))
        # print("newPos: ", str(newPos))
        # print("newFood: ", str(newFood))
        # print("newGhostStates: ", [str(newGhostStates) for state in newGhostStates])
        # print("newScaredTimes: ", str(newScaredTimes))

        # Calculate distance between newPos and the closest ghost
        ghost_dists = [l1Distance(newPos, ghost_pos) for ghost_pos in newGhostPositions]
        closest_ghost_dist = min(ghost_dists)
        # print("ghost_dists: ", str(ghost_dists))
        for i in range(len(ghost_dists)):
            if ghost_dists[i] == closest_ghost_dist:
                closest_ghost_idx = i

        if closest_ghost_dist == 0:
            ghost_score = -1e6
        else:
            ghost_score = -10 / closest_ghost_dist

        # Calculate distance between newPos and the closet food
        food_list = newFood.asList()
        closest_food_dist = 0
        if food_list:
            food_dists = [l1Distance(newPos, food_pos) for food_pos in food_list]
            closest_food_dist = min(food_dists)

        if len(food_list) == 0 or closest_food_dist == 0:
            food_score = 0
        else:
            food_score = -2 * closest_food_dist

        # Reward agent if it eats a food pellet
        food_left = len(food_list)
        num_score = 30 * food_left
        # Calculate evaluation score
        # print(newScaredTimes)
        if newScaredTimes[closest_ghost_idx] > 0:
            return food_score - num_score
        else:
            return food_score + ghost_score - num_score

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

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        num_agents = gameState.getNumAgents()

        def _minimax(s, iters):
            _scores = []
            if iters >= self.depth * num_agents or s.isWin() or s.isLose(): # as ghosts move in sequence, actual depth = depth * num_agents
                return self.evaluationFunction(s), _scores
            agent_idx = iters % num_agents
            # Ghost's turn (Minimize score)
            if agent_idx != 0:
                _result = float('inf')
                for action in s.getLegalActions(agent_idx):
                    if action == 'Stop':
                        continue
                    s_next = s.generateSuccessor(agent_idx, action)
                    _result = min(_result, _minimax(s_next, iters + 1)[0])

            # Pac-man's turn
            else:
                _result = -float('inf')
                for action in s.getLegalActions(agent_idx):
                    if action == 'Stop':
                        continue
                    s_next = s.generateSuccessor(agent_idx, action)
                    _result = max(_result, _minimax(s_next, iters + 1)[0])
                    if iters == 0:
                        _scores.append(_result)
            return _result, _scores

        result, scores = _minimax(gameState, 0)
        actions = gameState.getLegalActions(0) # All moves for pac-man

        final_actions = []
        for a in actions:
            if a != 'Stop':
                final_actions.append(a)
        # print(final_actions)
        return final_actions[scores.index(max(scores))]

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        num_agents = gameState.getNumAgents()

        def _alphabeta(s, iters, alpha, beta):
            _scores = []
            if iters >= self.depth * num_agents or s.isWin() or s.isLose(): # as ghosts move in sequence, actual depth = depth * num_agents
                return self.evaluationFunction(s), _scores
            agent_idx = iters % num_agents
            # Ghost's turn (Minimize score)
            if agent_idx != 0:
                _result = float('inf')
                for action in s.getLegalActions(agent_idx):
                    if action == 'Stop':
                        continue
                    s_next = s.generateSuccessor(agent_idx, action)
                    _result = min(_result, _alphabeta(s_next, iters + 1, alpha, beta)[0])
                    beta = min(beta, _result)
                    if beta < alpha:
                        break

            # Pac-man's turn
            else:
                _result = -float('inf')
                for action in s.getLegalActions(agent_idx):
                    if action == 'Stop':
                        continue
                    s_next = s.generateSuccessor(agent_idx, action)
                    _result = max(_result, _alphabeta(s_next, iters + 1, alpha, beta)[0])
                    if iters == 0:
                        _scores.append(_result)
                    alpha = max(alpha, _result)
                    if beta < alpha:
                        break
            return _result, _scores

        result, scores = _alphabeta(gameState, 0, -float('inf'), float('inf'))
        actions = gameState.getLegalActions(0) # All moves for pac-man

        final_actions = []
        for a in actions:
            if a != 'Stop':
                final_actions.append(a)
        # print(final_actions)
        return final_actions[scores.index(max(scores))]

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
        num_agents = gameState.getNumAgents()

        def _expectiMax(s, iters):
            _scores = []
            if iters >= self.depth * num_agents or s.isWin() or s.isLose():  # as ghosts move in sequence, actual depth = depth * num_agents
                return self.evaluationFunction(s), _scores
            agent_idx = iters % num_agents
            # Ghost's turn (Minimize score)
            if agent_idx != 0:
                successorScores = []
                for action in s.getLegalActions(agent_idx):
                    if action == 'Stop':
                        continue
                    s_next = s.generateSuccessor(agent_idx, action)

                    # Calculate expected score
                    successorScores.append(_expectiMax(s_next, iters + 1)[0])
                expectation = sum([float(x) for x in successorScores]) / len(successorScores)
                return expectation, _scores
            # Pac-man's turn
            else:
                _result = -float('inf')
                for action in s.getLegalActions(agent_idx):
                    if action == 'Stop':
                        continue
                    s_next = s.generateSuccessor(agent_idx, action)
                    _result = max(_result, _expectiMax(s_next, iters + 1)[0])
                    if iters == 0:
                        _scores.append(_result)
            return _result, _scores

        result, scores = _expectiMax(gameState, 0)
        actions = gameState.getLegalActions(0)  # All moves for pac-man

        final_actions = []
        for a in actions:
            if a != 'Stop':
                final_actions.append(a)
        # print(final_actions)
        return final_actions[scores.index(max(scores))]

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: Our solution can be divided into 4 parts
    1. ghost_score: when ghosts are not scared, run away from the ghost. Being too closed to the ghost results in a huge penalty
    2. food_score: reward the pac-man if it's closer to the farthest food (analogical to the foodSearch heuristic)
    3. capsule_score: reward the pac-man if it's closer to the capsule
    4. reward the pac-man if it's able to eat one food pellet
    """
    "*** YOUR CODE HERE ***"
    # successorGameState = currentGameState.generatePacmanSuccessor(action)
    currentPos = currentGameState.getPacmanPosition()
    currentFood = currentGameState.getFood()
    currentGhostStates = currentGameState.getGhostStates()
    currentGhostPositions = currentGameState.getGhostPositions()
    currentScaredTimes = [ghostState.scaredTimer for ghostState in currentGhostStates]

    # Calculate distance between newPos and the closest ghost
    ghost_dists = [l1Distance(currentPos, ghost_pos) for ghost_pos in currentGhostPositions]
    closest_ghost_idx = ghost_dists.index(min(ghost_dists))
    closest_ghost_dist = ghost_dists[closest_ghost_idx]
    if currentScaredTimes[closest_ghost_idx] <= 1:
        ghost_score = -pow(max(5-closest_ghost_dist, 0), 2) # when not having scaredTimer, run away from the ghost
    else:
        ghost_score = pow(max(10-closest_ghost_dist, 0), 2) # otherwise try to eat the ghost when it's nearby

    # Calculate distance between newPos and the closet food
    food_list = currentFood.asList()
    food_dists = [l1Distance(currentPos, food_pos) for food_pos in food_list]
    num_food_left = len(food_dists)
    if num_food_left > 0:
        food_score = 1 / max(food_dists)
    else:
        food_score = 0

    # Calculate distance between pacman and the closest capsule
    capsules = currentGameState.getCapsules()
    if len(capsules) > 0:
        capsule_dists = [l1Distance(currentPos, capsule_pos) for capsule_pos in capsules]
        capsule_score = 20 / min(capsule_dists)
    else:
        capsule_score = 0

    # Reward agent if it eats a food pellet
    food_left = len(food_list)

    # Calculate evaluation score

    return currentGameState.getScore() + food_score + ghost_score + capsule_score - 30 * len(food_list)

# Abbreviation
better = betterEvaluationFunction

# Some helper functions
def l1Distance(pos1, pos2):
    x1, y1 = pos1
    x2, y2 = pos2
    return abs(x1 - x2) + abs(y1 - y2)
