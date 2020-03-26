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

    def getCostOfactions_dict(self, actions_dict):
        """
         actions_dict: A list of actions_dict to take

        This method returns the total cost of a particular sequence of actions_dict.
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

    Your search algorithm needs to return a list of actions_dict that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    # print("Start: ", problem.getStartState())
    # print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    frontier = util.Stack()
    visited = []
    paths_to_current = util.Stack()     # Maintain a stack of path to the current node
    cur_state = problem.getStartState()
    cur_path = []       # The path to return if reach the goal
    frontier.push(cur_state)
    paths_to_current.push(cur_path)

    while not frontier.isEmpty():
        cur_state = frontier.pop()
        cur_path = paths_to_current.pop()
        if cur_state not in visited:    # cycle checking
            visited.append(cur_state)
            if problem.isGoalState(cur_state):
                return cur_path
            else:
                for node, path, cost in problem.getSuccessors(cur_state):   # expand current node
                    frontier.push(node)
                    paths_to_current.push(cur_path + [path])

    return []


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    # print("Start: ", problem.getStartState())
    # print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    frontier = util.Queue()
    visited = []  # Maintain a hashset for fast lookup of visited nodes
    paths_to_current = util.Queue()  # Maintain a queue of path to the current node
    cur_state = problem.getStartState()
    cur_path = []  # The path to return if reach the goal
    frontier.push(cur_state)
    paths_to_current.push(cur_path)

    while not frontier.isEmpty():
        cur_state = frontier.pop()
        cur_path = paths_to_current.pop()
        if cur_state not in visited:  # cycle checking
            visited.append(cur_state)
            if problem.isGoalState(cur_state):
                return cur_path
            else:
                for node, path, cost in problem.getSuccessors(cur_state):  # expand current node
                    frontier.push(node)
                    paths_to_current.push(cur_path + [path])

    return []

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    frontier = util.PriorityQueue()
    visited = []
    paths_to_current = util.PriorityQueue()  # Maintain a priority queue of path to the current node
    cur_state = problem.getStartState()
    cur_path = []  # The path to return if reach the goal
    aggre_cost = 0  # Maintain the aggregated cost from starting node to current node
    frontier.push([cur_state, 0], aggre_cost)
    paths_to_current.push(cur_path, aggre_cost)

    while not frontier.isEmpty():
        cur_state, aggre_cost = frontier.pop()
        cur_path = paths_to_current.pop()
        if cur_state not in visited:  # cycle checking
            visited.append(cur_state)
            if problem.isGoalState(cur_state):
                return cur_path
            else:
                for node, path, cost in problem.getSuccessors(cur_state):  # expand current node
                    frontier.push([node, aggre_cost + cost], aggre_cost + cost)
                    paths_to_current.push(cur_path + [path], aggre_cost + cost)

    return []

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    frontier = util.PriorityQueue()
    visited = []
    paths_to_current = util.PriorityQueue()  # Maintain a priority queue of path to the current node
    cur_state = problem.getStartState()
    cur_path = []  # The path to return if reach the goal
    aggre_cost = 0  # Maintain the aggregated cost from starting node to current node
    frontier.push([cur_state, 0], aggre_cost + heuristic(cur_state, problem))
    paths_to_current.push(cur_path, aggre_cost + heuristic(cur_state, problem))

    while not frontier.isEmpty():
        cur_state, aggre_cost = frontier.pop()
        cur_path = paths_to_current.pop()
        if cur_state not in visited:  # cycle checking
            visited.append(cur_state)
            if problem.isGoalState(cur_state):
                return cur_path
            else:
                for node, path, cost in problem.getSuccessors(cur_state):  # expand current node
                    frontier.push([node, aggre_cost + cost], aggre_cost + heuristic(node, problem) + cost)
                    paths_to_current.push(cur_path + [path], aggre_cost + heuristic(node, problem) + cost)

    return []


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
