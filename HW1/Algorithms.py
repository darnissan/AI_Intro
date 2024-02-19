import numpy as np

from DragonBallEnv import DragonBallEnv
from typing import List, Tuple
import heapdict


class Node:
    def __init__(self,index, parent, g_value, f_value,h_value):
        self.index = index
        self.parent = parent
        self.g_value = g_value
        self.f_value = f_value
        self.h_value = h_value

class NodeBFS:
    def __init__(self, state, parent_node, action):
        self.state = state
        self.parent_node = parent_node
        self.action_from_parent = action

class BFSAgent:
    def __init__(self) -> None:
        self.env = None

    def search(self, env: DragonBallEnv) -> Tuple[List[int], float, int]:
        self.env = env
        self.env.reset()
        state = self.env.get_initial_state()
        self.current_node = NodeBFS(state, None, None)

        if self.env.is_final_state(self.current_node.state):
            return self.solution(self.current_node)

        open_nodes = [self.current_node]
        close = []
        self.n_expended = 0

        while open_nodes:
           self.current_node = open_nodes.pop(0)
           close.append(self.current_node.state)
           self.n_expended += 1
                   
           for action in env.succ(self.current_node.state):
                self.env.reset()
                self.env.set_state(self.current_node.state)
                s, cost, terminated = self.env.step(action)

                if terminated is True and self.env.is_final_state(s) is False:
                    continue

                child = NodeBFS(s, self.current_node, action)

                if child.state not in close and child not in open_nodes:
                    if self.env.is_final_state(child.state):
                        return self.solution(child)
                    open_nodes.append(child)

    def solution(self, node):
        actions = []
        total_cost = 0
        current_node = node

        while current_node.action_from_parent is not None:
            actions.insert(0, current_node.action_from_parent)
            current_node = current_node.parent_node

        self.env.reset()

        for a in actions:
            new_state, cost, terminated = self.env.step(a)
            total_cost += cost
        
        return [actions, total_cost, self.n_expended]


class WeightedAStarAgent:
    def __init__(self) -> None:
        raise NotImplementedError
       

    def search(self, env: DragonBallEnv, h_weight) -> Tuple[List[int], float, int]:
        raise NotImplementedError
        # env.reset()
        # OPEN = heapdict.heapdict()
        # CLOSED = {}
        # start_node =
                
                
class AStarEpsilonAgent:
    def __init__(self) -> None:
        raise NotImplementedError

    def ssearch(self, env: DragonBallEnv, epsilon: int) -> Tuple[List[int], float, int]:
        raise NotImplementedError
