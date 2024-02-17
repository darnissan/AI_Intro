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


class BFSAgent:
    def __init__(self) -> None:
        raise NotImplementedError

    def search(self, env: DragonBallEnv) -> Tuple[List[int], float, int]:
        raise NotImplementedError


class WeightedAStarAgent:
    def __init__(self) -> None:
        
       

    def search(self, env: DragonBallEnv, h_weight) -> Tuple[List[int], float, int]:
        env.reset()
        OPEN = heapdict.heapdict()
        CLOSED = {}
        start_node =
                
                
class AStarEpsilonAgent:
    def __init__(self) -> None:
        raise NotImplementedError

    def ssearch(self, env: DragonBallEnv, epsilon: int) -> Tuple[List[int], float, int]:
        raise NotImplementedError
