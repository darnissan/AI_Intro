import numpy as np

from DragonBallEnv import DragonBallEnv
from typing import List, Tuple
import heapdict
import heapdict


class W_AStar_Node:
    def __init__(self, state, parent_node, action, weight, g_value, h_value):
        self.state = state
        self.parent = parent_node
        self.action_from_parent = action
        self.g_value = g_value
        self.h_value = h_value
        self.f_value = (weight * h_value) + (1 - weight) * g_value

    def __lt__(self, other):
        if self.f_value == other.f_value:
            return self.state[0] < other.state[0]
        return self.f_value < other.f_value


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

            for action, successor in env.succ(self.current_node.state).items():
                if successor[0] is None:
                    continue

                self.env.reset()
                self.env.set_state(self.current_node.state)
                s, _, _ = self.env.step(action)

                if self.env.is_final_state(s) is False and s[0] in [
                    sg[0] for sg in self.env.get_goal_states()
                ]:
                    continue

                child = NodeBFS(s, self.current_node, action)

                if child.state not in close and child.state not in [
                    n.state for n in open_nodes
                ]:
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
            _, cost, _ = self.env.step(a)
            total_cost += cost

        return [actions, total_cost, self.n_expended]


"""
A* Search – pseudo code (1/2)
A* search (search problem - P)
 OPEN <- make_node(P.start, NIL, 0, h(P.start)) //order according to f-value
 CLOSE <- {}
 While OPEN
▪ n <- OPEN.pop_min()
▪ CLOSE <- CLOSE {n}
▪ If P.goal_test(n)
o Return path(n)
▪ For s in P.SUCC (n)
o new_g <- n.g() + P.COST(n.s,s); new_f = new_g + h(s) //newly-computed cost to reach s
o If s  not in OPEN and not in CLOSED
❖ n' <- make_node(s, n, new_g, new_f )
❖ OPEN.insert(n’)
o Else if s in OPEN
❖ n_curr <- node in OPEN with state s
❖ If new_f < n_curr.f () //found better path to s
• n_curr <- update_node(s, n, new_g , new_f)
• OPEN.update_key(n_curr) //don’t forget to update place in OPEN…
❖ Else
• //do nothing – existing path is better
o Else // s in CLOSED
❖ n_curr <- node in CLOSED with state s
❖ If new_f < n_curr.f () //found better path to s
• n_curr <- update_node(s, n, new_g , new_f)
• OPEN.insert(n_curr)
• CLOSED.remove(n_curr)
❖ Return failure
"""


class WeightedAStarAgent:
    def __init__(self) -> None:
        self.Dragongoals = []
        self.env = None

    def MSAP_Heuristic(self, state, env):
        # return min (manhatan distance from the state any of  goal states )
        state_location = env.to_row_col(state)
        min_manhatan_distance = np.inf
        if state[1] and self.env.d1 in self.Dragongoals:
            self.Dragongoals.remove(self.env.d1)
        if state[2] and self.env.d2 in self.Dragongoals:
            self.Dragongoals.remove(self.env.d2)

        for g in self.Dragongoals:
            current_location = env.to_row_col(g)
            manhatan_distance = abs(state_location[0] - current_location[0]) + abs(
                state_location[1] - current_location[1]
            )
            if manhatan_distance < min_manhatan_distance:
                min_manhatan_distance = manhatan_distance

        return min_manhatan_distance

    def search(self, env, h_weight) -> tuple:
        self.env = env
        self.env.reset()
        self.Dragongoals.extend(env.get_goal_states())
        self.Dragongoals.append(env.d1)
        self.Dragongoals.append(env.d2)

        n_state = self.env.get_initial_state()
        self.current_node = W_AStar_Node(
            n_state, None, None, h_weight, 0, self.MSAP_Heuristic(n_state, env)
        )
        close = {}
        OPEN = heapdict.heapdict()
        OPEN[self.current_node.state] = (self.current_node.f_value, self.current_node)
        self.n_expended = 0

        while OPEN:
            n = OPEN.popitem()[1][1]
            close[n.state] = n
            self.n_expended += 1
            if self.env.is_final_state(n.state):
                return self.solution(n, self.n_expended)
            for action, successor in env.succ(n.state).items():
                if successor[0] is None:
                    continue
                self.env.reset()
                self.env.set_state(n.state)
                s, _, _ = self.env.step(action)
                if self.env.is_final_state(s) is False and s[0] in [
                    sg[0] for sg in self.env.get_goal_states()
                ]:
                    continue
                new_g = n.g_value + s[1]
                new_h = self.MSAP_Heuristic(s, env)
                new_f = (h_weight * new_h) + (1 - h_weight) * new_g
                child = W_AStar_Node(s, n, action, h_weight, new_g, new_h)
                if child.state not in close and child.state not in OPEN:

                    OPEN[child.state] = (child.f_value, child)
                elif child.state in OPEN:

                    if new_f < OPEN[child.state][1].f_value:
                        OPEN[child.state] = (new_f, child)
                else:
                    if new_f < close[child.state].f_value:
                        OPEN[child.state] = (new_f, child)
                        close.remove(child.state)

    def solution(self, node, n_expended):
        actions = []
        total_cost = 0
        self.env.reset()
        while node.parent is not None:
            actions.insert(0, node.action_from_parent)
            node = node.parent
            total_cost += node.g_value
        return actions, total_cost, n_expended

    class AStarEpsilonAgent:
        def __init__(self) -> None:
            pass

        def ssearch(
            self, env: DragonBallEnv, epsilon: int
        ) -> Tuple[List[int], float, int]:
            raise NotImplementedError
