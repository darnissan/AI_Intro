import numpy as np

from DragonBallEnv import DragonBallEnv
from typing import List, Tuple
import heapdict
import heapdict


class Agent:
    def __init__(self) -> None:
        pass

    def search(self, env: DragonBallEnv) -> Tuple[List[int], float, int]:
        pass

    def MSAP_Heuristic(self, state, env):
        # return min (manhatan distance from the state any of  goal states )
        state_location = env.to_row_col(state)
        min_manhatan_distance = np.inf

        for g in self.Dragongoals:

            if g == env.d1 and state[1] == True:  # and state[0]!=g[0] :
                continue
            if g == env.d2 and state[2] == True:  # and state[0]!=g[0]:
                continue

            current_location = env.to_row_col(g)
            manhatan_distance = abs(state_location[0] - current_location[0]) + abs(
                state_location[1] - current_location[1]
            )
            if manhatan_distance < min_manhatan_distance:
                min_manhatan_distance = manhatan_distance

        return min_manhatan_distance


class W_AStar_Node:
    def __init__(
        self,
        state,
        parent_node,
        action,
        weight,
        g_value,
        h_value,
        terminated,
    ):
        self.state = state
        self.parent = parent_node
        self.action_from_parent = action
        self.g_value = g_value
        self.h_value = h_value
        self.f_value = (weight * h_value) + (1 - weight) * g_value
        self.is_terminated = terminated

    def __lt__(self, other):
        if self.f_value == other.f_value:
            return self.state[0] < other.state[0]
        return self.f_value < other.f_value


class Epsilon_AStar_Node:
    def __init__(
        self, state, parent_node, action, g_value, h_value, terminated
    ) -> None:
        self.state = state
        self.parent = parent_node
        self.action_from_parent = action
        self.g_value = g_value
        self.h_value = h_value
        self.is_terminated = terminated
        self.f_value = self.g_value + self.h_value

    def __lt__(self, other):
        if self.f_value == other.f_value:
            return self.state[0] < other.state[0]
        return self.f_value < other.f_value


class NodeBFS:
    def __init__(self, state, parent_node, action, cost, terminated):
        self.state = state
        self.parent_node = parent_node
        self.action_from_parent = action
        self.cost = cost
        self.is_terminated = terminated


class BFSAgent:
    def __init__(self) -> None:
        self.env = None

    def search(self, env: DragonBallEnv) -> Tuple[List[int], float, int]:
        self.env = env
        self.env.reset()
        state = self.env.get_initial_state()
        self.current_node = NodeBFS(state, None, None, 0, False)

        if self.env.is_final_state(self.current_node.state):
            return self.solution(self.current_node)

        open_nodes = [self.current_node]
        close = []
        self.n_expended = 0

        while open_nodes:
            self.current_node = open_nodes.pop(0)
            close.append(self.current_node.state)

            self.n_expended += 1

            if self.current_node.is_terminated:
                continue

            for action, successor in env.succ(self.current_node.state).items():
                self.env.reset()
                self.env.set_state(self.current_node.state)
                s, cost, terminated = self.env.step(action)

                child = NodeBFS(s, self.current_node, action, cost, terminated)

                if child.state not in close and child.state not in [
                    n.state for n in open_nodes
                ]:
                    if self.env.is_final_state(child.state):
                        return self.solution(child)
                    open_nodes.append(child)
        return ([], 0, 0)

    def solution(self, node):
        actions = []
        total_cost = 0
        current_node = node

        while current_node.action_from_parent is not None:
            actions.insert(0, current_node.action_from_parent)
            total_cost += current_node.cost
            current_node = current_node.parent_node

        return [actions, total_cost, self.n_expended]


class WeightedAStarAgent(Agent):
    def __init__(self) -> None:
        self.Dragongoals = []
        self.env = None

    def search(self, env, h_weight) -> tuple:
        self.env = env
        self.env.reset()
        self.Dragongoals = []
        self.Dragongoals.extend(env.get_goal_states())
        self.Dragongoals.append(env.d1)
        self.Dragongoals.append(env.d2)

        self.current_node = W_AStar_Node(
            self.env.get_initial_state(),
            None,
            None,
            h_weight,
            0,
            self.MSAP_Heuristic(self.env.get_initial_state(), env),
            False,
        )
        close = {}
        OPEN = heapdict.heapdict()
        OPEN[self.current_node.state] = (self.current_node.f_value, self.current_node)
        self.n_expended = 0
        while OPEN:
            self.current_node = OPEN.popitem()[1][1]
            self.env.set_state_2(self.current_node.state)
            close[self.current_node.state] = self.current_node
            if self.env.is_final_state(self.current_node.state):
                return self.solution(self.current_node, self.n_expended)
            self.n_expended += 1
            if self.current_node.is_terminated:
                # self.n_expended += 1# meaning we reached a hole
                continue

            if self.env.is_final_state(
                self.current_node.state
            ) is False and self.current_node.state[0] in [
                sg[0] for sg in self.env.get_goal_states()
            ]:
                # self.n_expended += 1
                continue

            for action, _ in env.succ(self.current_node.state).items():

                self.env.reset()
                self.env.set_state_2(self.current_node.state)
                steped_state, steped_cost, steped_is_terminated = self.env.step(action)

                new_g = self.current_node.g_value + steped_cost
                new_h = self.MSAP_Heuristic(steped_state, env)
                new_f = (h_weight * new_h) + (1 - h_weight) * new_g
                child = W_AStar_Node(
                    steped_state,
                    self.current_node,
                    action,
                    h_weight,
                    new_g,
                    new_h,
                    steped_is_terminated,
                )

                #                if self.env.is_final_state(child.state):
                # meaning we reached goal

                #                   return self.solution(child, self.n_expended)

                if (child.state in OPEN) == False and (child.state in close) == False:
                    OPEN[child.state] = (child.f_value, child)
                elif child.state in OPEN:
                    if new_f < OPEN[child.state][0]:

                        OPEN[child.state] = (new_f, child)
                else:
                    if new_f < close[child.state].f_value:
                        OPEN[child.state] = (new_f, child)
                        close.pop(child.state)
        return ([], 0, 0)

    def solution(self, node, n_expended):
        actions = []
        total_cost = 0
        current_node = node

        while current_node.action_from_parent is not None:
            actions.insert(0, current_node.action_from_parent)
            current_node = current_node.parent

        self.env.reset()

        for a in actions:
            _, cost, _ = self.env.step(a)
            total_cost += cost

        return [actions, total_cost, self.n_expended]


class AStarEpsilonAgent(Agent):
    def __init__(self) -> None:
        self.Dragongoals = []
        self.env = None

    def update_focal(self, Focal, OPEN, epsilon):
        Focal.clear()
        min_f = OPEN.peekitem()[1][1].f_value
        for state, (f, node) in OPEN.items():
            if f<= min_f * (1 + epsilon):
                Focal[state] = (node.g_value,node.state[0], node)
        return Focal

    def search(self, env: DragonBallEnv, epsilon: int) -> Tuple[List[int], float, int]:
        self.env = env
        self.env.reset()
        self.Dragongoals = []
        self.Dragongoals.extend(env.get_goal_states())
        self.Dragongoals.append(env.d1)
        self.Dragongoals.append(env.d2)

        self.current_node = Epsilon_AStar_Node(
            self.env.get_initial_state(),
            None,
            None,
            0,
            self.MSAP_Heuristic(self.env.get_initial_state(), env),
            False,
        )
        close = {}
        OPEN = heapdict.heapdict()
        OPEN[self.current_node.state] = (self.current_node.f_value, self.current_node)
        self.n_expended = 0
        Focal = heapdict.heapdict()
        while OPEN:
            Focal = self.update_focal(Focal, OPEN, epsilon)
            self.current_node = Focal.popitem()[1][2]
            OPEN.pop(self.current_node.state)

            close[self.current_node.state] = self.current_node
            if self.env.is_final_state(self.current_node.state):
                return self.solution(self.current_node, self.n_expended)
            self.n_expended += 1
            if self.current_node.is_terminated:
                # self.n_expended += 1# meaning we reached a hole
                continue

            if self.env.is_final_state(
                self.current_node.state
            ) is False and self.current_node.state[0] in [
                sg[0] for sg in self.env.get_goal_states()
            ]:
                # self.n_expended += 1
                continue

            for action, _ in env.succ(self.current_node.state).items():

                self.env.reset
                self.env.set_state_2(self.current_node.state)
                steped_state, steped_cost, steped_is_terminated = self.env.step(action)

                new_g = self.current_node.g_value + steped_cost
                new_h = self.MSAP_Heuristic(steped_state, env)
                new_f = new_h + new_g
                child = Epsilon_AStar_Node(
                    steped_state,
                    self.current_node,
                    action,
                    new_g,
                    new_h,
                    steped_is_terminated,
                )

                #                if self.env.is_final_state(child.state):
                # meaning we reached goal

                #                   return self.solution(child, self.n_expended)

                if (child.state in OPEN) == False and (child.state in close) == False:
                    OPEN[child.state] = (child.f_value, child)
                elif child.state in OPEN:
                    if new_f < OPEN[child.state][1].f_value:
                        OPEN[child.state] = (new_f, child)
                else:
                    if new_f < close[child.state].f_value:
                        OPEN[child.state] = (new_f, child)
                        close.pop(child.state)
        return ([], 0, 0)

    def solution(self, node, n_expended):
        actions = []
        total_cost = 0
        current_node = node

        while current_node.action_from_parent is not None:
            actions.insert(0, current_node.action_from_parent)
            current_node = current_node.parent

        self.env.reset()

        for a in actions:
            _, cost, _ = self.env.step(a)
            total_cost += cost

        return [actions, total_cost, self.n_expended]
