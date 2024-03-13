import numpy as np

from Agent import Agent, AgentGreedy
from WarehouseEnv import WarehouseEnv, manhattan_distance
import random




def smart_heuristic_for_robot (env: WarehouseEnv, robot_id : int) :
    robot = env.get_robot(robot_id)
    if robot.battery == 0 :
        return robot.credit
    max = -np.inf
    if (robot.package is None):
        for p in env.packages:
            if p.on_board:
                current = 2 * manhattan_distance(p.position, p.destination) - manhattan_distance(p.position,
                                                                                                 robot.poistion)
            if current > max:
                max = current
    else:
        max = 2 * manhattan_distance(robot.package.position, robot.package.destination)

    min = np.inf
    for c in env.charge_stations:
        current = manhattan_distance(c.position, robot.poistion)
        if current < min:
            min = current
    h_robot = robot.credit + max - min
    return h_robot

# TODO: section a : 3
def smart_heuristic(env: WarehouseEnv, robot_id: int):
    robot= env.get_robot(robot_id)
    result=robot.credit
    min_dist_to_pckg = np.inf
    if robot.package is None :
        for pckg in env.packages :
            min_dist_to_pckg= min (min_dist_to_pckg,manhattan_distance(robot.position,pckg.position))
    else :
        min_dist_to_pckg = manhattan_distance(robot.position,robot.package.destination)
    
    min_dist_to_charging = np.inf
    for charging_station in env.charge_stations :
        min_dist_to_charging = min (min_dist_to_charging,manhattan_distance(robot.position,charging_station.position))
    result-=min_dist_to_charging
    result-=min_dist_to_pckg
    return result

'''def smart_heuristic(env: WarehouseEnv, robot_id: int):
    robot_h=smart_heuristic_for_robot(env,robot_id)
    other_robot_h=smart_heuristic_for_robot(env,(robot_id-1)%2)
    return robot_h - other_robot_h
'''


class AgentGreedyImproved(AgentGreedy):
    def heuristic(self, env: WarehouseEnv, robot_id: int):


        return smart_heuristic(env, robot_id)


class AgentMinimax(Agent):
    def __init__(self) :
        self.epsilon = 0.001
    # TODO: section b : 1
    def run_minimax(self, env: WarehouseEnv, agent_id, time_limit, is_maximazing):
        if env.done() : 
            return smart_heuristic(env,agent_id)
        if is_maximazing :
            currentMax = -np.inf
            for leagel_op in env.get_legal_operators(agent_id):
                env_clone = env.clone()
                env_clone.apply_operator(agent_id,leagel_op)
                evaluation = (self.run_minimax(env_clone, agent_id,time_limit-self.epsilon,False))
                currentMax = max(currentMax,evaluation)
            return currentMax
        else :
            currentMin= np.inf
            for leagel_op in env.get_legal_operators((agent_id+1)%2):
                env_clone = env.clone()
                env_clone.apply_operator((agent_id+1)%2,leagel_op)
                values = (self.run_minimax(env, (agent_id+1)%2,time_limit-self.epsilon,True))
                currentMin = min(currentMin,values)
            return currentMin



    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        return self.run_minimax(env, agent_id, time_limit,True)




        raise NotImplementedError()


class AgentAlphaBeta(Agent):
    # TODO: section c : 1
    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        raise NotImplementedError()


class AgentExpectimax(Agent):
    # TODO: section d : 1
    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        raise NotImplementedError()


# here you can check specific paths to get to know the environment
class AgentHardCoded(Agent):
    def __init__(self):
        self.step = 0
        # specifiy the path you want to check - if a move is illegal - the agent will choose a random move
        self.trajectory = ["move north", "move east", "move north", "move north", "pick_up", "move east", "move east",
                           "move south", "move south", "move south", "move south", "drop_off"]

    def run_step(self, env: WarehouseEnv, robot_id, time_limit):
        if self.step == len(self.trajectory):
            return self.run_random_step(env, robot_id, time_limit)
        else:
            op = self.trajectory[self.step]
            if op not in env.get_legal_operators(robot_id):
                op = self.run_random_step(env, robot_id, time_limit)
            self.step += 1
            return op

    def run_random_step(self, env: WarehouseEnv, robot_id, time_limit):
        operators, _ = self.successors(env, robot_id)

        return random.choice(operators)