import numpy as np
import time
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
'''
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

def smart_heuristic(env: WarehouseEnv, robot_id: int):
    robot_h=smart_heuristic_for_robot(env,robot_id)
    other_robot_h=smart_heuristic_for_robot(env,(robot_id-1)%2)
    return robot_h - other_robot_h
'''


class AgentGreedyImproved(AgentGreedy):
    def heuristic(self, env: WarehouseEnv, robot_id: int):


        return smart_heuristic(env, robot_id)
    
def smart_heuristic(env: WarehouseEnv, robot_id: int):
    robot=env.get_robot(robot_id)
    result=0
    if robot.package is  None :
        pckg1=env.packages[0]
        dist_pack1_to_dest=manhattan_distance(pckg1.position,pckg1.destination)
        dist_pckg1_to_robot = manhattan_distance(pckg1.position,robot.position)
        

        pckg2=env.packages[1]
        dist_pack2_to_dest=manhattan_distance(pckg2.position,pckg2.destination)
        dist_pckg2_to_robot = manhattan_distance(pckg2.position,robot.position)

        result = max(dist_pack1_to_dest-dist_pckg1_to_robot,dist_pack2_to_dest-dist_pckg2_to_robot)
    
    else :
        current_pckg=robot.package
        dist_pckg_to_dest= manhattan_distance(current_pckg.position,current_pckg.destination)
        dist_dest_to_robot=manhattan_distance(current_pckg.destination,robot.position)
        result =2*dist_pckg_to_dest - dist_dest_to_robot
        
    return result  


class AgentMinimax(Agent):
    def __init__(self) :
        self.epsilon = 0.01
    # TODO: section b : 1
    def run_minimax(self, env: WarehouseEnv, agent_id, time_limit, is_maximazing,deadline,last_op):
        if time.time() + self.epsilon > deadline or env.done() :
            return last_op, smart_heuristic(env,agent_id) or env.done()
        if is_maximazing :
            operators = env.get_legal_operators(agent_id)
            children = [env.clone() for _ in operators]
            for child, op in zip(children, operators):
                child.apply_operator(agent_id, op)
            children_heuristics = [self.run_minimax(child,agent_id,time_limit,False,deadline,op)[1] for child,op  in zip (children,operators)]
            max_heuristic = max(children_heuristics)
            index_selected = children_heuristics.index(max_heuristic)
            return operators[index_selected] , max_heuristic


        else :
            operators = env.get_legal_operators(agent_id)
            children = [env.clone() for _ in operators]
            for child, op in zip(children, operators):
                child.apply_operator(agent_id, op)
            children_heuristics = [self.run_minimax(child,(agent_id+1)%2,time_limit,True,deadline,op)[1] for child,op  in zip (children,operators)]
            min_heuristic = min(children_heuristics)
            index_selected = children_heuristics.index(min_heuristic)
            return operators[index_selected] , min_heuristic



    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        start_time = time.time()
        deadline = start_time + time_limit
        op= self.run_minimax(env, agent_id, time_limit,True,deadline,'park')[0]
        return op


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