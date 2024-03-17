import numpy as np
import time
from Agent import Agent, AgentGreedy
from WarehouseEnv import WarehouseEnv, manhattan_distance
import random



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


def smart_heuristic(env: WarehouseEnv, robot_id: int):
    robot = env.get_robot(robot_id)
    other_robot = env.get_robot(1 - robot_id)

    # Weights for different components of the heuristic
    weight_distance_to_package = -1  # We want to minimize distance
    weight_distance_to_destination = -1  # Minimize distance to destination
    weight_battery_level = 5  # Higher weight because battery is critical
    weight_score_difference = 10  # Prioritize maintaining/increasing score lead
    weight_packages_delivered = 15  # High priority on delivering packages

    nearest_package_distance = min([manhattan_distance(robot.position, package.position)
                                    for package in env.packages if package.on_board], default=0)
    if robot.package:
        distance_to_destination = manhattan_distance(robot.position, robot.package.destination)
        nearest_package_distance=0
    else:
        distance_to_destination = 0  # No package, no distance to calculate

    battery_level = robot.battery
    score_difference = robot.credit - other_robot.credit
    packages_delivered = robot.credit // 2  # Assuming each delivery scores 2 points

    heuristic_value = (weight_distance_to_package * nearest_package_distance +
                       weight_distance_to_destination * distance_to_destination +
                       weight_battery_level * battery_level +
                       weight_score_difference * score_difference +
                       weight_packages_delivered * packages_delivered)
    return heuristic_value


class AgentGreedyImproved(AgentGreedy):
    def heuristic(self, env: WarehouseEnv, robot_id: int):


        return smart_heuristic(env, robot_id)
    
 


def minimax_decision(state, agent_id, depth, time_limit, start_time, is_maximizing):
    """Perform the minimax decision making."""
    real_player= agent_id if is_maximizing else (1-agent_id)
    if (time.time() - start_time) >= time_limit*0.9:
        raise TimeoutError
    if state.done() or depth == 0 : 
        return smart_heuristic(state,real_player), None
    
    best_value = float('-inf') if is_maximizing == True else float('inf')
    best_operator = None
    for operator, child_state in successors(state, agent_id):
        if (time.time() - start_time) >= time_limit*0.9:
            raise TimeoutError
        value, _ = minimax_decision(child_state,1- agent_id, depth - 1, time_limit, start_time, not is_maximizing)
        if is_maximizing:  # Maximizing player
            if value >= best_value:
                best_value, best_operator = value, operator
        else:  # Minimizing player
            if value <= best_value:
                best_value, best_operator = value, operator
    return best_value, best_operator

def heuristic_value(state, agent_id):
    """A simple heuristic function to evaluate the game state."""
    # This can be replaced with a more sophisticated heuristic function
    robot = state.get_robot(agent_id)
    other_robot = state.get_robot(1 - agent_id)
    return robot.credit - other_robot.credit

def successors(state, agent_id):
    """Generate successors for the current state and agent."""
    operators = state.get_legal_operators(agent_id)
    children = []
    for op in operators:
        child = state.clone()
        child.apply_operator(agent_id, op)
        children.append((op, child))
    return children

class AgentMinimax(Agent):
    def __init__(self):
      super().__init__()
      self.last_calculated_move_value=None
        

    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        start_time = time.time()
        self.depth=1
        self.best_move=None
        self.best_value=None
        try :
            while True :
                best_iteration_value, best_iteration_operator = minimax_decision(env, agent_id, self.depth, time_limit, start_time, True)
                if self.best_value is None or best_iteration_value > self.best_value:
                    self.best_value = best_iteration_value
                    self.best_move = best_iteration_operator
                self.depth+=1
        #_, best_operator = minimax_decision(env, agent_id, depth=3, time_limit=time_limit, start_time=start_time,is_maximizing=True)
        except TimeoutError:
            pass
        return self.best_move if self.best_move is not None else random.choice(env.get_legal_operators(agent_id))


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