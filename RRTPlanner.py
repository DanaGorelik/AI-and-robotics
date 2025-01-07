import numpy as np
from RRTTree import RRTTree
import time

class RRTPlanner(object):

    def __init__(self, planning_env, ext_mode, goal_prob):

        # set environment and search tree
        self.planning_env = planning_env
        self.tree = RRTTree(self.planning_env)

        # set search params
        self.ext_mode = ext_mode
        self.goal_prob = goal_prob

    def plan(self) -> np.array:
        '''
        Compute and return the plan. The function should return a numpy array containing the states (positions) of the robot.
        '''
        start_time = time.time()

        # initialize an empty plan.
        plan = []

        # TODO: Task 3
        start = self.planning_env.start
        goal = self.planning_env.goal
        self.tree.add_vertex(start)
        
        q_new = start
        while self.planning_env.compute_distance(goal, q_new) != 0:
            q_rand = self.sample_random_state()
            if not self.planning_env.state_validity_checker(q_rand):
                raise ValueError('Rand state must be within the map limits')
            # q_near = self.tree.get_nearest_vertex(q_rand)
            q_near = self.tree.get_nearest_state(q_rand)
            if not self.planning_env.state_validity_checker(q_near):
                raise ValueError('Near state must be within the map limits')
            q_new = self.extend(q_near, q_rand)
            if q_new is not None:
                self.tree.add_vertex(q_new)
                self.tree.add_edge(q_near, q_new)
                
        plan = self.plan_from_path(self.tree.get_idx_for_state(goal))           
        
        # print total path cost and time
        print('Total cost of path: {:.2f}'.format(self.compute_cost(plan)))
        print('Total time: {:.2f}'.format(time.time()-start_time))

        return np.array(plan)

    def compute_cost(self, plan: np.array) -> float:
        '''
        Compute and return the plan cost, which is the sum of the distances between steps.
        @param plan A given plan for the robot.
        '''
        # TODO: Task 3
        total_cost: float = 0
        for i in range(len(plan)-1):
            total_cost += self.planning_env.compute_distance(plan[i], plan[i+1])

        return total_cost

    def extend(self, near_state, rand_state):
        '''
        Compute and return a new position for the sampled one.
        @param near_state The nearest position to the sampled position.
        @param rand_state The sampled position.
        '''
        # TODO: Task 3
        if self.ext_mode == "E1":
            # Full extension to the random state
            new_state = rand_state
        elif self.ext_mode == "E2":
            # Incremental extension towards the random state with step size η
            direction = rand_state - near_state
            distance = np.linalg.norm(direction)
            step_size = 5  # Define a step size (can also be passed dynamically)
            if distance <= step_size:
                new_state = rand_state
            else:
                new_state = near_state + (direction / distance) * step_size
        else:
            raise ValueError("Invalid extension mode: {}".format(self.ext_mod))
        
        return new_state
    
    def plan_from_path(self, goal_idx):
        '''
        Reconstruct the path from the goal to the start using parent links in the tree.
        Args:
            goal_idx: Index of the goal node in the tree.
        Returns:
            List of states (path) from start to goal.
        '''
        path = []
        current_idx = goal_idx

        while current_idx is not None:
            # Add the current state's coordinates to the path
            path.append(self.tree.get_vertex(current_idx))
            # Move to the parent of the current node
            current_idx = self.tree.get_parent(current_idx)
        
        # Reverse the path to start from the initial position
        return path[::-1]

    def sample_random_state(self):
            '''
            Sample a random state within the environment's bounds.
            '''
            x = np.random.uniform(self.planning_env.xlimit[0], self.planning_env.xlimit[1])
            y = np.random.uniform(self.planning_env.ylimit[0], self.planning_env.ylimit[1])
            return np.array([x, y])
    