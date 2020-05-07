# Travelling Salesman Problem using Held-Karp Algorithim

import sys
import argparse
import math
import subprocess
import time as MASTER_CLOCK
import itertools

# Node class for all coordinates given by user
class Node:
    def __init__(self, node_id, x_coord, y_coord):
        self.node_id = node_id 
        self.x_coord = x_coord
        self.y_coord = y_coord

'''
--------------------------------------------------------------------------------------------------------------------
Set of functions for getting user arguments and miscellaneous functions
--------------------------------------------------------------------------------------------------------------------
'''

# function that gets user arguments from the command line
def get_user_args():
    # get the arguments from the cmd
    parser = argparse.ArgumentParser(description="This file processes the Travelling Salesman Problem using the Held-Karp Algorithim with some tweaks. ")
    parser.add_argument(
        "input_coordinates_file",
        help="File to get node coordinates from.",
    )
    parser.add_argument("output_tour_file", help="Output file for results of TSP algorithim.")
    parser.add_argument("time", type=int, help="Time for the TSP to execute.")

    args = parser.parse_args()  # get the arguments passed in

    open(args.input_coordinates_file, "r").close() # checks if input file is valid
    open(args.output_tour_file, "w").close()  # makes new output file if doesn't exist
    args.time # time that will be used to terminate
    return args


# function that checks if program should terminate
def check_timeout(args):
    if (MASTER_CLOCK.perf_counter()  - start_time) >= args.time:
        exit()

# function to print out the cost matrix (TESTING PURPOSES ONLY)
def print_cost_matrix(cost_matrix):
    for row in range(len(cost_matrix[0])):
        for col in range(len(cost_matrix[0])):
            print(cost_matrix[row][col], "  ", end = "")
            
            if col == len(cost_matrix[0]) - 1:
                print(" ")

'''
--------------------------------------------------------------------------------------------------------------------
Set of functions for computing cost and building cost matrix
--------------------------------------------------------------------------------------------------------------------
'''

# function for getting euclidiean distance
def euc_dist(node1, node2):
    euc_dist = round(math.sqrt((node2.x_coord - node1.x_coord) ** 2 + (node2.y_coord - node1.y_coord) ** 2)) # Euclidean Distance
    return euc_dist

# function that generates cost matrix based on given nodes
def build_cost_matrix(list_of_nodes, cost_matrix):
    for row_node in list_of_nodes:
        for column_node in list_of_nodes:
            cost_matrix[row_node.node_id - 1][column_node.node_id - 1] = euc_dist(row_node, column_node)

# function that reads nodes from input file and builds cost matrix              
def get_cost_matrix(args):
    list_of_nodes = [] # list to hold all nodes provided by the user

    # read the coordinates line by line and enter them as nodes in a list
    with open(args.input_coordinates_file) as cf:
        line = cf.readline().rstrip('\n')
        while line:
            node_id, x_coord, y_coord = line.split()
            list_of_nodes.append(Node(int(node_id), float(x_coord), float(y_coord)))
            line = cf.readline().rstrip('\n')

    # initilize  & build cost matrix (width and height are the size of given number of coordinets)
    cost_matrix = [[int for j in range(len(list_of_nodes))] for i in range(len(list_of_nodes))]
    build_cost_matrix(list_of_nodes, cost_matrix)
    return cost_matrix

'''
--------------------------------------------------------------------------------------------------------------------
Below are functions that assist the held-karp algorthim in its tasks
--------------------------------------------------------------------------------------------------------------------
'''

# function that utilizes Held-Karp algorithim logic to get the best cost of the given key
# for each subproblem. Utilizing Dynamic Programming principles, any subproblems
# that are computed in the process of computing larger subproblems are stored
# to speed up the run time.
# NOTE: for the final original problem this method will comute the entire optimol path and cost
# and return it.
# NOTE: some of the comments below are written to show what potentiol
# examples may be for that situation.
# NOTE: based on testing done with various inputs, the current optimol cost and path converges
# to a cost that is equal to or very close to the true optimol cost
def get_best_cost(dictOfElements, key, cost_matrix, curr_opt_PATH):
    global interm_path_list
    if key in dictOfElements.keys(): # CASE: if the subproblem is already stored
        from_via = key[1]
        value_list_pair = dictOfElements.get(key)
        return [value_list_pair[0], from_via]

    if len(key[1]) == 1:
        if key[1][0] == 1: # CASE: [2, {1}] or [3, {1}] etc.
            return dictOfElements[key] # it is 0 since the cost will be calculated from start node
        else: # CASE: for [2, {3}] you get [3, {1}]...
            via_key = (key[1][0], (1,))
            from_via = key[1][0] #gets 3
            sub_value = dictOfElements.get(via_key) #gets [3, {1}]
            part_cost = cost_matrix[key[0] - 1][from_via - 1] #
            if key not in dictOfElements.keys(): # if key doesnt exist but is being done for a larger subset key, you can save it to not need to recalculate
                dictOfElements[key] = [part_cost + sub_value[0], from_via]
                #TEST: print(f'calculating for key {key} of cost {part_cost} with sub key {via_key} of value {sub_value} giving {[part_cost + sub_value[0], from_via]}')
            return [part_cost + sub_value[0], from_via] #...returns [2, {3}] whichc is [2, {3}] + [3,{1}]
    else: # CASE: given [2, {3, 4, 5}] you...
        cost_value_pairs = []
        orig_to_node = key[0] #...is 2

        for curr_from_index in range(len(key[1])): # calculates the first part whihc is [2, {3}] or [2, {4}] or...
            part_subset = (key[1][curr_from_index]) #...gets the 3
            part_result = cost_matrix[orig_to_node - 1][part_subset - 1] # gets [2, {3}] from cost matrix

            sub_from_subset = list(key[1])
            sub_to = sub_from_subset.pop(curr_from_index) #...makes {3,4,5...} into {4,5,...} and pulls out 3
            sub_key = (sub_to, tuple(sub_from_subset)) #makes sub key [3, {4,5,...}]
            sub_result = get_best_cost(dictOfElements, sub_key, cost_matrix, curr_opt_PATH) #gets sub result [cost, from] of [3, {4, 5,...}]
            
            # saves all intermediate subproblems
            interm_path_list.insert(len(interm_path_list), sub_key)

             #TEST: print(f'calculating for key {key} cost of part key  is {part_result}, sub_result of sub_key is {sub_key} is {sub_result}')

            cost_value_pairs.append( [part_result + sub_result[0],  part_subset] )
        
        opt_value = min(cost_value_pairs, key = lambda t: t[0]) #picks the cheapest cost and respective parent pair
        
        # grabs the sub path that is part of the optimol solution
        curr_opt_PATH.append(opt_value[1])
        for next_step in interm_path_list:
            if next_step[0] == opt_value[1]:
                for i in next_step[1]: 
                    curr_opt_PATH.append(i)
                break
        interm_path_list.clear()

        return opt_value

# funtion that rearranges the optimol path based on the newly found optimality of diffrent nodes' positons
# NOTE: This function also makes sure the current optimol cost is improving and does not allow for the 
# cost to increase. During all iterations it will revaluate any changes done to the optimol path and 
# get the latest cost. If the changes result in an increase in cost, the changes will be reversed
# NOTE: \/ --- IF THIS VARIATION OF HELD KARP ISN'T PATENTED IT'S MINE!!! --- \/
def get_opt_path_and_cost(cost_matrix, opt_PATH, curr_opt_PATH, weighted_opt_PATH, max_range, start_node_id, curr_opt_cost, key=None, cost_from_pair=None):
    # NOTE: this if cluase is called when the final solution has been computed all other cases are managed in the else clause
    if len(curr_opt_PATH) == max_range-2: # CASE: entire optimol path has been computed and is in curr_opt_PATH
        opt_PATH = curr_opt_PATH[:]
        opt_PATH.append(start_node_id)
        opt_PATH.insert(0, start_node_id)
        return opt_PATH, weighted_opt_PATH, None
    else:  # CASE: portion of the optimol path has been computed curr_opt_PATH
        to_node_id = key[0]
        from_node_id = cost_from_pair[1]

        to_index = opt_PATH.index(to_node_id)
        from_index = opt_PATH.index(from_node_id)
        
        #TEST: print(f'to_node is {to_node_id} with index {to_index}, from_node is {from_node_id} with index {from_index}')
        old_cost_fromNode_pair = None # cost-pair of any weight that is removed (used in case of undoing changes)
        opt_path_updated = 0 # boolean for any changes made (digit represent which type of edit happened in case of reverse)

        if weighted_opt_PATH[to_index][1] == None: # CASE: if the id-weight of the to_node pair have None for weight...
            weighted_opt_PATH[to_index][1] = cost_from_pair

        elif weighted_opt_PATH[to_index][1] != None and weighted_opt_PATH[from_index][1] != None: # CASE: if the id-weight pairis not None for to or from nodes...
            old_cost_fromNode_pair = weighted_opt_PATH[to_index][1]
            if weighted_opt_PATH[to_index][1][0] >= cost_from_pair[0]:
                if weighted_opt_PATH[to_index][1][1] != key[1]: # CASE: [4, {1482, 2}] and cost_from_pair: [444, 3]
                    if to_index == from_index - 1 or to_index == from_index + 1: #CASE: if the to_node is right before from_node --> swap to and from node positons
                        opt_path_updated = 1
                        opt_PATH[to_index], opt_PATH[from_index] = opt_PATH[from_index], opt_PATH[to_index]
                        weighted_opt_PATH[to_index][1] = cost_from_pair
                        weighted_opt_PATH[to_index], weighted_opt_PATH[from_index] = weighted_opt_PATH[from_index], weighted_opt_PATH[to_index]
                    else: #CASE: if the to_node is before from_node but not right before (e.g: incorrect ordering) --> place to_node right before from_node
                        opt_path_updated = 2
                        weighted_opt_PATH[to_index][1] = cost_from_pair
                        opt_to_pair = opt_PATH.pop(to_index)
                        weighted_to_pair = weighted_opt_PATH.pop(to_index)
                        
                        opt_PATH.insert(from_index, opt_to_pair)
                        weighted_opt_PATH.insert(from_index, weighted_to_pair)       
                else:
                    weighted_opt_PATH[to_index][1] = cost_from_pair
            else: # CASE: [4, {1482, 2}] and cost_from_pair: [444, 2] (you have to update for bigger problems where the from_node list in the key are longer)
                weighted_opt_PATH[to_index][1] = cost_from_pair 
    
        #updates the current best cost based on any changes done to the opitmol path or if there is no known optimol cost
        if opt_path_updated != 0 or curr_opt_cost == None:
            new_opt_cost = 0
            for i in range(len(opt_PATH)-1):
                new_opt_cost += cost_matrix[opt_PATH[i] - 1][opt_PATH[i+1] - 1]
            
            if curr_opt_cost == None: # CASE: no current optimol cost is stored
                curr_opt_cost = new_opt_cost
            else:
                if curr_opt_cost >= new_opt_cost: # CASE: changes improve best cost or have no effect at all
                    curr_opt_cost = new_opt_cost
                else: # Below code will reverse any swapping due to a increase in the best cost
                    if opt_path_updated == 1:
                        opt_PATH[from_index], opt_PATH[to_index] = opt_PATH[to_index], opt_PATH[from_index]
                        weighted_opt_PATH[to_index][1] = old_cost_fromNode_pair
                        weighted_opt_PATH[from_index], weighted_opt_PATH[to_index] = weighted_opt_PATH[to_index], weighted_opt_PATH[from_index]
                    
                    
                    elif opt_path_updated == 2:
                        opt_to_pair = opt_PATH.pop(from_index)
                        weighted_to_pair = weighted_opt_PATH.pop(from_index)
                        opt_PATH.insert(to_index, opt_to_pair)
                        weighted_opt_PATH.insert(to_index, weighted_to_pair)

                        weighted_opt_PATH[to_index][1] = old_cost_fromNode_pair


    return opt_PATH, weighted_opt_PATH, curr_opt_cost

'''
--------------------------------------------------------------------------------------------------------------------
Below is held-karp algorithim logic code
--------------------------------------------------------------------------------------------------------------------
'''
# function that utilizes the held-karp algorithim design 
# NOTE: This is NOT pure held-karp as it utilizes custom function get_opt_path_and_cost()  
# to speed up calculation of intermediate iterations. These result in
# the intermediate calculations converging to the the true optimol cost and path.
def held_karp_variant(cost_matrix, start_node_id, opt_PATH, max_range):
    curr_opt_PATH = []
    curr_opt_cost = None
    traversal_order = {} # will hold all of the diffrent scenarios [3,{2}] or [3,{2,4,5,6...}] ...
    # TEST: print(f'original opt path: {opt_PATH}')
    # stores all costs from start node to all other nodes
    for to_node_id in range(2, max_range): #NOTE: KEY STRUCTURE: [to_node, { from_node(s) }] = [cost, from_parent_node_id]
        key = (to_node_id, tuple([start_node_id]))
        cost = cost_matrix[to_node_id - 1][start_node_id - 1]
        traversal_order[key] = [cost, 1]
    
    # weighted_opt_PATH is included to be the "memory" of why id's are ordered the way the are in the optimol path
    # each index is: [node_id, [cheapest cost cost-from pair]]
    # NOTE: any changes done on the weighted_opt_PATH will be reflected in opt_PATH and vice versa
    weighted_opt_PATH = []
    for i in opt_PATH:
        id_and_weight = [i, None] 
        weighted_opt_PATH.append(id_and_weight)

    # this loop works for all subproblems starting with the simpelest after the paths directly form the starting node
    # NOTE: the first if clause immideatly below is to skip any subproblems that the Held-Karp algorithim does not deal with
    # as they are invalid arguments.
    for subset_size in range(1, max_range - 2):
        for subset in itertools.combinations(range(2, max_range), subset_size):
            for to_node_id in range(1, max_range):
                if to_node_id == 1 or to_node_id in subset: # CASE: if the to_node is the staring node or is in the subset
                    continue
                else:
                    key = (to_node_id, tuple(subset)) # NOTE: KEY STRUCTURE: [to_node, ("potentiol" from_node(s))] = [cost, from_node_id]
                    if len(key[1]) == 1 and key[1][0] == 1: # CASE: if the from_subset is one value such as [2,(3)] or [2,[1]] then just pull it from cost matrix
                        continue
                    else: # CASE: if the from_subset is like [2,[3,4...]] 
                        cost_from_pair = get_best_cost(traversal_order, key, cost_matrix, curr_opt_PATH)
                        opt_PATH, weighted_opt_PATH, curr_opt_cost = get_opt_path_and_cost(cost_matrix, opt_PATH, curr_opt_PATH, weighted_opt_PATH, max_range, start_node_id, curr_opt_cost, key, cost_from_pair)
                        curr_opt_PATH.clear()

                        traversal_order[key] = cost_from_pair

                        # NOTE: (TESTING PURPOSES) uncomment below section to see changes done at each iteration
                        # print(f'key {key} now equals: {cost_from_pair}')
                        # print(f'current optimol cost is: {curr_opt_cost}, current optimol path is: {opt_PATH}')
                        # print(f'weighted opt path is: {weighted_opt_PATH}')
                        # print("------TOTAL TIME: %s seconds ------" % (MASTER_CLOCK.perf_counter()  - start_time))
                        # print()


                        check_timeout(args) # check if program should timeout

                        # clears the output file
                        open(args.output_tour_file, 'w').close() # NOTE: COMMENT FOR ALL INTERMEDIATE OPT. PATH & COST
                        # writes the new current optimol cost and optmol path
                        with open(args.output_tour_file, 'a') as cf:
                            cf.write("Current optimol cost: " + str(curr_opt_cost) + "\nCurrent optimol path: [")
                            for oi in opt_PATH:
                                cf.write(str(oi) + ', ')
                            cf.write(']\n')

    # calculate final optimol solution:
    final_solution_subset = [item for item in range(2, max_range)]
    opt_key = (1, tuple(final_solution_subset))
    opt_result = get_best_cost(traversal_order, opt_key, cost_matrix, curr_opt_PATH) # NOTE: the True is to enable the aggregate collection of the optimol path
    
    # NOTE: below, opt_cost does not have to be retrived from get_opt_path_and_cost() as it is already calculated via get_total_cost()
    opt_PATH, weighted_opt_PATH, _ = get_opt_path_and_cost(cost_matrix, opt_PATH, curr_opt_PATH, weighted_opt_PATH, max_range, start_node_id, curr_opt_cost, key, cost_from_pair)
    curr_opt_PATH.clear()

    # clears the output file
    open(args.output_tour_file, 'w').close() # NOTE: COMMENT FOR ALL INTERMEDIATE OPT. PATH & COST
    # writes the final optimol cost and path to the output file
    with open(args.output_tour_file, 'a') as cf:
        cf.write("FINAL OPTIMOL COST: " + str(opt_result[0]) + "\nFINAL OPTIMOL PATH: [")
        for oi in opt_PATH:
            cf.write(str(oi) + ', ')
        cf.write(']\n')

'''
--------------------------------------------------------------------------------------------------------------------
Below is the testing code
--------------------------------------------------------------------------------------------------------------------
'''
# --- main clause ---
if __name__ == '__main__':
    start_time = MASTER_CLOCK.perf_counter() # initilizes start time to check for termination
    interm_path_list = [] # global list that assists opt_PATH in getting the optimol path each iteration when opt_PATH is used
    start_node_id = 1 # NOTE: program assumes start vertex is node 1

    args = get_user_args() # get user arguments (input file of nodes, output file, and time duration)
    check_timeout(args) # check if program should timeout
    cost_matrix = get_cost_matrix(args) # get the cost matrix (contains all edge costs between nodes)
    check_timeout(args) # check if program should timeout

    max_range = len(cost_matrix[0]) + 1 # global range that will utilized in many loops in algorithim methods
    opt_PATH = [item for item in range(1, max_range)] # global list to store the optimol path, this is only called upon the last run to get the full optimol path
    opt_PATH.append(start_node_id)

    held_karp_variant(cost_matrix, start_node_id, opt_PATH, max_range) # execution of the algorithim begins...




        