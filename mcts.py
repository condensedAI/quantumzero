from __future__ import division
import collections
import random
import numpy as np
import sys
import math
import ast
from collections import Counter

class Node:
    def __init__(self, value, children_values, parent=None, level=0, position=None, struct=None, v=1, w=0.0,
                 children={}, min_range=float(math.pow(10, 10)), max_range=0.0):

        self.value = value
        self.parent = parent

        # Depth in tree
        self.level = level

        # The actual state for this node
        self.struct = struct
        # The current index into the struct, in case we are not searching in
        # a standard order
        self.position = position

        # Visit count and score, for UCT
        self.v = v
        self.w = w

        self.min_range = min_range
        self.max_range = max_range

        self.children = children
        self.children_values = children_values

    def has_children(self):
        return len(self.children) != 0

    def has_all_children(self):
        return len(self.children) == len(self.children_values)

    def select_best_child(self):
        if len(self.children) == 0:
            raise RuntimeError("Trying to select child on node without children")

        #print("Selecting best child (out of ", len(self.children), ") for node ", self.value, self.children )
        #print([child for child in self.children])
        scores = np.array([[idx, child.w / child.v + 2*np.sqrt(2 * np.log(self.v) / child.v)] for (idx,child) in self.children.items()])
        maxscore = np.argmax(scores[:,1])
        return self.children[scores[maxscore][0]]

    def select(self, max_flag, ucb_mean):

        #print("Selecting from node %s"%self.value)
        # If this node has not yet fully been expanded, we stop the recursion here
        if not self.has_all_children():
            #print("Node %s does not yet have all children"%self.value)
            return self

        # If it has all children expanded, make a selection (using UCT) on those
        # and recursively search for the next.
        c = 2
        u_scores = {}
        for child in iter(self.children.values()):
            if ucb_mean:
                u_scores[child.value] = child.w / child.v + (c if max_flag else -c) * np.sqrt( 2 * np.log(self.v) / child.v)
            else:
                u_scores[child.value] = child.max_range + (c * (np.sqrt((2 * po.log(self.v)) / child.v)))

        if max_flag:
            idxs = [i for i, x in iter(u_scores.items()) if x == max(iter(u_scores.values()))]
        else:
            idxs = [i for i, x in iter(u_scores.items()) if x == min(iter(u_scores.values()))]
        idx = np.random.choice(idxs)

        # Recursively search from the selected child node
        return self.children[idx].select(max_flag, ucb_mean)

    def select_v2(self, position, num_children_to_expand):
        if not self.has_all_children():
            expanded = self.expand(position, num_children_to_expand)
            return expanded
        else:
            return [self.select_best_child()]

    def back_prop(self, reward):
        # Back propagate through this node
        self.v += 1  # Increase visit count
        self.w += reward  # Increase value

        # update max and min rewards seen
        if reward > self.max_range:
            self.max_range = reward
        if reward < self.min_range:
            self.min_range = reward

        # Continue back propagating to parent
        if self.parent is not None:
            return self.parent.back_prop(reward)

    def adjust_c(self, adjust_value):
        self.adjust_val += adjust_value
        if self.parent is not None:
            return self.parent.adjust_c(adjust_value)

    def expand(self, position, num_children_to_expand):
        avl_child_values = list(set(self.children_values) - set(self.children.keys()))

        # If we are asked to expand more children than there are left, we cap the number
        no_chosen_values = np.min([len(avl_child_values), num_children_to_expand])

        # Pick them at random
        chosen_values = np.random.choice(avl_child_values, no_chosen_values, replace=False)

        expanded = []
        for child_value in chosen_values:
            child_struct = self.struct[:]
            child_struct[position] = child_value

            # Add this expanded node to the list of children
            self.children[child_value] = Node(value=child_value,
                                                children_values=self.children_values,
                                                parent=self,
                                                level=self.level + 1,
                                                position=position,
                                                struct=child_struct,
                                                children={})

            # Add this node to the list
            expanded.append(self.children[child_value])

        return expanded

    def get_info(self):
        nodes = 1
        visits = self.v
        depth = self.level

        if self.has_children():
            for child in iter(self.children.values()):
                if child is not None:
                    x, y, z = child.get_info()
                    nodes +=x
                    visits += y
                    if z > depth:
                        depth = z
        return nodes, visits, depth


class Tree:
    def __init__(self, get_reward, positions_order="reverse", max_flag=True, expand_children=1,
                 space=None, candidate_pool_size=None, no_positions=None, atom_types=None, atom_const=None, play_out=1, play_out_selection="best", ucb="mean"):

        if space is None:
            self.space=None
            if (no_positions is None) or (atom_types is None):
                raise ValueError("no_positions and atom_types should not be None")
            else:
                self.no_positions = no_positions
                self.atom_types = atom_types
                self.atom_const = atom_const
                self.candidate_pool_size = candidate_pool_size
        else:
            self.space = space.copy()
            self.one_hot_space = self.one_hot_encode(self.space)
            self.no_positions = space.shape[1]
            self.atom_types = np.unique(space)

        if positions_order == "direct":
            self.positions_order = list(range(self.no_positions))
        elif positions_order == "reverse":
            self.positions_order = list(range(self.no_positions))[::-1]
        elif positions_order == "shuffle":
            self.positions_order = random.sample(list(range(self.no_positions)), self.no_positions)
        elif isinstance(positions_order, list):
            self.positions_order = positions_order
        else:
            sys.exit("Please specify positions order as a list")

        # If we want to always expand all children, instead of just 1, update
        if expand_children == "all":
            self.expand_children = len(self.atom_types)

        elif isinstance(expand_children, int):
            if (expand_children > len(self.atom_types)) or (expand_children == 0):
                sys.exit("Please choose appropriate number of children to expand")
            else:
                self.expand_children = expand_children

        if play_out_selection == "best":
            self.play_out_selection_mean = False
        elif play_out_selection =="mean":
            self.play_out_selection_mean = True
        else:
            raise ValueError("Please set play_out_selection to either mean or best")

        if ucb == "best":
            self.ucb_mean = False
        elif ucb == "mean":
            self.ucb_mean = True
        else:
            raise ValueError("Please set ucb to either mean or best")

        self.chkd_candidates = collections.OrderedDict()
        self.max_flag = max_flag
        self.acc_threshold = 0.1
        self.get_reward = get_reward

        # Create root node
        self.root = Node(value='R', children_values=self.atom_types, struct=[None]*self.no_positions)
        self.play_out = play_out

    def _enumerate_cand(self, struct, size):
        # Make a copy
        structure = struct[:]

        chosen_candidates = []
        if self.atom_const is not None:
            for value_id in range(len(self.atom_types)):
                if structure.count(self.atom_types[value_id]) > self.atom_const[value_id]:
                    return chosen_candidates
            for pout in range(size):
                cand = structure[:]
                for value_id in range(len(self.atom_types)):
                    diff = self.atom_const[value_id] - cand.count(self.atom_types[value_id])
                    if diff != 0:
                        avl_pos = [i for i, x in enumerate(cand) if x is None]
                        to_fill_pos = np.random.choice(avl_pos, diff, replace=False)
                        for pos in to_fill_pos:
                            cand[pos] = self.atom_types[value_id]
                chosen_candidates.append(cand)
        else:
            for play_out_number in range(size):
                # Copy current set of components
                cand = structure[:]

                # Get empty positions
                avl_pos = [i for i, x in enumerate(cand) if x is None]
                # And for each, fill in a random choice of action
                for pos in avl_pos:
                    cand[pos] = np.random.choice(self.atom_types)

                # Add to the list
                chosen_candidates.append(cand)
        return chosen_candidates

    def one_hot_encode(self,space):
        no_atoms=len(self.atom_types)
        new_space = np.empty((space.shape[0], space.shape[1], no_atoms), dtype=int)
        for at_ind, at in enumerate(self.atom_types):
            one_hot = np.zeros(no_atoms, dtype=int)
            one_hot[at_ind] = 1
            new_space[space == at] = one_hot
        return new_space.reshape(space.shape[0],space.shape[1]*no_atoms)

    def _simulate(self, struct, lvl):
        if self.space is None:
            return self._enumerate_cand(struct, self.play_out)

    def rollout(self, start_node, num_rollouts):
        candidates = self._enumerate_cand(start_node.struct, num_rollouts)

        rewards = []
        for candidate in candidates:
            if str(candidate) not in self.chkd_candidates.keys():
                self.chkd_candidates[str(candidate)] = self.get_reward(candidate)
            rewards.append(self.chkd_candidates[str(candidate)])

        return np.mean(rewards)

    def get_best_next_node(self, start_node, position, num_simulations):
        """
        Perform num_simulations of tree expansion
        """
        for i in range(num_simulations):
            # Select
            current = start_node.select_v2(position, self.expand_children)
            
            # Get rewards for each expanded child
            for c in current:
                reward = self.rollout(c, self.play_out)
                
                # Back propagate the reward
                c.back_prop(reward)
            
        # Return best
        return start_node.select_best_child()

    def find_best_candidate(self, num_simulations=100):
        current = self.root
        for m in range(self.no_positions):
            # Perform num_simulations to get the next best node
            next_node = self.get_best_next_node(current, m, num_simulations)
            current = next_node
        return current.struct[:], self.get_reward(current.struct[:])

    def search(self, no_candidates=None, display=True):
        prev_len = 0
        prev_current = None
        round_no = 1

        if no_candidates is None:
            raise ValueError("Please specify no_candidates")
            return

        fidelity = 0
        fidelity_old = 0
        self.chkd_candidates = {}
        #modified convergence condition: if fon num_stuck_iter the fidelity does not improve more than convengence_tol
        #the mcts is going to stop

        convergence_tol = 0.001 #TO BE MODIFIED IN FUTURE VERSION
        num_am_I_stuck=0
        num_stuck_iter = 20
        # Search until we hit a good fidelity or until we've checked more candidates than asked for
        while num_am_I_stuck < num_stuck_iter and len(self.chkd_candidates) < no_candidates:

            # Select new node (traverse tree from root until we hit a non-fully expanded node)
            current = self.root.select(self.max_flag, self.ucb_mean)

            # Check if we have reached the bottom layer
            if current.level == self.no_positions:
                # See if we've already visited this node,
                # and either use it's already computed reward or
                # compute and store it
                struct = current.struct[:]
                if str(struct) not in self.chkd_candidates.keys():
                    self.chkd_candidates[str(struct)] = self.get_reward(struct)
                e = self.chkd_candidates[str(struct)]

                # Update the tree
                current.back_prop(e)

            else:
                # Pick the next frequency component to optimize
                position = self.positions_order[current.level]

                # Expand node
                try_children = current.expand(position, self.expand_children)

                for try_child in try_children:

                    # Random rollout from here, meaning run until bottom
                    all_struct = self._simulate(try_child.struct, try_child.level)

                    rewards = []
                    for struct in all_struct:
                        # Add reward to list if it is not there yet
                        if str(struct) not in self.chkd_candidates.keys():
                            self.chkd_candidates[str(struct)] = self.get_reward(struct)
                            round_no+=1
                        e = self.chkd_candidates[str(struct)]
                        rewards.append(e)

                    # If there were new cases
                    if len(rewards)!=0:
                        if self.play_out_selection_mean:
                            best_e = np.mean(rewards)
                        else:
                            best_e = max(rewards) if self.max_flag else min(rewards)

                        # Back propagate
                        try_child.back_prop(best_e)

            # Adjust C constant for UCT
            # if (current == prev_current) and (len(self.chkd_candidates) == prev_len):
            #     adjust_val = (no_candidates-len(self.chkd_candidates))/no_candidates
            #     if adjust_val < self.acc_threshold:
            #         adjust_val = self.acc_threshold
            #     current.adjust_c(adjust_val)

            # Keep track of how many candidates we have checked
            prev_len = len(self.chkd_candidates)
            # Keep track of previous node
            prev_current = current

            # Get best candidate from all checked ones so far
            optimal_fx = max(iter(self.chkd_candidates.values()))
            DS = [(ast.literal_eval(x), v) for (x,v) in self.chkd_candidates.items()]
            optimal_candidate = [k for (k, v) in DS if v == optimal_fx]
            fidelity = optimal_fx

            #round_no += 1
            if np.abs(fidelity-fidelity_old) < convergence_tol:
                num_am_I_stuck +=1
            fidelity_old=np.copy(fidelity)

        #self.result.format(no_candidates=no_candidates, chkd_candidates=self.chkd_candidates, max_flag=self.max_flag)
        #self.result.no_nodes, visits, self.result.max_depth_reached = self.root.get_info()
        #self.result.avg_node_visit = visits / self.result.no_nodes
        print("Num rounds: ", round_no)
        return optimal_candidate, fidelity, round_no 
