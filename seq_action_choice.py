# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 08:52:57 2019

Theoretical Neuroscience by Peter Dayan and L. F. Abbott

Chapter 9 Classical Conditioning and Reinforcement Learning

In classic Pavlovian experiment, the food is the unconditioned stimulus and
the bell is the conditioned stimulus (it only works under the condition that
there has been prior learning). This chapter refers to stimuli, rewards and
expecation of reward.

How does one predict reward?! The Rescorla-Wagner Rule.

9.4 Sequential Action Choice

Reward is based on a sequence of actions

0----B----5
     |        2----C----0
     |             |
     +------A------+
            ^
            |
          Enter

Goal: To optimize the total reward, i.e., moving left at A and right at B.
Method: Policy iteration; the reiforcement learning version of policy iteration
mantains and improves a stochastic policy, which determines the action at each
decision point (i.e., left or right turns at A, B, or C) through action values
and the softmax distribution:
    P[a] = exp(Beta * m_a)/Sum_a' exp(Beta * m_a'),
where a' = left or right.

Policy iteration: involves two elements.
    1] The _critic_ uses temporal difference learning to estimate the total
    future reward that's expected when starting from A, B, or C, when the
    current policy is followed.
    2] The _actor_ mantains and improves the policy. Adjustment of the action
    values at A is beased on predictions of the expected future reward associated
    with points B and C provided by the critic.

@author: Efren A. Serra
"""

import math, pylab
import random

def Kronecker_delta(a, b):
    """The Kronecker delta function.
    
    Parameters
    ----------
    a : integer
    b : integer
    """
    res = 0
    if a == b:
        res = 1
    return res

def softmax(beta, action, node):
    """The softmax distribution.

        P[L] = exp(Beta * m_L)/(exp(Beta * m_L) + exp(Beta * m_R))
        P[R] = exp(Beta * m_L)/(exp(Beta * m_R) + exp(Beta * m_R))

    Parameters
    ----------
    beta : float
    action : list
    node : @Maze_node

    Raises
    ------
    """

    Z = 0.
    for a in node.action_values.actions:
        Z += math.exp(beta * node.action_values.m[a])

    return math.exp(beta * node.action_values.m[action])/Z

class Action_value_vector(object):
    """The Action_value_vector class.

    Attributes
    ----------
    actions : list
        a list of string to represent actions

    m : dictionary
        a dictionary representing the action values vector per action

    Methods
    -------
    """

    def __init__(self, actions, values):
        """
        Parameters
        ----------
        actions : list
        values : list
        
        Raises
        ------
        """

        self.actions = actions
        self.m = dict(map(lambda k,v: (k,v), actions, values))

        empty_values = [[] for n in range(len(actions))]
        self.P = dict(map(lambda k,v: (k,v), actions, empty_values))

    def __repr__(self):
        """
        Parameters
        ----------
        self : object
        
        Raises
        ------
        """
        return "<m: %s>"%(self.m)

beta = 1.0
epsilon = 0.5

"""
The Maze_node class:
"""
class Maze_node(object):
    def __init__(self, name, reward_vector, action_values, is_absorbing=False):
        self.name = name
        self.reward_vector = reward_vector
        self.action_values = action_values
        self.is_absorbing = is_absorbing

        """ The predicted future reward: v(u) = w(u) """
        self.weight = 0.
        self.weight_history = []

        self.left_node = None
        self.right_node = None

    def __repr__(self):
        return "<name: %s weight: %0.2f m: %s>"%(self.name, self.weight, self.action_values)

    def immediate_reward(self, action):
        if action == 'L':
            r = self.reward_vector[0]

        if action == 'R':
            r = self.reward_vector[1]

        return r

    def modify_weight(self, action):
        r = self.immediate_reward(action)
        v_at_u = self.weight
        v_at_u_prime = 0.

        if action == 'L' and self.left_node:
            v_at_u_prime = self.left_node.weight

        if action == 'R' and self.right_node:
            v_at_u_prime = self.right_node.weight

        delta = r + (v_at_u_prime - v_at_u)

        self.weight = self.weight + epsilon * delta
        self.weight_history.append(self.weight)

        print(self)

    def modify_action_values(self, action):
        r = self.immediate_reward(action)
        v_at_u = self.weight
        v_at_u_prime = 0.

        if action == 'L' and self.left_node:
            v_at_u_prime = self.left_node.weight

        if action == 'R' and self.right_node:
            v_at_u_prime = self.right_node.weight

        delta = r + (v_at_u_prime - v_at_u)

        """ Policy improvement or actor learning rule """
        for a in self.action_values.actions:
            self.action_values.m[a] = \
            self.action_values.m[a] + \
            epsilon * (Kronecker_delta(action, a) - softmax(beta, a, self)) * delta
            self.action_values.P[a].append(softmax(beta, a, self))

        print(self)

    def next_location(self, action):
        if action == 'L':
            node = self.left_node
        if action == 'R':
            node = self.right_node

        return node

"""
The Maze class:
"""
class Maze(object):
    def __init__(self, start_location, end_locations):
        self.start_location = start_location
        self.end_locations = end_locations

    def policy_evaluation(self, max_trials):
        n = 0
        while n < max_trials:
            node = self.start_location
            action = random.choice(['R', 'L'])
            node.modify_weight(action)
            node = node.next_location(action)
            while not node.is_absorbing:
                action = random.choice(['L', 'R'])
                node.modify_weight(action)
                node = node.next_location(action)
            n += 1

    def policy_improvement(self, max_trials):
        n = 0
        while n < max_trials:
            node = self.start_location
            action = random.choice(['R', 'L'])
            node.modify_action_values(action)
            node = node.next_location(action)
            while not node.is_absorbing:
                action = random.choice(['L', 'R'])
                node.modify_action_values(action)
                node = node.next_location(action)
            n += 1


def main():
    n_trials = 50
    maze = Maze(start_location=Maze_node('A', [0., 0.],
                                         action_values=Action_value_vector(
                                                 actions=['L', 'R'],
                                                 values=[0., 0.],)), end_locations=[Maze_node('B',
                                                 [0., 5.], action_values=Action_value_vector(
                                                         actions=['L', 'R'],
                                                         values=[0., 0.],)),
                                                 Maze_node('C', [2., 0.], action_values=Action_value_vector(
                                                         actions=['L', 'R'],
                                                         values=[0., 0.],)),],)

    maze.start_location.left_node = maze.end_locations[0]
    maze.start_location.right_node = maze.end_locations[1]
    for node in maze.end_locations:
        node.left_node = Maze_node('', [0., 0.],
                                   action_values=None,
                                   is_absorbing=True)
        node.right_node = Maze_node('', [0., 0.],
                                    action_values=None,
                                    is_absorbing=True)

    maze.policy_evaluation(n_trials)

    pylab.figure(1)
    pylab.plot(maze.start_location.weight_history[:30])
    pylab.plot(maze.end_locations[0].weight_history[:30])
    pylab.plot(maze.end_locations[1].weight_history[:30])
    pylab.ylim(0., 5.)
    pylab.legend(('w(A)','w(B)','w(C)',), loc='upper right')

    maze.policy_improvement(2*n_trials)

    pylab.figure(2)
    pylab.plot(maze.start_location.action_values.P['L'][:])
    pylab.plot(maze.end_locations[0].action_values.P['L'][:])
    pylab.plot(maze.end_locations[1].action_values.P['L'][:])
    pylab.ylim(0., 1.0)
    pylab.legend(('P[L; u = A]','P[L; u = B]','P[L; u = C]',), loc='upper right')

    pylab.figure(3)
    pylab.plot(maze.start_location.action_values.P['R'][:])
    pylab.plot(maze.end_locations[0].action_values.P['R'][:])
    pylab.plot(maze.end_locations[1].action_values.P['R'][:])
    pylab.ylim(0., 1.0)
    pylab.legend(('P[R; u = A]','P[R; u = B]','P[R; u = C]',), loc='upper right')

"""
The main entry point
"""

if __name__ == "__main__":
    main()
