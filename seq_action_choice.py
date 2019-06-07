# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 08:52:57 2019

Classical Conditioning and Reinforcement Learning

Sequential Action Choice

reward is based on a sequence of actions

0----B----5
     |    2----C----0
     |         |
     +----A----+
          ^
          |
        enter

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

class Action_value_vector(object):
    def __init__(self, N):
        self.N = N
        self.m = [0. for n in range(N)]

beta = 0.5
epsilon = 0.5

class Maze_node(object):
    def __init__(self, name, reward_vector, is_absorbing=False):
        self.name = name
        self.reward_vector = reward_vector
        self.is_absorbing = is_absorbing

        """ The predicted future reward: v(u) = w(u) """
        self.weight = 0.
        self.weight_history = []

        self.left_node = None
        self.right_node = None

    def immediate_reward(self, action):
        if action == 'L':
            r = self.reward_vector[0]

        if action == 'R':
            r = self.reward_vector[1]

        return r

    def modify_weight(self, action):
        print(self.name)
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

    def next_location(self, action):
        if action == 'L':
            node = self.left_node
        if action == 'R':
            node = self.right_node

        return node

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

"""
The softmax distribution:
    P[L] = exp(Beta * m_L)/(exp(Beta * m_L) + exp(Beta * m_R))
    P[R] = exp(Beta * m_L)/(exp(Beta * m_R) + exp(Beta * m_R))
"""
def softmax(a, action_values):
    Z = 0.
    for n in range(action_values.N):
        Z += math.exp(beta * action_values.m[n]) 

    return math.exp(beta * action_values.m[a])/Z

def main():
    n_trials = 60
    maze = Maze(start_location=Maze_node('A', [0., 0.]),
                end_locations=[
                        Maze_node('B', [0., 5.]),
                        Maze_node('C', [2., 0.]),
                    ])

    maze.start_location.left_node = maze.end_locations[0]
    maze.start_location.right_node = maze.end_locations[1]
    for node in maze.end_locations:
        node.left_node = Maze_node('', [0., 0.], is_absorbing=True)
        node.right_node = Maze_node('', [0., 0.], is_absorbing=True)

    maze.policy_evaluation(n_trials)

    pylab.plot(maze.start_location.weight_history[:int(n_trials/2)])
    pylab.plot(maze.end_locations[0].weight_history[:int(n_trials/2)])
    pylab.plot(maze.end_locations[1].weight_history[:int(n_trials/2)])
    pylab.ylim(0., 5.)
    pylab.legend(('w(A)','w(B)','w(C)',), loc='upper right')

if __name__ == "__main__":
    main()
