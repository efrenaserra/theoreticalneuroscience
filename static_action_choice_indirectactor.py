# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 08:52:57 2019

Theoretical Neuroscience by Peter Dayan and L. F. Abbott

Chapter 9 Classical Conditioning and Reinforcement Learning

9.3 Static Action Choice

The following algorithm implements an example of static action choice, where
bees forage among flowers in search of nectar, as described in Section 9.3 of
Chapter 9. This algorithm implements the indirect actor method for solving the
bees foraging task. In this method, the bees learn to estimate the expected nectar 
volumes provided by each flower by using a delta rule.

@author: Efren A. Serra
"""

import math, matplotlib.pyplot as plt
import random
from enum import Enum, auto

class Color(Enum):
    b = auto()
    """Blue"""

    y = auto()
    """Yellow"""

class Location(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __repr__(self):
        return "<x: %d, y: %d>"%(self.x, self.y)

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

def softmax(beta, action_values, action) -> float:
    """The softmax distribution.
        P[b] = exp(Beta * m_b)/(exp(Beta * m_b) + exp(Beta * m_y))
        P[y] = exp(Beta * m_y)/(exp(Beta * m_b) + exp(Beta * m_y))

    Parameters
    ----------
    beta : float
    action_values : list of Action_value
    action: Action_value

    Raises
    ------
    """

    Z = 0.
    for a in action_values.actions:
        Z += math.exp(beta * action_values.m[a])

    return math.exp(beta * action_values.m[action])/Z

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

    def __repr__(self):
        """
        Parameters
        ----------
        self : object
        
        Raises
        ------
        """
        return "<m: %s>"%(self.m)

Beta = 1.0
Epsilon = 0.1

"""
The Flower class:
"""
class Flower(object):
    def __init__(self, color: Color, location: int, nectar_volume: float) -> None:
        self.color = color
        self.location = location
        self.nectar_volume = nectar_volume

    def __repr__(self):
        return "<color: %s location: %s nectar_volume: %f>"%(self.color, self.location, self.nectar_volume)

    def get_nectar_volume(self) -> float:
        return random.uniform(0., self.nectar_volume)

    def set_nectar_volume(self, new_nectar_volume: float) -> None:
        self.nectar_volume = new_nectar_volume

"""
The Garden class:
"""
class Garden(object):
    def __init__(self, colors: [Color], n_flowers_per_color) -> None:
        self.n_flowers_per_color = n_flowers_per_color
        self.the_flowers = dict(map(lambda k,v: (k,v), colors, [[] for n in range(self.n_flowers_per_color)]))

        for color in colors:
            for n in range(n_flowers_per_color):
                self.the_flowers[color].append(Flower(color, n, (1 if color is Color.b else 2)))

    def __repr__(self):
        return "<the_flowers: %s>"%(self.the_flowers)

    def get_flower(self, color: Color) -> Flower:
        return self.the_flowers[color][0]

"""
The Bee class:
"""
class Bee(object):
    def __init__(self, action_values: Action_value_vector, garden: Garden):
        self.action_values = action_values
        self.the_garden = garden
        self.the_flowers = self.the_garden.the_flowers

        """ Statistic variables """
        self.m_history = None
        self.sum_visits = None

    def __repr__(self):
        return "<m: %s>"%(self.action_values)

    def stochastic_policy(self) -> Color:
        """
        Compute P[mb] and P[|my]
        """
        P = dict(map(lambda k,v: (k,v), self.action_values.actions, [0.0]))
        for action in self.action_values.actions:
            P[action] = softmax(Beta, self.action_values, action)

        # Select action value with max probability
        static_action = None
        stochastic_probability = 0.
        for action in P:
            print('action: %s; P[%s]: %f'%(action,action,P[action]))
            if P[action] > stochastic_probability:
                stochastic_probability = P[action]
                static_action = action

        return static_action

    def update_action_values(self, t: int, flower: Flower):
        """
        Parameters:
            t : int - the simulation time step.
            flower: Flower : the flower the bee landed on.
        """
        action = flower.color
        r = flower.get_nectar_volume()
        delta = r - self.action_values.m[action]

        """ Indirect actor learning rule """
        self.action_values.m[action] = self.action_values.m[action] + (Epsilon * delta)

        """ Update statistics """
        for k in self.action_values.m.keys():
            if k is not action:
                self.m_history[k][t+1] = self.action_values.m[k] + \
                              softmax(Beta, self.action_values, k)
                self.sum_visits[k][t+1] = self.sum_visits[k][t]
            else:
                self.m_history[action][t+1] = self.action_values.m[action] + \
                      softmax(Beta, self.action_values, action)
                self.sum_visits[action][t+1] = self.sum_visits[action][t] + 1

    def policy_evaluation(self, n_flower_visits: int) -> None:
        N = math.ceil(self.the_garden.n_flowers_per_color/2)
        self.m_history = dict(map(lambda k,v: (k,v), self.action_values.actions, \
                                   [ [0. for n in range(n_flower_visits)] for n in range(len(self.action_values.actions)) ] ))
        self.sum_visits = dict(map(lambda k,v: (k,v), self.action_values.actions, \
                                   [ [0. for n in range(n_flower_visits)] for n in range(len(self.action_values.actions)) ] ))

        for n in range(0, int(n_flower_visits/2)):
            if random.randint(0, self.the_garden.n_flowers_per_color) <= N:
                action = self.stochastic_policy()
                flower = self.the_garden.get_flower(action)
            else:
                flower = self.the_garden.get_flower(random.choice([Color.b, Color.y]))
            self.update_action_values(n, flower)

        for action in self.action_values.actions:
            for i in range(self.the_garden.n_flowers_per_color):
                self.the_flowers[action][i].set_nectar_volume((2 if action is Color.b else 1))

        for n in range(int(n_flower_visits/2), n_flower_visits - 1):
            if random.randint(0, self.the_garden.n_flowers_per_color) <= N:
                action = self.stochastic_policy()
                flower = self.the_garden.get_flower(action)
            else:
                flower = self.the_garden.get_flower(random.choice([Color.b, Color.y]))
            self.update_action_values(n, flower)

def main():
    n_flower_visits: int = 200
    the_garden = Garden([Color.b, Color.y], 4)

    bee = Bee(action_values=Action_value_vector(actions=[Color.b, Color.y], values=[0., 0.]), \
              garden=the_garden)

    bee.policy_evaluation(n_flower_visits)

    plt.figure(1)
    plt.plot(bee.m_history[Color.b][:], label='$m_{b}$')
    plt.plot(bee.m_history[Color.y][:], label='$m_{y}$')
    plt.ylim((0, 3))
    plt.xlabel('visits to flowers')
    plt.ylabel('$m$', rotation='horizontal')
    plt.legend()

    plt.figure(2)
    plt.plot(bee.sum_visits[Color.b][:], label='$blue$')
    plt.plot(bee.sum_visits[Color.y][:], label='$yellow$')
    plt.xlabel('visits to flowers')
    plt.ylabel('$sum\ visits$')
    plt.legend()

"""
The main entry point
"""
if __name__ == "__main__":
    main()