# baseline_team.py
# ---------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# baseline_team.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import random
import time
import contest.util as util
from contest.capture_agents import CaptureAgent
from contest.game import Directions
from contest.util import nearest_point


#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='Hella', second='Aron', num_training=0):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --redOpts and --blueOpts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """
    return [eval(first)(first_index), eval(second)(second_index)]


##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
    """
    A base class for reflex agents that choose score-maximizing actions
    To check computation-time use the get_best_actions method
    """

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start_pos = None
        self.initial_food_num = None
        self.should_move_back = False

    def register_initial_state(self, game_state):
        self.start_pos = game_state.get_agent_position(self.index)
        self.initial_food_num = len(self.get_food(game_state).as_list())
        CaptureAgent.register_initial_state(self, game_state)

    def get_best_actions(self, actions, game_state):
        # start = time.time()
        values_of_actions = [self.evaluate(game_state, a) for a in actions]
        # print('eval time for agent %d: %.4f' % (self.index, time.time() - start))
        max_value = max(values_of_actions)
        best_actions = [a for a, v in zip(actions, values_of_actions) if v == max_value]
        return best_actions

    def check_if_should_move_back(self, game_state):
        num_returned_food = game_state.get_agent_state(self.index).num_returned
        num_food_carrying = game_state.get_agent_state(self.index).num_carrying
        return num_returned_food >= (self.initial_food_num - 2) or num_food_carrying >= 4

    def compute_action_value(self, action, game_state):
        successor = self.get_successor(game_state, action)
        pos_successor = successor.get_agent_position(self.index)
        successor_state = successor.get_agent_state(self.index)
        features = util.Counter()

        # compute fear of ghosts to stay away from them
        features = self.add_features_to_run_from_ghost(features, successor, successor_state, pos_successor)
        fear_of_ghosts = -13 * features["ghost_distance"] + -1000 * features["num_ghosts"]

        # compute dist to start
        dist_to_start = -1 * self.get_maze_distance(self.start_pos, pos_successor)

        action_value = dist_to_start + fear_of_ghosts
        return action_value

    def choose_action_for_moving_back(self, actions, game_state):
        best_action_value = -9999
        best_action = None
        for action in actions:
            action_value = self.compute_action_value(action, game_state)
            if action_value > best_action_value:
                best_action = action
                best_action_value = action_value
        return best_action

    def choose_action(self, game_state):
        """
        Picks among the actions with the highest Q(s,a).
        """
        actions = game_state.get_legal_actions(self.index)
        best_actions = self.get_best_actions(actions, game_state)

        self.should_move_back = self.check_if_should_move_back(game_state)
        if self.should_move_back:
            best_action = self.choose_action_for_moving_back(actions, game_state)
            return best_action
        return random.choice(best_actions)

    def get_successor(self, game_state, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearest_point(pos):
            # Only half a grid position was covered
            return successor.generate_successor(self.index, action)
        else:
            return successor

    def evaluate(self, game_state, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights

    def get_features(self, game_state, action):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        features['successor_score'] = self.get_score(successor)
        return features

    def add_features_to_improve_movement(self, features, action, game_state):
        if action == Directions.STOP:
            features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev:
            features['reverse'] = 1
        return features

    def add_features_to_run_from_ghost(self, features, successor, successor_state, successor_pos):
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        ghosts = [a for a in enemies if not a.is_pacman and a.get_position() is not None and a.scared_timer < 3]
        features['num_ghosts'] = len(ghosts)
        if len(ghosts) > 0 and successor_state.is_pacman:
            min_dist = min([self.get_maze_distance(successor_pos, a.get_position()) for a in ghosts])
            features['ghost_distance'] = 5 - min_dist
        return features

    def get_weights(self, game_state, action):
        """
        Normally, weights do not depend on the game state.  They can be either
        a counter or a dictionary.
        """
        return {'successor_score': 1.0}


class Hella(ReflexCaptureAgent):
    """
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
  """
    def get_location_of_enemy_capsule(self, successor, game_state):
        # if enemy has eaten capsule take any
        # if enemy not eaten capsule take first if blue secind if red
        #if game_state.data.capsule_eaten:
        #location_capsules = successor.get_capsules()
        return None

    def add_features_to_eat_food_or_capsule(self, features, successor, successor_pos):
        food_list = self.get_food(successor).as_list()
        # only add capsule_location if not eaten already
        # location_capsule = self.get_location_of_enemy_capsule(successor)
        features['num_of_food_left'] = -len(food_list)  # smaller if a lot of food around
        if len(food_list) > 0:
            min_dist = min([self.get_maze_distance(successor_pos, food) for food in food_list])
            features['distance_to_food'] = min_dist
        return features

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        successor_state = successor.get_agent_state(self.index)
        successor_pos = successor_state.get_position()

        # get features to run from ghost
        features = self.add_features_to_run_from_ghost(features, successor, successor_state, successor_pos)

        # get features to eat nearest food or power-capsule
        features = self.add_features_to_eat_food_or_capsule(features, successor, successor_pos)

        # get features to stop pac from freezing
        features = self.add_features_to_improve_movement(features, action, game_state)

        return features

    def get_weights(self, game_state, action):
        return {'num_of_food_left': 100,
                'distance_to_food': -1,
                'num_ghosts': -1000,
                'ghost_distance': -13,
                'stop': -100,
                'rev': -30}


class Aron(ReflexCaptureAgent):
    """
    A reflex agent that keeps its side Pacman-free. Again,
    this is to give you an idea of what a defensive agent
    could be like.  It is not the best or only way to make
    such an agent.
    """

    def add_feature_to_stay_defending(self, features, successor_state):
        features['on_defense'] = 1
        if successor_state.is_pacman:
            features['on_defense'] = 0
        return features

    def add_feature_to_kill_enemies(self, features, successor, successor_pos, game_state):
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        visible_invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        features['num_visible_invaders'] = len(visible_invaders)
        if len(visible_invaders) > 0:
            min_dists = min([self.get_maze_distance(successor_pos, a.get_position()) for a in visible_invaders])
            features['visible_invader_distance'] = min_dists
        # for invisible enemies
        dist = game_state.get_agent_distances()
        features['invisible_invader_distance'] = min(dist)
        return features

    def add_features_to_patrol_along_food(self, features, successor, successor_pos):
        food_list = self.get_food_you_are_defending(successor).as_list()
        if len(food_list) > 0:
            min_distance = min([self.get_maze_distance(successor_pos, food) for food in food_list])
            features['distance_to_food'] = min_distance
        return features

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        successor_state = successor.get_agent_state(self.index)
        successor_pos = successor_state.get_position()

        # add features to stay in our half
        features = self.add_feature_to_stay_defending(features, successor_state)

        # add features to kill visible enemies
        features = self.add_feature_to_kill_enemies(features, successor, successor_pos, game_state)

        # add features to patrol along food
        features = self.add_features_to_patrol_along_food(features, successor, successor_pos)

        # add features to not get stuck
        features = self.add_features_to_improve_movement(features, action, game_state)

        # add features to not waste time at start position
        features['distance_to_start'] = self.get_maze_distance(successor_pos, self.start_pos)

        return features

    def get_weights(self, game_state, action):
        return {'num_visible_invaders': -1000,
                'on_defense': 100,
                'visible_invader_distance': -10,
                'invisible_invader_distance': -5,
                'stop': -100,
                'reverse': -2,
                'distance_to_food': -0.5,
                'distance_to_start': 0.01}
