import tensorflow as tf
import numpy as np
import useful_tools
import sys

from rl.deep_q_network import DeepQNetwork
from game import Game


class RlBot(object):
    def __init__(self, boardSize):
        self.state_dim   = np.product(boardSize)
        self.num_actions = boardSize[1]
        self.q_network = None
        self.sess = tf.Session()
        self.optimizer = tf.train.RMSPropOptimizer(learning_rate=0.0001, decay=0.9)
        self.create_model()

    def create_model(self):
        def value_network(states):
            W1 = tf.get_variable("W1", [self.state_dim, 256],
                initializer=tf.random_normal_initializer(stddev=0.1))
            b1 = tf.get_variable("b1", [256],
                initializer=tf.constant_initializer(0))
            h1 = tf.nn.relu(tf.matmul(states, W1) + b1)

            W2 = tf.get_variable("W2", [256, 64],
                initializer=tf.random_normal_initializer(stddev=0.1))
            b2 = tf.get_variable("b2", [64],
                initializer=tf.constant_initializer(0))
            h2 = tf.nn.relu(tf.matmul(h1, W2) + b2)

            Wo = tf.get_variable("Wo", [64, self.num_actions],
                initializer=tf.random_normal_initializer(stddev=0.1))
            bo = tf.get_variable("bo", [self.num_actions],
                initializer=tf.constant_initializer(0))

            p = tf.matmul(h2, Wo) + bo
            return p

        self.q_network = DeepQNetwork(
            self.sess, self.optimizer, value_network, self.state_dim, self.num_actions,
            init_exp=0.6,         # initial exploration prob
            final_exp=0.1,        # final exploration prob
            anneal_steps=120000,  # N steps for annealing exploration
            discount_factor=0.8)  # no need for discounting

        # load checkpoint if there is any
        saver = tf.train.Saver()
        checkpoint = tf.train.get_checkpoint_state("model")
        # if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(self.sess, checkpoint.model_checkpoint_path)
        print(checkpoint.model_checkpoint_path)
        print("successfully loaded checkpoint")

rl_bot = RlBot([6, 7])

def bot( gameState , round, playerID ):
    # ---------------------------------------------------
    # description:
    #   randomly chooses a column to place token
    # inputs:
    #   gameState ==> a copy of the current board (incl. tokens)
    #   round ==> the iteration number for the game (starting at 1)
    #   playerID ==> whether player is 1 or 2
    # outputs:
    #   col ==> column location to place token (between 1 and boardSize)
    # ---------------------------------------------------
    # action = None
    # while action is None or useful_tools.isColumnFull(action+1, gameState):
    bmap = np.zeros(3)
    bmap[playerID] = -1
    bmap[2-playerID] = 1
    if round == 1 and playerID == 1:
        action = np.random.randint(gameState.shape[1])
    else:
        action = rl_bot.q_network.eGreedyAction(
            bmap[gameState.ravel()[np.newaxis, ...]], explore=False)
        while useful_tools.isColumnFull(action + 1, gameState):
            action = rl_bot.q_network.eGreedyAction(
               bmap[gameState.ravel()[np.newaxis, ...]], explore=True)
    return action + 1




