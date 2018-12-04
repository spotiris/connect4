import random
import sys
import copy
import sys
import numpy as np

from stacker_bot import bot as stacker_bot
from random_bot import bot as random_bot
from connect4 import Game as C4Game


class Game(C4Game):
    def __init__(self, boardSize, target):
        super(Game, self).__init__(boardSize, target)
        self.kernels = [
            np.ones((self.target, 1)),
            np.ones((1, self.target)),
            np.eye(self.target), np.eye(self.target)[:, ::-1]
        ]
        self.player_id, self.bot_id = 1, 2
        self.opponent = None
        self.round = 0
        self.bmap = np.array([0, -1, 1])
        self.reset()

    @property
    def board(self):
        return self.bmap[self.gameState.ravel()].astype(np.float32)

    def reset(self):
        self.round = 0
        self.gameState = np.zeros(self.boardSize, dtype=np.int64)
        self.opponent = np.random.choice([stacker_bot, random_bot])
        return self.board

    def is_winner(self, board, player_id):
        return self.check_for_win(player_id, target=self.target)

    def is_board_full(self):
        "checks if the board is full"
        return self.check_for_draw()

    def step(self, action):
      over = False
      reward = 0

      if not self.take_turn(self.player_id, action+1):
        # If the move is not valid, penalty
        return None, -33, True

      if self.is_winner(self.board, self.player_id):
        # If move is winning, high reward
        reward = 100
        over = True
      elif self.is_board_full():
        # If move is draw, low reward
        reward = 10
        over = True
      else:
        # Otherwise it's the bot's turn
        bot_move = self.get_bot_move()
        valid_turn = self.take_turn(self.bot_id, bot_move)

        if self.is_winner(self.board, self.bot_id):
            # Check if the bot won this round, if so penalise
          reward = -1
          over = True
        elif self.is_board_full():
          # Check if the bot's move caused a draw, low reward
          reward = 10
          over = True
      self.round += 1

      return self.board, reward, over

    def get_bot_move(self):
        return self.opponent(self.gameState, self.round, self.bot_id)


if __name__ == "__main__":
    game = Game((6, 7), 4)