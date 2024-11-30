from enum import Enum
import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np

import chess
import chess.svg
import io
import random
from typing import Union

from .utils import get_chess_grid

class Actions(Enum):
    right = 0
    up = 1
    left = 2
    down = 3


class ChessEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None):
        self.board = chess.Board() # chess game board
        self.window_size = 512  # The size of the PyGame window

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2,
        # i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Box(0, 1, (8,8,12), np.int8)

        # We have 4 actions, corresponding to "right", "up", "left", "down", "right"
        self.action_space = spaces.Dict({
            "from_square": spaces.Discrete(63),
            "to_square": spaces.Discrete(63),
        })

        """
        The following dictionary maps abstract actions from `self.action_space` to 
        the direction we will walk in if that action is taken.
        i.e. 0 corresponds to "right", 1 to "up" etc.
        """

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None
    
    def _action_to_move(self, action) -> chess.Move:
        return self.board.find_move(*action)

    def _get_obs(self):
        return get_chess_grid(self.board)

    def _get_info(self):
        return {
            # "distance": np.linalg.norm(
            #     self._agent_location - self._target_location, ord=1
            # )
            "fen": self.board.fen(),
        }
    
    def _action_sample(self) -> Union[tuple[int, int], None]:
        """
        Returns a legal action
        """
        legal_moves = list(self.board.generate_legal_moves())
        
        if len(legal_moves) == 0:
            return None

        move = random.choice(legal_moves)
        return move.from_square, move.to_square

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        self.board.reset()

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        # Map the action from (from_square, to_square) to uci move
        move = self._action_to_direction(action)

        # Push move to board
        self.board.push(move)
        
        #
        reward = 0
        terminated = False

        # check_out_comes
        outcome = self.board.outcome()
        if outcome:
            terminated = True
            print(outcome)
            if outcome.termination == chess.Termination.CHECKMATE:
                reward = 10 if outcome.winner else -1
            else:
                reward = -10
                
        #
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        # canvas = pygame.Surface((self.window_size, self.window_size))
        # canvas.fill((255, 255, 255))
        boardsvg = chess.svg.board(board=self.board)
        boardsvg = io.BytesIO(boardsvg.encode())

        #Step 2: Blit the image
        image = pygame.image.load(boardsvg)
        image = pygame.transform.scale(image, (self.window_size, self.window_size))
        # canvas.blit(image,(0,0))
        pygame.display.flip()
                

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(image, image.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to
            # keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(image)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()