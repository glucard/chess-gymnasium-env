from enum import Enum
import time
import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np

import chess
import chess.svg
import io
import random
from typing import Union
import cairosvg
from typing import List

from .utils import get_chess_grid

class Actions(Enum):
    right = 0
    up = 1
    left = 2
    down = 3

# Define simple piece values
PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0 # King value is irrelevant for capture
}
    
class ChessEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": None} # 1 }

    def __init__(self, render_mode=None):
        self.board = chess.Board() # chess game board
        self.window_size = 512  # The size of the PyGame window

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2,
        # i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Box(0, 1, (8,8,12), np.int8)

        # We have 4 actions, corresponding to "right", "up", "left", "down", "right"
        self.action_space = spaces.Discrete(64*64)

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
        action = action//64, action%64
        return self.board.find_move(*action)

    def _get_obs(self):
        return get_chess_grid(self.board)

    def _get_info(self):
        return {
            # "distance": np.linalg.norm(
            #     self._agent_location - self._target_location, ord=1
            # )
            #"action_mask": self._get_action_mask(),
            #"fen": self.board.fen(),
        }
    
    def _legal_moves(self) -> List[chess.Move]:
        return list(self.board.generate_legal_moves())

    def _action_sample(self) -> Union[tuple[int, int], None]:
        """
        Returns a legal action
        """
        legal_moves = self._legal_moves()
        
        if len(legal_moves) == 0:
            return None

        move = random.choice(legal_moves)
        return move.from_square * 64 + move.to_square
    
    def _get_action_mask(self):
        """
        Returns a binary mask for valid actions.
        Example: Masking some actions based on custom rules.
        """
        # Create a mask for all actions (1 = valid, 0 = invalid)
        mask = np.zeros((64 * 64), dtype=np.int32)

        legal_moves = self._legal_moves()
        for move in legal_moves:
            mask[move.from_square * 64 + move.to_square] = 1
        return mask
    
    def _opponent_step(self) -> tuple[bool, float]:
        """ Opponent makes a greedy move.
        Returns:
            if opponent has moved return True. If not return False
        """
        legal_moves = list(self.board.generate_legal_moves())
        if not legal_moves:
            return False, 0

        best_move = None
        best_value = -float('inf')

        # Shuffle moves to add randomness among equal-value moves
        random.shuffle(legal_moves)

        for move in legal_moves:
            # 1. Check for checkmate
            if self.board.gives_check(move) and self.board.is_checkmate():
                best_move = move
                break # Found the best possible move

            # 2. Check for captures
            value = 0
            if self.board.is_capture(move):
                captured_piece = self.board.piece_at(move.to_square)
                if captured_piece:
                    value = PIECE_VALUES.get(captured_piece.piece_type, 0)
            
            if value > best_value:
                best_value = value
                best_move = move

        # 3. If no good captures, best_move will still be set
        #    from the loop (as the first random move with value 0)
        if best_move is None:
            best_move = legal_moves[0] # Should not happen if legal_moves > 0

        self.board.push(best_move)
        return True, best_value
    
    def _count_pieces(self, color:bool) -> int:
        return sum([piece.color==color for piece in self.board.piece_map().values()])

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
        # Validate the action
        assert self.action_space.contains(action), f"Invalid action: {action}"
        
        # Check pieces to calculate rewards
        start_pieces_count = self._count_pieces(True)
        opponent_start_pieces_count = self._count_pieces(False)

        # Map the action from (from_square, to_square) to uci move
        move = self._action_to_move(action)
        
        # Push move to board

        reward = 0 #-0.2
        captured_piece = self.board.piece_at(move.to_square)
        if captured_piece:
            reward += PIECE_VALUES[captured_piece.piece_type] / 9
        self.board.push(move)            

        # Opponent tries a move if not has outcome yet:
        if not self.board.outcome():
            _, captured_piece_value = self._opponent_step()

            reward += -captured_piece_value / 9


        terminated = False
        
        # Atributte rewards to remaning pieces counts
        # taken_rewards = (start_pieces_count - self._count_pieces(True)) * (-1)
        # take_rewards = (opponent_start_pieces_count - self._count_pieces(False)) * 1
        # if take_rewards != 0 or taken_rewards != 0:
        #     reward = take_rewards
        #     reward += taken_rewards

        # check_out_comes
        
        info = self._get_info()

        outcome = self.board.outcome()
        if outcome:
            # print("outcome:", outcome)
            terminated = True
            if outcome.termination == chess.Termination.CHECKMATE:
                reward = 10 if outcome.winner else -5
                # print("win" if outcome.winner else "lost")
                info["env/win"] = 1 if outcome.winner else 0
            else:
                # print("draw")
                info["env/win"] = 0
                reward = -5
        elif self._count_pieces(True) <= 2:
            terminated = True
            reward = -5
            info["env/win"] = 0
        

                
        #
        observation = self._get_obs()

        if self.render_mode == "human":
            self._render_frame()

        if terminated:
            info["env/captured_pieces"] = 16 - self._count_pieces(False)
            info["env/lost_pieces"] = 16 - self._count_pieces(True)

        # print(reward)
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

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        # Create the chess board SVG
        board_svg = chess.svg.board(board=self.board)

        # Convert SVG to PNG using cairosvg
        png_data = cairosvg.svg2png(bytestring=board_svg.encode())

        # Load PNG into Pygame
        image = pygame.image.load(io.BytesIO(png_data))

        # Scale the image
        image = pygame.transform.scale(image, (512, 512))

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            # Display the image
            self.window.blit(image, image.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to
            # keep the framerate stable.
            if self.metadata["render_fps"] is not None:
                self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(image)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
