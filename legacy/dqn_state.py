import numpy as np
import copy
from collections import Counter

class ScrabbleDQNState:
    def __init__(self):
        self.BOARD_SIZE = 15
        self.NUM_LETTERS = 26
        self.LETTER_TO_INDEX = {chr(ord('a') + i): i for i in range(self.NUM_LETTERS)}
        self.BLANK_INDEX = self.NUM_LETTERS
        
    def encode_board(self, is_tile_present, blank_tile_assignments=None):
        """
        Encode the board state as a 15x15x27 tensor
        - 26 channels for letters A-Z
        - 1 channel for blank tiles
        
        Args:
            is_tile_present: Dictionary mapping positions to [is_present, letter]
            blank_tile_assignments: Dictionary mapping positions to assigned letters for blank tiles
            
        Returns:
            15x15x27 numpy tensor representing board state
        """
        board_tensor = np.zeros((self.BOARD_SIZE, self.BOARD_SIZE, self.NUM_LETTERS + 1))
        
        for position, (is_present, letter) in is_tile_present.items():
            if is_present:
                try:
                    row = ord(position[0]) - ord('a')
                    col = int(position[1:]) - 1
                    
                    is_blank = blank_tile_assignments and position in blank_tile_assignments
                    
                    if is_blank:
                        assigned_letter = blank_tile_assignments[position]
                        letter_idx = self.LETTER_TO_INDEX.get(assigned_letter.lower(), 0)
                        board_tensor[row, col, letter_idx] = 1
                        board_tensor[row, col, self.BLANK_INDEX] = 1
                    else:
                        letter_idx = self.LETTER_TO_INDEX.get(letter.lower(), 0)
                        board_tensor[row, col, letter_idx] = 1
                except (IndexError, ValueError):
                    continue
        
        return board_tensor
    
    def encode_hand(self, player_hand):
        """
        Encode the player's hand as a 27 dimensional vector
        - 26 elements for letters A-Z
        - 1 element for blank tiles

        Args:
            player_hand: List of letters in player's hand

        Returns:
            27 dimensional numpy vector representing the hand
        """
        hand_vector = np.zeros(self.NUM_LETTERS + 1)

        for i, letter in enumerate(player_hand):
            if letter == '_':
                hand_vector[self.BLANK_INDEX] += 1
            else:
                letter_idx = self.LETTER_TO_INDEX.get(letter.lower(), 0)
                hand_vector[letter_idx] += 1

        return hand_vector
    
    def encode_unseen_tiles(self, game_state, player_hand):
        """
        Encode unseen tiles (bag + opponents hand) as a 27 dimensional vector
        
        Args:
            game_state: Current game state
            player_hand: List of letters in player's hand
            
        Returns:
            27 dimensional numpy vector representing unseen tiles
        """
        all_tiles = copy.deepcopy(game_state.letter_counts)
        unseen_vector = np.zeros(self.NUM_LETTERS + 1)
        
        for position, (is_present, letter) in game_state.is_tile_present.items():
            if is_present:
                if letter in all_tiles and all_tiles[letter] > 0:
                    all_tiles[letter] -= 1
        
        hand_counter = Counter(player_hand)
        for letter, count in hand_counter.items():
            if letter in all_tiles and all_tiles[letter] >= count:
                all_tiles[letter] -= count
        
        for letter, count in all_tiles.items():
            if letter == '_':
                unseen_vector[self.BLANK_INDEX] = count
            elif letter.lower() in self.LETTER_TO_INDEX:
                letter_idx = self.LETTER_TO_INDEX[letter.lower()]
                unseen_vector[letter_idx] = count
                
        return unseen_vector
    
    def encode_game_state_features(self, game_state, player_id):
        """
        Args:
            game_state: Current game state
            player_id: ID of the player (1 or 2)
            
        Returns:
            Numpy vector of additional game state features
        """
        player_score = game_state.player_1_score if player_id == 1 else game_state.player_2_score
        opponent_score = game_state.player_2_score if player_id == 1 else game_state.player_1_score
        
        tiles_in_bag = sum(game_state.letter_counts.values())
        
        opponent_hand = game_state.player_2_hand if player_id == 1 else game_state.player_1_hand
        tiles_in_opponent_hand = len(opponent_hand)
        
        features = np.array([
            player_score,
            opponent_score,
            tiles_in_bag,
            tiles_in_opponent_hand,
            player_score - opponent_score
        ])
        
        return features
    
    def get_state_representation(self, game_state, player_id):
        player_hand = game_state.player_1_hand if player_id == 1 else game_state.player_2_hand
        board_tensor = self.encode_board(game_state.is_tile_present, game_state.blank_tile_assignments)
        hand_vector = self.encode_hand(player_hand)
        unseen_vector = self.encode_unseen_tiles(game_state, player_hand)
        game_features = self.encode_game_state_features(game_state, player_id)

        return {
            'board': board_tensor,
            'hand': hand_vector,
            'unseen': unseen_vector,
            'game_features': game_features
        }
