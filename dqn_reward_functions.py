import numpy as np
from collections import Counter

class ScrabbleDQNReward:
    def __init__(self, win_reward=100, loss_penalty=-100, draw_reward=10):
        self.win_reward = win_reward
        self.loss_penalty = loss_penalty
        self.draw_reward = draw_reward
        
        self.letter_point_values = {
            'a': 1, 'b': 3, 'c': 3, 'd': 2, 'e': 1, 'f': 4, 'g': 2, 'h': 4, 'i': 1, 'j': 8, 'k': 5, 'l': 1, 'm': 3, 
            'n': 1, 'o': 1, 'p': 3, 'q': 10, 'r': 1, 's': 1, 't': 1, 'u': 1, 'v': 4, 'w': 4, 'x': 8, 'y': 4, 'z': 10,
            '_': 0
        }
        
        self.max_letter_value = 10
        
    def calculate_reward(self, game_state, player_id, move, next_game_state=None):
        """
        Calculate the reward for a move considering:
        1. Immediate score from the move
        2. Strategic value (limiting opponent's scoring opportunities)
        3. Tile usage efficiency
        4. Game outcome (win/loss/draw bonus)
        
        Args:
            game_state: Current game state before the move
            player_id: ID of the player making the move (1 or 2)
            move: The move being evaluated (dict with word, start_space, direction, etc.)
            next_game_state: Game state after the move (if available)
            
        Returns:
            float: calculated reward
        """
        reward = 0
        
        if move and 'score' in move:
            reward += move['score']
        
        if next_game_state:
            strategic_reward = self.calculate_strategic_reward(next_game_state, player_id, move)
            reward += strategic_reward
        
        if move:
            tile_usage_reward = self.calculate_tile_usage_reward(move, game_state)
            reward += tile_usage_reward
        
        if next_game_state and next_game_state.game_over:
            outcome_reward = self.calculate_outcome_reward(next_game_state, player_id)
            reward += outcome_reward
                
        return reward
    
    def calculate_strategic_reward(self, next_game_state, player_id, move):
        """
        Strategic reward, based on limiting opponent's next move opportunities
        
        Args:
            next_game_state: Game state after the move
            player_id: ID of the player making the move
            move: The move being evaluated
            
        Returns:
            float: Strategic reward
        """
        opponent_id = 3 - player_id
        
        opponent_moves = next_game_state.get_all_valid_moves(opponent_id)
        
        if not opponent_moves:
            return 10
        
        max_opponent_score = max([m['score'] for m in opponent_moves])
        
        strategic_penalty = -0.5 * max_opponent_score / (move['score'] + 1)
        
        return strategic_penalty
    
    def calculate_tile_usage_reward(self, move, game_state):
        """
        Args:
            move: The move being evaluated
            game_state: Current game state before the move
            
        Returns:
            float: Tile usage reward
        """
        tiles_used = 0
        
        if 'positions' not in move:
            return len(move['word']) * 2
        
        for pos in move['positions']:
            if pos in game_state.is_tile_present:
                if not game_state.is_tile_present[pos][0]:
                    tiles_used += 1
            else:
                tiles_used += 1
        
        if tiles_used == 7:
            return 150
    
    
    def calculate_outcome_reward(self, game_state, player_id):
        """
        Args:
            game_state: Game state after the move
            player_id: ID of the player making the move
            
        Returns:
            float: Outcome reward
        """
        player_score = game_state.player_1_score if player_id == 1 else game_state.player_2_score
        opponent_score = game_state.player_2_score if player_id == 1 else game_state.player_1_score
        
        if player_score > opponent_score:
            return self.win_reward
        elif player_score < opponent_score:
            return self.loss_penalty
        else:
            return self.draw_reward
    
    def estimate_move_quality(self, game_state, player_id, moves):
        """
        Args:
            game_state: Current game state
            player_id: ID of the player making the move
            moves: List of possible moves
            
        Returns:
            list: Moves with estimated quality scores
        """
        scored_moves = []
        
        for move in moves:
            quality = move['score']
            
            word_length = len(move['word'])
            tiles_from_hand = 0
            
            if 'positions' in move:
                for pos in move['positions']:
                    if pos in game_state.is_tile_present:
                        if not game_state.is_tile_present[pos][0]:
                            tiles_from_hand += 1
                    else:
                        tiles_from_hand += 1
            else:
                tiles_from_hand = word_length
            
            if tiles_from_hand == 7:
                quality += 15
            else:
                quality += tiles_from_hand * 2
                
            position_bonus = 0
            if 'positions' in move:
                for pos in move['positions']:
                    if pos in game_state.is_tile_present and not game_state.is_tile_present[pos][0]:
                        if pos in game_state.special_spaces and game_state.special_spaces[pos]:
                            special = game_state.special_spaces[pos]
                            if special == 'triple_word':
                                position_bonus += 3
                            elif special == 'double_word':
                                position_bonus += 2
                            elif special == 'triple_letter':
                                position_bonus += 1.5
                            elif special == 'double_letter':
                                position_bonus += 1
            
            quality += position_bonus
            
            if 'blank_assignments' in move and move['blank_assignments']:
                for idx, letter in move['blank_assignments'].items():
                    letter_value = self.letter_point_values.get(letter.lower(), 0)
                    value_ratio = letter_value / self.max_letter_value
                    quality += 5 * value_ratio
            
            scored_moves.append({
                'move': move,
                'estimated_quality': quality
            })
            
        scored_moves.sort(key=lambda x: x['estimated_quality'], reverse=True)
        
        return scored_moves