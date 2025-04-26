import numpy as np
from collections import Counter
from dqn_state import ScrabbleDQNState

class ScrabbleDQNReward:
    def __init__(self, win_reward=500.0, loss_penalty=-500.0, draw_reward=10.0):
        self.win_reward = win_reward
        self.loss_penalty = loss_penalty
        self.draw_reward = draw_reward
        self.state_encoder = ScrabbleDQNState()
        
        self.letter_point_values = {
            'a': 1, 'b': 3, 'c': 3, 'd': 2, 'e': 1, 'f': 4, 'g': 2, 'h': 4, 'i': 1, 'j': 8, 'k': 5, 'l': 1, 'm': 3, 
            'n': 1, 'o': 1, 'p': 3, 'q': 10, 'r': 1, 's': 1, 't': 1, 'u': 1, 'v': 4, 'w': 4, 'x': 8, 'y': 4, 'z': 10,
            '_': 0
        }
        
    def calculate_reward(self, game_state, player_id, move, next_game_state=None):
        """
        Calculate the reward for a move considering:
        1. Immediate score from the move
        2. Tile usage efficiency
        3. If bingo is not played and lower scoring move is all that is possible, reward for keeping better bingo tiles
        4. Game outcome (win/loss/draw)
        
        Args:
            game_state: Current game state before the move
            player_id: ID of the player making the move (1 or 2)
            move: The move being evaluated (dict with word, start_space, direction, etc.)
            next_game_state: Game state after the move
            
        Returns:
            float: calculated reward
        """
        reward = 0.0

        # immediate score
        if move and 'score' in move:
            immediate_score = move['score']
            if immediate_score is not None:
                reward += float(immediate_score)
            else:
                print(f"Warning: Move received with 'score': None. Move: {move}")


        # tile usage
        if move:
            tile_usage_reward = self.calculate_tile_usage_reward(move, game_state)
            if tile_usage_reward is not None:
                reward += tile_usage_reward
            else:
                print(f"Warning: calculate_tile_usage_reward returned None.")
        
        # kept tiles for bingo potential
        if move:
            bingo_potential_reward = self.calculate_bingo_potential_reward(move, game_state, player_id)
            if bingo_potential_reward is not None:
                reward += bingo_potential_reward
            else:
                print(f"Warning: calculate_bingo_potential_reward returned None.")


        # game outcome
        if next_game_state and next_game_state.game_over:
            outcome_reward = self.calculate_outcome_reward(next_game_state, player_id)
            if outcome_reward is not None:
                reward += outcome_reward
            else:
                print(f"Warning: calculate_outcome_reward returned None.")

        return reward
    
    
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
            return 50.0
        else:
            return 5.0
    
    
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
    
    def calculate_bingo_potential_reward(self, move, game_state, player_id):
        """
        Calculate a reward based on the bingo potential of the hand after playing a move.
        """
        current_hand = game_state.player_1_hand if player_id == 1 else game_state.player_2_hand
        
        state = self.state_encoder.get_state_representation(game_state, player_id)
        unseen_vector = state['unseen']
        
        hand_after_move = current_hand.copy()
        
        if 'positions' in move:
            for pos in move['positions']:
                if pos not in game_state.is_tile_present or not game_state.is_tile_present[pos][0]:
                    letter = None
                    for i, placed_pos in enumerate(move['positions']):
                        if placed_pos == pos:
                            letter = move['word'][i].lower()
                            break
                    
                    if letter and letter in hand_after_move:
                        hand_after_move.remove(letter)
                    elif '_' in hand_after_move:
                        hand_after_move.remove('_')
        
        bingo_potential = self._evaluate_bingo_potential(hand_after_move, unseen_vector)
        
        if move['score'] < 30:
            # # debug
            # print(f"VAR bingo_potential | FUNC calculate_bingo_potential_reward (move was LESS than 30 points): {bingo_potential * 3.0}")
            return bingo_potential * 3.0
        else:
            # # debug
            # print(f"VAR bingo_potential | FUNC calculate_bingo_potential_reward (move was GREATER than 30 points): {bingo_potential * 0.5}")
            return bingo_potential * 0.5
    
    def _evaluate_bingo_potential(self, hand, unseen_vector):
        """
        Evaluate how likely the hand is to form a bingo on the next turn, 
        considering vowel/consonant balance.
        
        Args:
            hand: Current hand after move
            unseen_vector: 27-dimensional numpy vector of unseen tiles
            
        Returns:
            float: Bingo potential score
        """
        vowels = {'a', 'e', 'i', 'o', 'u'}
        consonants = {'b', 'c', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'm', 
                    'n', 'p', 'q', 'r', 's', 't', 'v', 'w', 'x', 'y', 'z'}
        
        one_point_tiles = {'a', 'e', 'i', 'o', 'n', 'r', 's', 't', 'l'}
        unique_one_point_count = len(set(t for t in hand if t in one_point_tiles))
        
        blank_count = hand.count('_')
        
        non_blank_hand = [t for t in hand if t != '_']
        vowel_count = sum(1 for t in non_blank_hand if t in vowels)
        consonant_count = sum(1 for t in non_blank_hand if t in consonants)
        
        ideal_vowel_min, ideal_vowel_max = 2, 3
        vowel_balance_score = 0
        
        if vowel_count < ideal_vowel_min:
            vowel_balance_score = -1.0 * (ideal_vowel_min - vowel_count)
        elif vowel_count > ideal_vowel_max:
            vowel_balance_score = -1.0 * (vowel_count - ideal_vowel_max)
        else:
            vowel_balance_score = 1.0
        
        letter_counts = {}
        for letter in non_blank_hand:
            letter_counts[letter] = letter_counts.get(letter, 0) + 1
        
        good_duplicates = {
            'e': 3,
            's': 2,
            'a': 2, 
            'i': 2,
            'r': 2,
            'o': 2,
            't': 2,
            'n': 2,
            'l': 2
        }
        
        duplicate_penalty = 0
        good_duplicate_bonus = 0
        
        for letter, count in letter_counts.items():
            if count > 1:
                if letter in good_duplicates and count <= good_duplicates[letter]:
                    good_duplicate_bonus += 0.5 * (count - 1)
                else:
                    if letter in vowels and letter != 'e':
                        duplicate_penalty += 1.5 * (count - 1)
                    else:
                        duplicate_penalty += 1.0 * (count - 1)
        
        letter_mapping = {i: chr(ord('a') + i) for i in range(26)}
        letter_mapping[26] = '_'
        
        tiles_needed = 7 - len(hand)
        draw_potential = 0
        
        if tiles_needed > 0 and unseen_vector.sum() > 0:
            needed_letters = self._determine_needed_letters_balanced(
                vowel_count, consonant_count, letter_counts, unseen_vector)
            
            total_unseen = unseen_vector.sum()
            
            for letter in needed_letters:
                if letter == '_':
                    letter_idx = 26
                else:
                    letter_idx = ord(letter) - ord('a')
                
                if 0 <= letter_idx < len(unseen_vector) and unseen_vector[letter_idx] > 0:
                    weight = 1.0
                    if letter in vowels and vowel_count < ideal_vowel_min:
                        weight = 1.5
                    elif letter in consonants and consonant_count < 4:
                        weight = 1.5
                    
                    draw_potential += weight * unseen_vector[letter_idx] / total_unseen
        
        bingo_potential = (
            2.0 * unique_one_point_count +
            5.0 * blank_count +
            2.0 * vowel_balance_score +
            good_duplicate_bonus -
            duplicate_penalty +
            2.5 * draw_potential
        )
        
        # # debug
        # print(f"VAR bingo_potential | FUNC _evaluate_bingo_potential: {bingo_potential}")

        return bingo_potential

    def _determine_needed_letters_balanced(self, vowel_count, consonant_count, letter_counts, unseen_tiles):
        """
        Determine which letters would most help complete a bingo,
        considering vowel/consonant balance.
        
        Args:
            hand: Current hand
            vowel_count: Number of vowels in hand
            consonant_count: Number of consonants in hand
            letter_counts: Dictionary of current letter counts
            unseen_tiles: unseen tiles array (in opponents hand and in bag)
            
        Returns:
            set: Letters that would be most helpful
        """
        vowels = {'a', 'e', 'i', 'o', 'u'}
        one_point_tiles = {'a', 'e', 'i', 'o', 'n', 'r', 's', 't', 'l'}
        needed_letters = set()
        
        if vowel_count < 2:
            for v in vowels.intersection(one_point_tiles):
                if v not in letter_counts or letter_counts[v] == 0:
                    needed_letters.add(v)
                elif v == 'e' and letter_counts[v] == 1:
                    needed_letters.add(v)
        
        if consonant_count < 4:
            one_point_consonants = one_point_tiles.difference(vowels)
            for c in one_point_consonants:
                if c not in letter_counts or letter_counts[c] == 0:
                    needed_letters.add(c)
        
        if 2 <= vowel_count <= 3 and 3 <= consonant_count <= 5:
            high_value_useful = {'d', 'g', 'p'}
            for c in high_value_useful:
                if c not in letter_counts or letter_counts[c] == 0:
                    needed_letters.add(c)
        
        duplicate_count = sum(1 for letter, count in letter_counts.items() if count > 1)
        
        if duplicate_count >= 2:
            for letter in one_point_tiles:
                if letter not in letter_counts or letter_counts[letter] == 0:
                    needed_letters.add(letter)
        
        if unseen_tiles[26] > 0:
            needed_letters.add('_')
        
        # # debug
        # print(f"VAR needed_letters | FUNC _determine_needed_letters_balanced: {needed_letters}")

        return needed_letters

    def estimate_move_quality(self, game_state, moves):
        """
        Estimate the quality of potential moves, incorporating bingo potential.
        
        Args:
            game_state: Current game state
            moves: List of possible moves
            
        Returns:
            list: Moves with estimated quality scores
        """
        scored_moves = []
        avg_move_score = self.average_move_score(moves)
        
        player_id = game_state.current_player
        current_hand = game_state.player_1_hand if player_id == 1 else game_state.player_2_hand
        
        state = self.state_encoder.get_state_representation(game_state, player_id)
        unseen_vector = state['unseen']
        
        vowels = {'a', 'e', 'i', 'o', 'u'}
        consonants = set('bcdfghjklmnpqrstvwxyz')
        one_point_tiles = {'a', 'e', 'i', 'o', 'n', 'r', 's', 't', 'l'}
        
        for move in moves:
            if avg_move_score > 0:
                quality = move['score'] / avg_move_score
            else:
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
                quality *= 5
            
            position_bonus = 0
            if 'positions' in move:
                for pos in move['positions']:
                    if pos in game_state.is_tile_present and not game_state.is_tile_present[pos][0]:
                        if pos in game_state.special_spaces and game_state.special_spaces[pos]:
                            special = game_state.special_spaces[pos]
                            if special == 'triple_word':
                                position_bonus *= 3.5
                            elif special == 'double_word':
                                position_bonus *= 1.5
                            elif special == 'triple_letter':
                                position_bonus += 2
                            # elif special == 'double_letter':
                            #     position_bonus += 1
            
            quality += position_bonus
            
            hand_after_move = current_hand.copy()
            
            tiles_to_remove = []
            if 'positions' in move:
                for pos in move['positions']:
                    if pos in game_state.is_tile_present:
                        if not game_state.is_tile_present[pos][0]:
                            for i, placed_pos in enumerate(move['positions']):
                                if placed_pos == pos:
                                    tiles_to_remove.append(move['word'][i].lower())
                                    break
                    else:
                        for i, placed_pos in enumerate(move['positions']):
                            if placed_pos == pos:
                                tiles_to_remove.append(move['word'][i].lower())
                                break
            
            for tile in tiles_to_remove:
                if tile in hand_after_move:
                    hand_after_move.remove(tile)
                elif '_' in hand_after_move:
                    hand_after_move.remove('_')
            

            if tiles_from_hand < 7:
                unique_one_point_count = len(set(t for t in hand_after_move if t in one_point_tiles))
                
                blank_count = hand_after_move.count('_')
                
                non_blank_hand = [t for t in hand_after_move if t != '_']
                vowel_count = sum(1 for t in non_blank_hand if t in vowels)
                consonant_count = sum(1 for t in non_blank_hand if t in consonants)
                
                vowel_balance_score = 0
                if vowel_count < 2:
                    vowel_balance_score = -1.0
                elif vowel_count > 3:
                    vowel_balance_score = -0.5 * (vowel_count - 3)
                else:
                    vowel_balance_score = 1.0
                
                letter_counts = {}
                for letter in non_blank_hand:
                    letter_counts[letter] = letter_counts.get(letter, 0) + 1
                
                good_duplicates = {
                    'e': 3, 's': 2, 'a': 2, 'i': 2, 'r': 2, 
                    'o': 2, 't': 2, 'n': 2, 'l': 2
                }
                
                duplicate_bonus = 0
                duplicate_penalty = 0
                
                for letter, count in letter_counts.items():
                    if count > 1:
                        if letter in good_duplicates and count <= good_duplicates[letter]:
                            duplicate_bonus += 0.3 * (count - 1)
                        elif letter in vowels:
                            duplicate_penalty += 0.5 * (count - 1)
                        else:
                            duplicate_penalty += 0.3 * (count - 1)
                
                draw_potential = 0
                tiles_needed = 7 - len(hand_after_move)
                
                if tiles_needed > 0:
                    needed_vowels = max(0, 2 - vowel_count)
                    needed_consonants = max(0, 4 - consonant_count)
                    
                    total_unseen = unseen_vector.sum()
                    if total_unseen > 0:
                        unseen_vowels = sum(unseen_vector[ord(v) - ord('a')] for v in vowels)
                        unseen_consonants = sum(unseen_vector[ord(c) - ord('a')] for c in consonants)
                        unseen_blanks = unseen_vector[26]
                        
                        if needed_vowels > 0:
                            draw_potential += 0.5 * (unseen_vowels / total_unseen)
                        if needed_consonants > 0:
                            draw_potential += 0.5 * (unseen_consonants / total_unseen)
                        
                        draw_potential += 1.0 * (unseen_blanks / total_unseen)
                
                bingo_potential = (
                    1.0 * unique_one_point_count +
                    4.0 * blank_count +
                    1.5 * vowel_balance_score +
                    duplicate_bonus - 
                    duplicate_penalty +
                    2.0 * draw_potential
                )
                
                if move['score'] < 20:
                    quality += bingo_potential * 0.8
                elif move['score'] < 30:
                    quality += bingo_potential * 0.5
                else:
                    quality += bingo_potential * 0.2
            
            scored_moves.append({
                'move': move,
                'estimated_quality': quality
            })
        
        scored_moves.sort(key=lambda x: x['estimated_quality'], reverse=True)
        
        return scored_moves

    def average_move_score(self, moves: list[dict]):
        total_score = 0

        for a in moves:
            total_score += a["score"]
        
        result = total_score / len(moves)

        return result