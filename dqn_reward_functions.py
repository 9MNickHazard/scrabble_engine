import numpy as np
from collections import Counter
from dqn_state import ScrabbleDQNState

class ScrabbleDQNReward:
    def __init__(self, win_reward=200.0, loss_penalty=-200.0, draw_reward=10.0):
        self.win_reward = win_reward
        self.loss_penalty = loss_penalty
        self.draw_reward = draw_reward
        self.state_encoder = ScrabbleDQNState()
        
        self.letter_point_values = {
            'a': 1, 'b': 3, 'c': 3, 'd': 2, 'e': 1, 'f': 4, 'g': 2, 'h': 4, 'i': 1, 'j': 8, 'k': 5, 'l': 1, 'm': 3, 
            'n': 1, 'o': 1, 'p': 3, 'q': 10, 'r': 1, 's': 1, 't': 1, 'u': 1, 'v': 4, 'w': 4, 'x': 8, 'y': 4, 'z': 10,
            '_': 0
        }
        
    def calculate_reward(self, game_state, player_id, move, top_move_score, next_game_state=None):
        """
        Calculate the reward for a move considering:
        1. Immediate score from the move
        2. Tile usage
        3. If bingo is not played and lower scoring move is all that is possible, reward for keeping better bingo tiles
        4. Setuping up good future moves
        5. Defensive plays
        6. Game outcome (win/loss/draw)
        
        Args:
            game_state: Current game state before the move
            player_id: ID of the player making the move (1 or 2)
            move: The move being evaluated
            next_game_state: Game state after the move
            
        Returns:
            float: calculated reward
        """
        reward = 0.0

        state = self.state_encoder.get_state_representation(game_state, player_id)

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

        # played positions for defensive & bingo potential rewards
        played_positions = []
        if move and 'positions' in move:
            for pos in move['positions']:
                if pos in game_state.is_tile_present:
                    if not game_state.is_tile_present[pos][0]:
                        played_positions.append(pos)

        # kept tiles for bingo potential, only apply if there are no bingos and enough tiles to draw in the bag. 56 points is the lowest amount of points possible when bingoing
        player_hand_before_move = game_state.player_1_hand if player_id == 1 else game_state.player_2_hand

        hand_size_after_play_before_draw = len(player_hand_before_move) - len(played_positions)

        num_in_bag_before_this_turns_draw = sum(game_state.letter_counts.values())

        tiles_needed_to_refill = max(0, 7 - hand_size_after_play_before_draw)

        if top_move_score < 56 and (tiles_needed_to_refill == 0 or num_in_bag_before_this_turns_draw >= tiles_needed_to_refill):
            if move:
                bingo_potential_reward = self.calculate_bingo_potential_reward(move, game_state, player_id)
                if bingo_potential_reward is not None:
                    reward += bingo_potential_reward
                else:
                    print(f"Warning: calculate_bingo_potential_reward returned None.")
            
        # setup value reward
        if move:
            setup_reward = self.calculate_setup_value(game_state, move, player_id)
            reward += setup_reward
            
        # defensive play reward
        if move and played_positions:
            defensive_reward = self.calculate_defensive_value(game_state, move, played_positions, state['unseen'])
            reward += defensive_reward
        
        # endgame reward
        if num_in_bag_before_this_turns_draw <= 7:
            endgame_reward = self.calculate_endgame_strategy_reward(game_state, player_id, move, next_game_state, state)
            reward += endgame_reward

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
            return 1.0 * tiles_used
    
    
    def calculate_outcome_reward(self, game_state, player_id):
        """
        Args:
            game_state: Game state after the move
            player_id: Current player ID
            
        Returns:
            float: Game outcome reward
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

        Args:
            move: Move being evaluated
            game_state: Current game state
            player_id: Current player ID
        
        Returns:
            float: Bingo potential reward
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

        # # debug
        # print(f"VAR bingo_potential | FUNC calculate_bingo_potential_reward: {bingo_potential}")
        return bingo_potential

    
    def _evaluate_bingo_potential(self, hand, unseen_vector):
        """
        Evaluate how likely the hand is to form a bingo on the next turn, 
        considering vowel/consonant balance.
        
        Args:
            hand: Current hand after move
            unseen_vector: unseen tiles vector
            
        Returns:
            float: Bingo potential score
        """
        vowels = {'a', 'e', 'i', 'o', 'u'}
        consonants = {'b', 'c', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'm', 
                    'n', 'p', 'q', 'r', 's', 't', 'v', 'w', 'x', 'y', 'z'}
        
        one_point_tiles = {'a', 'e', 'i', 'o', 'u', 'n', 'r', 's', 't', 'l'}
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
            'e': 2,
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
            vowel_count: Number of vowels in hand
            consonant_count: Number of consonants in hand
            letter_counts: Dictionary of current letter counts
            unseen_tiles: unseen tiles vector
            
        Returns:
            set: Letters that would be most helpful
        """
        vowels = {'a', 'e', 'i', 'o', 'u'}
        one_point_tiles = {'a', 'e', 'i', 'o', 'u', 'n', 'r', 's', 't', 'l'}
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
            high_value_useful = {'b', 'c', 'd', 'g', 'h', 'm', 'p'}
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
        Estimate the quality of potential moves.
        
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

                played_positions = []
                if 'positions' in move:
                    for pos in move['positions']:
                        if pos in game_state.is_tile_present:
                            if not game_state.is_tile_present[pos][0]:
                                played_positions.append(pos)

                setup_value = self.calculate_setup_value(game_state, move, player_id)

                defensive_value = 0
                if tiles_from_hand < 7 and move['score'] < 50:
                    defensive_value = self.calculate_defensive_value(game_state, move, played_positions, unseen_vector)
                
                if move['score'] < 20:
                    quality += (bingo_potential * 0.8) + (setup_value * 0.3) + (defensive_value * 0.3)
                elif move['score'] < 30:
                    quality += (bingo_potential * 0.5) + (setup_value * 0.2) + (defensive_value * 0.2)
                else:
                    quality += (bingo_potential * 0.2) + (setup_value * 0.1) + (defensive_value * 0.1)
            
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
    

    def calculate_setup_value(self, game_state, move, player_id):
        """
        Calculate how well a move creates hook opportunities with letters
        still in the player's hand after the move.

        Args:
            game_state: Current game state
            move: Move being evaluated
            player_id: Current player ID
        
        Returns:
            float: Setup value of the move
        """
        value = 0.0
        
        # get word played
        word = move['word'].lower() if 'word' in move else ""
        if not word:
            return 0.0
        
        current_hand = game_state.player_1_hand if player_id == 1 else game_state.player_2_hand
        
        hand_after_move = current_hand.copy()
        
        # remove letters used in the played word
        if 'positions' in move:
            for pos in move['positions']:
                if pos in game_state.is_tile_present:
                    if not game_state.is_tile_present[pos][0]:
                        for i, placed_pos in enumerate(move['positions']):
                            if placed_pos == pos:
                                letter = move['word'][i].lower()
                                if letter in hand_after_move:
                                    hand_after_move.remove(letter)
                                elif '_' in hand_after_move:
                                    hand_after_move.remove('_')
                                break
        
        # for each letter left in hand after word played, check if it can hook onto the played word
        valid_hooks_found = 0
        blanks_count = hand_after_move.count('_')
        
        # check prefix hooks
        for letter in set(hand_after_move):
            if letter == '_':
                continue
                
            # check if adding this letter to the beginning of the word forms a valid word
            prefix_word = letter + word
            if prefix_word in game_state.words_set:
                valid_hooks_found += 1
                if letter == 's':
                    value += 3.0
                else:
                    value += 2.0
        
        # check suffix hooks
        for letter in set(hand_after_move):
            if letter == '_':
                continue
                
            # check if adding letter to the end of the word forms a valid word
            suffix_word = word + letter
            if suffix_word in game_state.words_set:
                valid_hooks_found += 1
                if letter == 's':
                    value += 3.0
                else:
                    value += 2.0
        
        # check blank hooks
        if blanks_count > 0:
            for letter in 'abcdefghijklmnopqrstuvwxyz':
                # prefix hook
                prefix_word = letter + word
                if prefix_word in game_state.words_set:
                    value += 1.0
                    break
                    
            for letter in 'abcdefghijklmnopqrstuvwxyz':
                # suffix hook
                suffix_word = word + letter
                if suffix_word in game_state.words_set:
                    value += 1.0
                    break
        
        # check other conditions for hook playability
        can_prefix = True
        can_suffix = True

        if 'positions' in move and len(move['positions']) > 0:
            direction = move.get('direction', '')
            
            # prefix limitations
            if direction == 'right':
                first_pos = move['positions'][0]
                first_col = int(''.join(filter(str.isdigit, first_pos)))
                if first_col == 1:  # left edge
                    can_prefix = False
            elif direction == 'down':
                first_pos = move['positions'][0]
                first_row = first_pos[0]
                if first_row == 'a':  # top edge
                    can_prefix = False
            
            # suffix limitations
            if direction == 'right':
                last_pos = move['positions'][-1]
                last_col = int(''.join(filter(str.isdigit, last_pos)))
                if last_col == 15:  # right edge
                    can_suffix = False
            elif direction == 'down':
                last_pos = move['positions'][-1]
                last_row = last_pos[0]
                if last_row == 'o':  # bottom edge
                    can_suffix = False

        # small negative reward if hooks are physically impossible
        if valid_hooks_found > 0:
            if not can_prefix and not can_suffix:
                value -= 1.0  # both hook types impossible
            elif not can_prefix or not can_suffix:
                value -= 0.5  # one hook type impossible
            else:
                value += 1.0
        
        return value

    def calculate_defensive_value(self, game_state, move, played_positions, unseen_vector):
        """
        Calculate how effectively a move blocks premium squares through strategic letter placement.
        
        Args:
            game_state: Current game state
            move: Move being evaluated
            played_positions: Positions where tiles are being placed
            unseen_vector: Vector of unseen tiles
        
        Returns:
            float: Defensive value of the move
        """

        if move and 'score' in move and move['score'] >= 50:
            return 0.0

        value = 0.0

        two_letter_words = {
            'aa', 'ab', 'ad', 'ae', 'ag', 'ah', 'ai', 'al', 'am', 'an', 'ar', 'as', 'at', 'aw', 'ax', 'ay',
            'ba', 'be', 'bi', 'bo', 'by', 'da', 'de', 'do', 'ed', 'ef', 'eh', 'el', 'em', 'en', 'er', 'es',
            'et', 'ew', 'ex', 'fa', 'fe', 'gi', 'go', 'ha', 'he', 'hi', 'hm', 'ho', 'id', 'if', 'in', 'is',
            'it', 'jo', 'ka', 'ki', 'la', 'li', 'lo', 'ma', 'me', 'mi', 'mm', 'mo', 'mu', 'my', 'ne', 'no',
            'nu', 'od', 'oe', 'of', 'oh', 'oi', 'ok', 'om', 'on', 'op', 'or', 'os', 'ow', 'ox', 'oy', 'pa',
            'pe', 'pi', 'po', 'qi', 're', 'sh', 'si', 'so', 'ta', 'te', 'ti', 'to', 'uh', 'um', 'un', 'up',
            'us', 'ut', 'we', 'wo', 'xi', 'xu', 'ya', 'ye', 'yo', 'za'
        }
        
        # lookup dictionaries for second-letter possibilities
        second_letter_options = {}
        for word in two_letter_words:
            if word[0] not in second_letter_options:
                second_letter_options[word[0]] = set()
            second_letter_options[word[0]].add(word[1])
        
        first_letter_options = {}
        for word in two_letter_words:
            if word[1] not in first_letter_options:
                first_letter_options[word[1]] = set()
            first_letter_options[word[1]].add(word[0])
        
        # premium squares to defend
        premium_squares = []
        for pos, special in game_state.special_spaces.items():
            if special in ['triple_word', 'double_word']:
                premium_squares.append((pos, special))
        
        # check for blocking moves around premium squares
        for pos in played_positions:
            row_char = pos[0]
            col_num = int(''.join(filter(str.isdigit, pos)))
            letter_placed = None
            
            # which letter was placed here
            if 'positions' in move:
                for i, placed_pos in enumerate(move['positions']):
                    if placed_pos == pos:
                        letter_placed = move['word'][i].lower()
                        break
            
            if not letter_placed:
                continue
                
            # check adjacent positions for premium squares
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                r_ord = ord(row_char) + dr
                c = col_num + dc
                
                if r_ord < ord('a') or r_ord > ord('o') or c < 1 or c > 15:
                    continue
                    
                adjacent_pos = f"{chr(r_ord)}{c}"
                
                # is this premium square thats empty
                for square_pos, square_type in premium_squares:
                    if adjacent_pos == square_pos and (adjacent_pos in game_state.is_tile_present and 
                                                    not game_state.is_tile_present[adjacent_pos][0]):
                        square_value = 3.0 if square_type == 'triple_word' else 2.0 if square_type == 'double_word' else 1.0
                        
                        # direction of potential play
                        is_vertical = dr != 0  # letter is above/below the premium square
                        
                        # how many valid two letter options exist
                        if is_vertical:
                            if dr < 0:  # premium square is below letter
                                connecting_options = second_letter_options.get(letter_placed, set())
                            else:  # premium square is above letter
                                connecting_options = first_letter_options.get(letter_placed, set())
                        else:
                            if dc < 0:  # premium square is right of letter
                                connecting_options = second_letter_options.get(letter_placed, set())
                            else:  # premium square is left of letter
                                connecting_options = first_letter_options.get(letter_placed, set())
                        
                        # how many of these options are available to opponent
                        available_options = 0
                        for option in connecting_options:
                            option_idx = ord(option) - ord('a')
                            if option_idx >= 0 and option_idx < 26 and unseen_vector[option_idx] > 0:
                                available_options += 1
                        
                        # blocking effectiveness (high when few or no options exist)
                        if len(connecting_options) == 0:
                            # no valid two letter words can be formed
                            value += square_value * 3.0
                        elif available_options == 0:
                            # connecting letters have been played already
                            value += square_value * 2.5
                        elif len(connecting_options) <= 2:
                            # limited options (like 'c' which only connects with 'h')
                            value += square_value * (1.0 + (1.0 - available_options/len(connecting_options)))
                        elif available_options <= 2:
                            # few connecting letters remain available
                            value += square_value * 0.5
        
        return value
    


    def calculate_endgame_strategy_reward(self, game_state, player_id, move, next_game_state, encoded_state):
        """
        Calculate reward for endgame strategy, focusing on playing out,
        minimizing rack penalty, and strategic tile management when bag is low.
        
        Args:
            game_state: Current game state before the move
            player_id: ID of the player making the move (1 or 2)
            move: The move being evaluated
            next_game_state: Game state after the move
            
        Returns:
            float: Calculated endgame strategy reward
        """
        reward = 0.0
        tiles_in_bag = sum(game_state.letter_counts.values())
        
        is_endgame = tiles_in_bag <= 7
        
        if not is_endgame and not (next_game_state and next_game_state.game_over):
            return 0.0
        
        if move is None:
            if next_game_state and next_game_state.game_over:
                player_hand_final = next_game_state.player_1_hand if player_id == 1 else next_game_state.player_2_hand
                opponent_hand_final = next_game_state.player_2_hand if player_id == 1 else next_game_state.player_1_hand
                
                player_rack_value = sum(self.letter_point_values.get(t.lower(), 0) for t in player_hand_final)
                opponent_rack_value = sum(self.letter_point_values.get(t.lower(), 0) for t in opponent_hand_final)
                
                if not player_hand_final:
                    reward += 20.0
                    reward += opponent_rack_value * 2
                else:
                    reward -= player_rack_value
                    
                    if not opponent_hand_final:
                        reward -= player_rack_value

            return reward
        
        tiles_played_from_hand = 0
        if 'positions' in move:
            for pos in move['positions']:
                if not game_state.is_tile_present[pos][0]:
                    tiles_played_from_hand += 1
        
        # playing out bonus
        if next_game_state and next_game_state.game_over:
            player_hand_after = next_game_state.player_1_hand if player_id == 1 else next_game_state.player_2_hand
            
            if not player_hand_after:
                reward += 20.0
                
                opponent_hand_after = next_game_state.player_2_hand if player_id == 1 else next_game_state.player_1_hand
                opponent_rack_value = sum(self.letter_point_values.get(t.lower(), 0) for t in opponent_hand_after)
                reward += opponent_rack_value * 2  # Double bonus as per Scrabble rules
        
        # opponent rack penalty
        if tiles_in_bag == 0 and tiles_played_from_hand > 0:
            reward += tiles_played_from_hand * 3.0
            
            unseen_vector = encoded_state['unseen']
            
            opponent_penalty = 0

            for i in range(27):
                if unseen_vector[i] > 0:
                    if i == 26:
                        continue
                    letter = chr(ord('a') + i)
                    point_value = self.letter_point_values[letter]
                    opponent_penalty += unseen_vector[i] * point_value
            
            if opponent_penalty > 10:
                reward += 6.0
            elif opponent_penalty > 5:
                reward += 3.0
        
        # score maximization
        if tiles_in_bag < 7 and move and 'score' in move:
            if tiles_played_from_hand > 0:
                score_efficiency = move['score'] / tiles_played_from_hand
            if score_efficiency > 10:
                reward += move['score'] * 0.3
            else:
                reward += move['score'] * 0.15
        
        # blocking opponent going out
        if tiles_in_bag == 0:
            premium_squares_blocked = 0
            if 'positions' in move:
                for pos in move['positions']:
                    if not game_state.is_tile_present[pos][0]:
                        special = game_state.special_spaces.get(pos)
                        if special in ['triple_word', 'triple_letter', 'double_word']:
                            premium_squares_blocked += 1
            
            if premium_squares_blocked > 0:
                unseen_tiles_count = encoded_state['unseen'].sum()
                
                if unseen_tiles_count <= 3:
                    reward += premium_squares_blocked * 5.0
                elif unseen_tiles_count <= 5:
                    reward += premium_squares_blocked * 2.0
        
        # efficient tile usage
        if tiles_in_bag < 3:
            if tiles_played_from_hand >= 5:
                reward += 10.0
            elif tiles_played_from_hand >= 3:
                reward += 5.0
            
            current_hand = game_state.player_1_hand if player_id == 1 else game_state.player_2_hand
            if len(current_hand) >= 6 and tiles_played_from_hand < 2:
                reward -= 5.0
        
        # debug
        print(f'Endgame reward: {reward}')

        return reward