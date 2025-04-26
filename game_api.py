import random
import copy
from collections import Counter
import time
import json
import scrabble_valid_moves_rust

class Board_and_Variables:
    special_spaces = {
        'a1': 'triple_word', 'a2': None, 'a3': None, 'a4': 'double_letter', 'a5': None, 'a6': None, 'a7': None, 'a8': 'triple_word', 'a9': None, 'a10': None, 'a11': None, 'a12': 'double_letter', 'a13': None, 'a14': None, 'a15': 'triple_word',
        'b1': None, 'b2': 'double_word', 'b3': None, 'b4': None, 'b5': None, 'b6': 'triple_letter', 'b7': None, 'b8': None, 'b9': None, 'b10': 'triple_letter', 'b11': None, 'b12': None, 'b13': None, 'b14': 'double_word', 'b15': None,
        'c1': None, 'c2': None, 'c3': 'double_word', 'c4': None, 'c5': None, 'c6': None, 'c7': 'double_letter', 'c8': None, 'c9': 'double_letter', 'c10': None, 'c11': None, 'c12': None, 'c13': 'double_word', 'c14': None, 'c15': None,
        'd1': 'double_letter', 'd2': None, 'd3': None, 'd4': 'double_word', 'd5': None, 'd6': None, 'd7': None, 'd8': 'double_letter', 'd9': None, 'd10': None, 'd11': None, 'd12': 'double_word', 'd13': None, 'd14': None, 'd15': 'double_letter',
        'e1': None, 'e2': None, 'e3': None, 'e4': None, 'e5': 'double_word', 'e6': None, 'e7': None, 'e8': None, 'e9': None, 'e10': None, 'e11': 'double_word', 'e12': None, 'e13': None, 'e14': None, 'e15': None,
        'f1': None, 'f2': 'triple_letter', 'f3': None, 'f4': None, 'f5': None, 'f6': 'triple_letter', 'f7': None, 'f8': None, 'f9': None, 'f10': 'triple_letter', 'f11': None, 'f12': None, 'f13': None, 'f14': 'triple_letter', 'f15': None,
        'g1': None, 'g2': None, 'g3': 'double_letter', 'g4': None, 'g5': None, 'g6': None, 'g7': 'double_letter', 'g8': None, 'g9': 'double_letter', 'g10': None, 'g11': None, 'g12': None, 'g13': 'double_letter', 'g14': None, 'g15': None,
        'h1': 'triple_word', 'h2': None, 'h3': None, 'h4': 'double_letter', 'h5': None, 'h6': None, 'h7': None, 'h8': 'double_word', 'h9': None, 'h10': None, 'h11': None, 'h12': 'double_letter', 'h13': None, 'h14': None, 'h15': 'triple_word',
        'i1': None, 'i2': None, 'i3': 'double_letter', 'i4': None, 'i5': None, 'i6': None, 'i7': 'double_letter', 'i8': None, 'i9': 'double_letter', 'i10': None, 'i11': None, 'i12': None, 'i13': 'double_letter', 'i14': None, 'i15': None,
        'j1': None, 'j2': 'triple_letter', 'j3': None, 'j4': None, 'j5': None, 'j6': 'triple_letter', 'j7': None, 'j8': None, 'j9': None, 'j10': 'triple_letter', 'j11': None, 'j12': None, 'j13': None, 'j14': 'triple_letter', 'j15': None,
        'k1': None, 'k2': None, 'k3': None, 'k4': None, 'k5': 'double_word', 'k6': None, 'k7': None, 'k8': None, 'k9': None, 'k10': None, 'k11': 'double_word', 'k12': None, 'k13': None, 'k14': None, 'k15': None,
        'l1': 'double_letter', 'l2': None, 'l3': None, 'l4': 'double_word', 'l5': None, 'l6': None, 'l7': None, 'l8': 'double_letter', 'l9': None, 'l10': None, 'l11': None, 'l12': 'double_word', 'l13': None, 'l14': None, 'l15': 'double_letter',
        'm1': None, 'm2': None, 'm3': 'double_word', 'm4': None, 'm5': None, 'm6': None, 'm7': 'double_letter', 'm8': None, 'm9': 'double_letter', 'm10': None, 'm11': None, 'm12': None, 'm13': 'double_word', 'm14': None, 'm15': None,
        'n1': None, 'n2': 'double_word', 'n3': None, 'n4': None, 'n5': None, 'n6': 'triple_letter', 'n7': None, 'n8': None, 'n9': None, 'n10': 'triple_letter', 'n11': None, 'n12': None, 'n13': None, 'n14': 'double_word', 'n15': None,
        'o1': 'triple_word', 'o2': None, 'o3': None, 'o4': 'double_letter', 'o5': None, 'o6': None, 'o7': None, 'o8': 'triple_word', 'o9': None, 'o10': None, 'o11': None, 'o12': 'double_letter', 'o13': None, 'o14': None, 'o15': 'triple_word'
    }

    letter_counts = {
        'a': 9, 'b': 2, 'c': 2, 'd': 4, 'e': 12, 'f': 2, 'g': 3, 'h': 2, 'i': 9, 'j': 1, 'k': 1, 'l': 4, 'm': 2,
        'n': 6, 'o': 8, 'p': 2, 'q': 1, 'r': 6, 's': 4, 't': 6, 'u': 4, 'v': 2, 'w': 2, 'x': 1, 'y': 2, 'z': 1,
        '_': 2
    }

    letter_point_values = {
        'a': 1, 'b': 3, 'c': 3, 'd': 2, 'e': 1, 'f': 4, 'g': 2, 'h': 4, 'i': 1, 'j': 8, 'k': 5, 'l': 1, 'm': 3, 
        'n': 1, 'o': 1, 'p': 3, 'q': 10, 'r': 1, 's': 1, 't': 1, 'u': 1, 'v': 4, 'w': 4, 'x': 8, 'y': 4, 'z': 10,
        '_': 0
    }

class Scrabble_Game:
    def __init__(self):
        self.special_spaces = Board_and_Variables.special_spaces.copy()
        self.letter_counts = Board_and_Variables.letter_counts.copy()
        self.letter_point_values = Board_and_Variables.letter_point_values.copy()
        self.is_tile_present = {}
        self.player_1_hand = []
        self.player_2_hand = []
        self.player_1_score = 0
        self.player_2_score = 0
        self.current_player = 1
        self.game_over = False
        self.consecutive_passes = 0
        self.move_history = []
        
        self.blank_tile_assignments = {}

        self.words_set = set()
        dictionary_path = "word_dictionary.csv"

        try:
            with open(dictionary_path, 'r', encoding='utf-8') as f:
                header = next(f, None)
                if header:
                    print(f"game_api: Skipped dictionary header: {header.strip()}")
                else:
                    print(f"game_api: WARNING - Dictionary file '{dictionary_path}' appears empty.")

                for line in f:
                    word = line.strip().lower()
                    if word:
                        self.words_set.add(word)

            if not self.words_set and header:
                print(f"game_api: WARNING - Dictionary file '{dictionary_path}' loaded, but no words were added to words_set.")
            elif self.words_set:
                print(f"game_api: Dictionary loaded successfully into self.words_set ({len(self.words_set)} words).")

        except FileNotFoundError:
            print(f"game_api: FATAL ERROR - Dictionary file not found at '{dictionary_path}'")
            raise
        except Exception as e:
            print(f"game_api: An unexpected error occurred loading dictionary into self.words_set: {e}")
            raise
        
        for row_char in 'abcdefghijklmno':
            for col_num in range(1, 16):
                self.is_tile_present[f"{row_char}{col_num}"] = [False, False]
    
    def initialize_game(self):
        for pos in self.is_tile_present:
            self.is_tile_present[pos] = [False, False]
        
        self.player_1_score = 0
        self.player_2_score = 0
        self.player_1_hand = []
        self.player_2_hand = []
        
        self.current_player = 1
        self.game_over = False
        self.consecutive_passes = 0
        self.move_history = []
        self.blank_tile_assignments = {}
        
        self.letter_counts = Board_and_Variables.letter_counts.copy()
        
        self.draw_tiles(1, 7)
        self.draw_tiles(2, 7)
    
    def draw_tiles(self, player_id, num_tiles):
        """
        Args:
            player_id: ID of the player (1 or 2)
            num_tiles: Number of tiles to draw
            
        Returns:
            list: The tiles drawn
        """
        hand = self.player_1_hand if player_id == 1 else self.player_2_hand
        
        tiles_left = sum(self.letter_counts.values())
        
        num_to_draw = min(num_tiles, tiles_left)
        
        for _ in range(num_to_draw):
            available_letters = []
            for letter, count in self.letter_counts.items():
                if count > 0:
                    available_letters.extend([letter] * count)
            
            if available_letters:
                letter = random.choice(available_letters)
                hand.append(letter)
                self.letter_counts[letter] -= 1

        return hand[-num_to_draw:] if num_to_draw > 0 else []
    
    def validate_move(self, player_id, word, start, direction, blank_assignments=None):
        """
        Args:
            player_id: ID of the player (1 or 2)
            word: The word to play
            start: Starting position (e.g., 'h8')
            direction: Direction ('right' or 'down')
            blank_assignments: Dictionary mapping positions to assigned letters for blank tiles
            
        Returns:
            bool: Whether the move is valid
        """
        if self.game_over:
            return False
        
        if player_id != self.current_player:
            return False
        
        hand = self.player_1_hand if player_id == 1 else self.player_2_hand
        
        row_char = start[0]
        if row_char < 'a' or row_char > 'o':
            return False
        
        col_num = int(''.join(filter(str.isdigit, start)))
        if col_num < 1 or col_num > 15:
            return False
        
        word_length = len(word)
        if direction == 'right':
            end_col = col_num + word_length - 1
            if end_col > 15:
                return False
            spaces_word_occupies = [f"{row_char}{c}" for c in range(col_num, col_num + word_length)]
        else:
            end_row_ord = ord(row_char) + word_length - 1
            if end_row_ord > ord('o'):
                return False
            spaces_word_occupies = [f"{chr(r)}{col_num}" for r in range(ord(row_char), ord(row_char) + word_length)]
        
        hand_copy = hand.copy()
        
        for i, sp in enumerate(spaces_word_occupies):
            if self.is_tile_present[sp][0]:
                if self.is_tile_present[sp][1].lower() != word[i].lower():
                    return False
            else:
                letter = word[i].lower()
                
                is_blank = blank_assignments and sp in blank_assignments
                
                if is_blank:
                    if '_' in hand_copy:
                        hand_copy.remove('_')
                    else:
                        return False
                else:
                    if letter in hand_copy:
                        hand_copy.remove(letter)
                    elif '_' in hand_copy:
                        hand_copy.remove('_')
                    else:
                        return False
        
        connects_to_existing = False
        for sp in spaces_word_occupies:
            if self.is_tile_present[sp][0]:
                connects_to_existing = True
                break
            
            row_char = sp[0]
            col_num = int(''.join(filter(str.isdigit, sp)))
            
            if row_char > 'a':
                up_space = f"{chr(ord(row_char) - 1)}{col_num}"
                if self.is_tile_present[up_space][0]:
                    connects_to_existing = True
                    break
            
            if row_char < 'o':
                down_space = f"{chr(ord(row_char) + 1)}{col_num}"
                if self.is_tile_present[down_space][0]:
                    connects_to_existing = True
                    break
            
            if col_num > 1:
                left_space = f"{row_char}{col_num - 1}"
                if self.is_tile_present[left_space][0]:
                    connects_to_existing = True
                    break
            
            if col_num < 15:
                right_space = f"{row_char}{col_num + 1}"
                if self.is_tile_present[right_space][0]:
                    connects_to_existing = True
                    break
        
        if not connects_to_existing:
            if 'h8' not in spaces_word_occupies:
                return False
        
        if word.lower() not in self.words_set:
             return False

        temp_board = copy.deepcopy(self.is_tile_present)
        placed_letters_info = {}

        for i, sp in enumerate(spaces_word_occupies):
            if not temp_board[sp][0]:
                actual_letter = word[i].lower()
                temp_board[sp] = [True, actual_letter]
                placed_letters_info[sp] = actual_letter

        for sp, placed_letter in placed_letters_info.items():
            row_char = sp[0]
            col_num = int(''.join(filter(str.isdigit, sp)))

            if direction == 'right':
                start_row_ord = ord(row_char)
                while start_row_ord > ord('a'):
                    up_space = f"{chr(start_row_ord - 1)}{col_num}"
                    if up_space in temp_board and temp_board[up_space][0]:
                        start_row_ord -= 1
                    else:
                        break
                start_row = chr(start_row_ord)

                end_row_ord = ord(row_char)
                while end_row_ord < ord('o'):
                    down_space = f"{chr(end_row_ord + 1)}{col_num}"
                    if down_space in temp_board and temp_board[down_space][0]:
                        end_row_ord += 1
                    else:
                        break
                end_row = chr(end_row_ord)

                if ord(end_row) - ord(start_row) > 0:
                    v_word = []
                    for r_ord in range(ord(start_row), ord(end_row) + 1):
                        v_space = f"{chr(r_ord)}{col_num}"
                        v_word.append(temp_board[v_space][1])
                    v_word_str = ''.join(v_word)

                    if v_word_str.lower() not in self.words_set:
                        return False

            elif direction == 'down':
                start_col = col_num
                while start_col > 1:
                    left_space = f"{row_char}{start_col - 1}"
                    if left_space in temp_board and temp_board[left_space][0]:
                        start_col -= 1
                    else:
                        break

                end_col = col_num
                while end_col < 15:
                    right_space = f"{row_char}{end_col + 1}"
                    if right_space in temp_board and temp_board[right_space][0]:
                        end_col += 1
                    else:
                        break

                if end_col - start_col > 0:
                    h_word = []
                    for c in range(start_col, end_col + 1):
                        h_space = f"{row_char}{c}"
                        h_word.append(temp_board[h_space][1])
                    h_word_str = ''.join(h_word)

                    if h_word_str.lower() not in self.words_set:
                        return False
        
        return True
    
    def point_calculation(self, word, start, direction, blank_assignments=None):
        """
        Args:
            word: The word to play
            start: Starting position (e.g., 'h8')
            direction: Direction ('right' or 'down')
            blank_assignments: Dictionary mapping positions to assigned letters for blank tiles
            
        Returns:
            int: The score for the move
        """
        row_char = start[0]
        col_num = int(''.join(filter(str.isdigit, start)))
        
        if direction == 'right':
            spaces_word_occupies = [f"{row_char}{c}" for c in range(col_num, col_num + len(word))]
        else:
            spaces_word_occupies = [f"{chr(r)}{col_num}" for r in range(ord(row_char), ord(row_char) + len(word))]
        
        word_score = 0
        word_multipliers = []
        cross_word_scores = 0
        
        for i, sp in enumerate(spaces_word_occupies):
            letter = word[i].lower()
            letter_score = self.letter_point_values.get(letter, 0)
            letter_multiplier = 1
            
            is_blank = blank_assignments and sp in blank_assignments
            if is_blank:
                letter_score = 0
            
            if not self.is_tile_present[sp][0]:
                if sp in self.special_spaces and self.special_spaces[sp]:
                    special = self.special_spaces[sp]
                    if special == 'double_letter':
                        letter_multiplier = 2
                    elif special == 'triple_letter':
                        letter_multiplier = 3
                    elif special == 'double_word':
                        word_multipliers.append(2)
                    elif special == 'triple_word':
                        word_multipliers.append(3)
            
            word_score += letter_score * letter_multiplier
            
            if not self.is_tile_present[sp][0]:
                cross_word = self.get_cross_word(sp, direction, letter)
                if cross_word:
                    cross_word_score = self.calculate_cross_word_score(sp, cross_word, direction, is_blank)
                    cross_word_scores += cross_word_score
        
        for multiplier in word_multipliers:
            word_score *= multiplier
        
        total_score = word_score + cross_word_scores
        
        tiles_from_hand = sum(1 for sp in spaces_word_occupies if not self.is_tile_present[sp][0])
        if tiles_from_hand == 7:
            total_score += 50
        
        return total_score

    def get_cross_word(self, position, main_direction, letter):
        """
        Get the cross-word formed at a certain position
        
        Args:
            position: Position on the board
            main_direction: Direction of the main word ('right' or 'down')
            letter: Letter being placed at this position
            
        Returns:
            tuple: (word, start_pos) if a cross-word is formed, None otherwise
        """
        row_char = position[0]
        col_num = int(''.join(filter(str.isdigit, position)))
        
        cross_direction = 'down' if main_direction == 'right' else 'right'
        
        if cross_direction == 'down':
            start_row = row_char
            while ord(start_row) > ord('a'):
                up_pos = f"{chr(ord(start_row) - 1)}{col_num}"
                if up_pos in self.is_tile_present and self.is_tile_present[up_pos][0]:
                    start_row = chr(ord(start_row) - 1)
                else:
                    break
            
            end_row = row_char
            while ord(end_row) < ord('o'):
                down_pos = f"{chr(ord(end_row) + 1)}{col_num}"
                if down_pos in self.is_tile_present and self.is_tile_present[down_pos][0]:
                    end_row = chr(ord(end_row) + 1)
                else:
                    break
            
            cross_word = []
            for r in range(ord(start_row), ord(end_row) + 1):
                pos = f"{chr(r)}{col_num}"
                if pos == position:
                    cross_word.append(letter)
                else:
                    if self.is_tile_present[pos][0]:
                        cross_word.append(self.is_tile_present[pos][1])
                    else:
                        return None
                        
            if len(cross_word) > 1:
                return (''.join(cross_word), f"{start_row}{col_num}")
        else:
            start_col = col_num
            while start_col > 1:
                left_pos = f"{row_char}{start_col - 1}"
                if left_pos in self.is_tile_present and self.is_tile_present[left_pos][0]:
                    start_col -= 1
                else:
                    break
            
            end_col = col_num
            while end_col < 15:
                right_pos = f"{row_char}{end_col + 1}"
                if right_pos in self.is_tile_present and self.is_tile_present[right_pos][0]:
                    end_col += 1
                else:
                    break
            
            cross_word = []
            for c in range(start_col, end_col + 1):
                pos = f"{row_char}{c}"
                if pos == position:
                    cross_word.append(letter)
                else:
                    if self.is_tile_present[pos][0]:
                        cross_word.append(self.is_tile_present[pos][1])
                    else:
                        return None
            
            if len(cross_word) > 1:
                return (''.join(cross_word), f"{row_char}{start_col}")
        
        return None

    def calculate_cross_word_score(self, position, cross_word_data, main_direction, is_blank_at_intersection):
        """
        Args:
            position: Position of the intersecting letter
            cross_word_data: Tuple of (cross_word, start_position)
            main_direction: Direction of the main word
            is_blank_at_intersection: Whether a blank tile is used at the intersection
            
        Returns:
            int: Score for the cross-word
        """
        cross_word, start_pos = cross_word_data
        cross_direction = 'down' if main_direction == 'right' else 'right'
        
        score = 0
        word_multiplier = 1
        
        row_char = start_pos[0]
        col_num = int(''.join(filter(str.isdigit, start_pos)))
        
        if cross_direction == 'down':
            positions = [f"{chr(ord(row_char) + i)}{col_num}" for i in range(len(cross_word))]
        else:
            positions = [f"{row_char}{col_num + i}" for i in range(len(cross_word))]
        
        for i, pos in enumerate(positions):
            letter = cross_word[i].lower()
            letter_score = self.letter_point_values.get(letter, 0)
            
            if pos == position and is_blank_at_intersection:
                letter_score = 0
            
            if pos == position:
                if pos in self.special_spaces and self.special_spaces[pos]:
                    special = self.special_spaces[pos]
                    if special == 'double_letter':
                        letter_score *= 2
                    elif special == 'triple_letter':
                        letter_score *= 3
                    elif special == 'double_word':
                        word_multiplier *= 2
                    elif special == 'triple_word':
                        word_multiplier *= 3
            
            score += letter_score
        
        score *= word_multiplier
        
        return score

    def play_move(self, player_id, word, start, direction, blank_assignments=None):
        """
        Args:
            player_id: ID of the player making the move
            word: Word to play
            start: Starting position (e.g., 'h10')
            direction: Direction ('right' or 'down')
            blank_assignments: Dictionary mapping positions to assigned letters for blank tiles
        """
        if not self.validate_move(player_id, word, start, direction, blank_assignments):
            return False
        
        score = self.point_calculation(word, start, direction, blank_assignments)
        
        if player_id == 1:
            self.player_1_score += score
        else:
            self.player_2_score += score
        
        row_char = start[0]
        col_num = int(''.join(filter(str.isdigit, start)))
        
        if direction == 'right':
            spaces_word_occupies = [f"{row_char}{c}" for c in range(col_num, col_num + len(word))]
        else:
            spaces_word_occupies = [f"{chr(r)}{col_num}" for r in range(ord(row_char), ord(row_char) + len(word))]
        
        hand = self.player_1_hand if player_id == 1 else self.player_2_hand
        
        tiles_placed_count = 0

        for i, sp in enumerate(spaces_word_occupies):
            if not self.is_tile_present[sp][0]:
                tiles_placed_count += 1
                letter = word[i].lower()
                
                is_blank = blank_assignments and sp in blank_assignments
                
                removed_tile = False
                if is_blank:
                    if '_' in hand:
                        hand.remove('_')
                        self.blank_tile_assignments[sp] = letter
                        removed_tile = True
                    else:
                        print("Error in line 628.")
                else:
                    if letter in hand:
                        hand.remove(letter)
                        removed_tile = True
                    elif '_' in hand:
                        hand.remove('_')
                        self.blank_tile_assignments[sp] = letter
                        removed_tile = True
                    else:
                        print("error on line 636.")
                
                if removed_tile:
                    self.is_tile_present[sp] = [True, letter]
                else:
                    print(f"ERROR in play_move: Failed to find tile '{letter}' or blank for placement at {sp}")
                    return False
        
        if tiles_placed_count > 0:
            #  # debug
            #  print(f"Hand after playing move: {hand}")
             newly_drawn = self.draw_tiles(player_id, tiles_placed_count)
        else:
             print("DEBUG (Game API): No tiles placed, skipping draw.")
        
        move_data = {
            'player': player_id,
            'word': word,
            'start': start,
            'direction': direction,
            'score': score,
            'blank_assignments': blank_assignments
        }
        self.move_history.append(move_data)
        
        self.current_player = 3 - player_id
        
        self.consecutive_passes = 0
        
        return True

    def pass_turn(self, player_id):
        """
        Args:
            player_id: ID of the player passing
        """
        if player_id != self.current_player:
            return False
        
        move_data = {
            'player': player_id,
            'type': 'pass'
        }
        self.move_history.append(move_data)
        
        self.current_player = 3 - player_id
        
        self.consecutive_passes += 1
        
        if self.consecutive_passes >= 2:
            self.end_game()
        
        return True

    def exchange_tiles(self, player_id, tiles_to_exchange):
        """
        Args:
            player_id: ID of the player
            tiles_to_exchange: List of tiles to exchange
        """
        if player_id != self.current_player:
            return False
        
        tiles_in_bag = sum(self.letter_counts.values())
        if tiles_in_bag < len(tiles_to_exchange):
            return False
        
        hand = self.player_1_hand if player_id == 1 else self.player_2_hand
        
        hand_copy = hand.copy()
        for tile in tiles_to_exchange:
            if tile in hand_copy:
                hand_copy.remove(tile)
            else:
                return False
        
        for tile in tiles_to_exchange:
            hand.remove(tile)
            self.letter_counts[tile] += 1
        
        self.draw_tiles(player_id, len(tiles_to_exchange))
        
        move_data = {
            'player': player_id,
            'type': 'exchange',
            'tiles': tiles_to_exchange
        }
        self.move_history.append(move_data)
        
        self.current_player = 3 - player_id
        
        self.consecutive_passes = 0
        
        return True

    def end_game(self):
        self.game_over = True
        
        p1_penalty = sum(self.letter_point_values[t] for t in self.player_1_hand)
        p2_penalty = sum(self.letter_point_values[t] for t in self.player_2_hand)
        
        self.player_1_score -= p1_penalty
        self.player_2_score -= p2_penalty
        
        move_data = {
            'type': 'game_over',
            'player_1_score': self.player_1_score,
            'player_2_score': self.player_2_score
        }
        self.move_history.append(move_data)

    def get_all_valid_moves(self, player_id):
        """
        Args:
            player_id: ID of the player

        Returns:
            list: List of valid moves (list of dictionaries)
        """
        current_hand = self.player_1_hand if player_id == 1 else self.player_2_hand

        board_for_rust = {}
        for pos, data in self.is_tile_present.items():
            if data[0]:
                board_for_rust[pos] = data[1]

        try:
            valid_moves = scrabble_valid_moves_rust.get_valid_moves_rs(board_for_rust, current_hand)
        except Exception as e:
            print(f"Error calling Rust move generator: {e}")
            print(f"Board passed: {board_for_rust}")
            print(f"Hand passed: {current_hand}")
            valid_moves = []


        for move in valid_moves:
            if 'positions' not in move or not move['positions']:
                print(f"Warning: Move missing positions: {move}")
                start = move['start_space']
                direction = move['direction']
                word = move['word']
                positions = []
                try:
                    row_char = start[0]
                    col_num = int(''.join(filter(str.isdigit, start)))
                    if direction == 'right':
                        positions = [f"{row_char}{c}" for c in range(col_num, col_num + len(word))]
                    elif direction == 'down':
                        positions = [f"{chr(r)}{col_num}" for r in range(ord(row_char), ord(row_char) + len(word))]
                    move['positions'] = positions
                except Exception as pos_err:
                    print(f"Error reconstructing positions for move {move}: {pos_err}")
                    move['positions'] = []


            move['player_id'] = player_id
            move['score'] = int(move.get('score', 0))


        return valid_moves
