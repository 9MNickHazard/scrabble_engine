import pandas as pd
from collections import Counter
from itertools import permutations, product
import copy
import time
import multiprocessing
import os
import cProfile
import pstats

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

WORDS_SET = set()
WORD_PREFIXES = set()
DICT_LOADED = False

def load_dictionary():
    global WORDS_SET, WORD_PREFIXES, DICT_LOADED
    if DICT_LOADED:
        return
    try:
        script_dir = os.path.dirname(__file__)
        csv_path = os.path.join(script_dir, "word_dictionary.csv")

        if not os.path.exists(csv_path):
             print(f"ERROR: Dictionary file not found at: {csv_path}")
             WORDS_SET = set()
             WORD_PREFIXES = set()
             DICT_LOADED = False
             return

        with open(csv_path, 'r') as file:
            next(file)
            WORDS_SET = set(line.strip().lower() for line in file if line.strip())
        
        for word in WORDS_SET:
            for i in range(1, len(word) + 1):
                WORD_PREFIXES.add(word[:i])
        
        DICT_LOADED = True
    except Exception as e:
        print(f"Error loading dictionary: {e}")
        WORDS_SET = set()
        WORD_PREFIXES = set()
        DICT_LOADED = False

def init_worker():
    load_dictionary()

class ScrabbleAllValidMoves:
    def __init__(self, board_state, player_hand):
        self.special_spaces = Board_and_Variables.special_spaces.copy()
        self.is_tile_present = board_state
        self.letter_point_values = Board_and_Variables.letter_point_values.copy()
        if player_hand:
            self.player_hand = [tile.lower() for tile in player_hand]
        else:
            self.player_hand = []
        self.blank_tile_assignments = {}

        self.words_set = WORDS_SET
        self.word_prefixes = WORD_PREFIXES
        if not self.words_set:
            print(f"Warning: Dictionary not loaded in worker {os.getpid()}. Move generation will fail.")
    
    def process_starting_position(self, anchor):
        """
        Process an anchor square, considering it as ANY position in potential words.
        """
        valid_moves_local = []
        seen_moves_local = set()

        rack_letters = self.player_hand
        rack_length = len(rack_letters)

        for direction in ['right', 'down', 'left', 'up']:
            max_space = self.get_max_space(anchor, direction)
            
            for num_rack_tiles_used in range(1, min(rack_length + 1, max_space + 1)):
                for combo_tuple in set(permutations(rack_letters, num_rack_tiles_used)):
                    combo_list_original = list(combo_tuple)
                    blanks_in_combo = combo_list_original.count('_')

                    if blanks_in_combo > 0:
                        alphabet = 'abcdefghijklmnopqrstuvwxyz'
                        blank_sub_combinations = product(alphabet, repeat=blanks_in_combo)
                    else:
                        blank_sub_combinations = [None]

                    for substitutions in blank_sub_combinations:
                        combo_list_modified = []
                        current_blank_info = {}
                        sub_iter = iter(substitutions) if substitutions is not None else None
                        
                        for idx, tile in enumerate(combo_list_original):
                            if tile == '_':
                                assigned_letter = next(sub_iter)
                                combo_list_modified.append(assigned_letter)
                                current_blank_info[idx] = assigned_letter
                            else:
                                combo_list_modified.append(tile)
                        
                        for anchor_idx in range(len(combo_list_modified)):
                            self.try_place_with_anchor_at_position(
                                anchor, direction, combo_list_modified, 
                                combo_list_original, current_blank_info,
                                anchor_idx, valid_moves_local, seen_moves_local
                            )

        return valid_moves_local
    
    def get_max_space(self, anchor, direction):
        """
        Get maximum available space in the given direction from the anchor.
        """
        row_char = anchor[0]
        col_num = int(''.join(filter(str.isdigit, anchor)))
        
        if direction == 'right':
            return 15 - col_num + 1
        elif direction == 'down':
            return ord('o') - ord(row_char) + 1
        elif direction == 'left':
            return col_num
        elif direction == 'up':
            return ord(row_char) - ord('a') + 1
        return 0
    
    def try_place_with_anchor_at_position(self, anchor, direction, combo_list_modified,
                                         combo_list_original, current_blank_info,
                                         anchor_idx, valid_moves_local, seen_moves_local):
        """
        Try placing a word with the anchor square at a specific position in the word.
        """
        row_char = anchor[0]
        col_num = int(''.join(filter(str.isdigit, anchor)))
        
        start_row, start_col = ord(row_char), col_num
        
        if direction == 'right':
            start_col = col_num - anchor_idx
        elif direction == 'down':
            start_row = ord(row_char) - anchor_idx
        elif direction == 'left':
            start_col = col_num + anchor_idx - (len(combo_list_modified) - 1)
        elif direction == 'up':
            start_row = ord(row_char) + anchor_idx - (len(combo_list_modified) - 1)
        
        if (start_row < ord('a') or start_row > ord('o') or 
            start_col < 1 or start_col > 15):
            return
        
        start_pos = f"{chr(start_row)}{start_col}"
        
        self.validate_and_place_word(
            start_pos, direction, combo_list_modified, 
            combo_list_original, current_blank_info,
            valid_moves_local, seen_moves_local
        )
    
    def validate_and_place_word(self, start, direction, combo_list_modified,
                           combo_list_original, current_blank_info,
                           valid_moves_local, seen_moves_local):
        """
        Validates and places a word starting at 'start' in 'direction'.
        ALWAYS validates words in standard reading direction (left-to-right or top-to-bottom).
        """
        row_char = start[0]
        col_num = int(''.join(filter(str.isdigit, start)))
        
        formed_word_segments = []
        temp_rack_letters = combo_list_modified.copy()
        current_pos_str = start
        move_blank_assignments = {}
        tiles_placed_count = 0
        original_combo_idx = 0
        
        self.check_tiles_before_start(start, direction, formed_word_segments)
        
        current_row, current_col = ord(row_char), col_num
        
        while True:
            if (current_row < ord('a') or current_row > ord('o') or 
                current_col < 1 or current_col > 15):
                break
                
            current_pos_str = f"{chr(current_row)}{current_col}"
            if current_pos_str not in self.is_tile_present:
                break
            
            if self.is_tile_present[current_pos_str][0]:
                formed_word_segments.append((self.is_tile_present[current_pos_str][1], current_pos_str, False))
            elif temp_rack_letters:
                letter_to_place = temp_rack_letters.pop(0)
                formed_word_segments.append((letter_to_place, current_pos_str, True))
                tiles_placed_count += 1
                
                if original_combo_idx < len(combo_list_original) and combo_list_original[original_combo_idx] == '_':
                    move_blank_assignments[current_pos_str] = letter_to_place
                
                original_combo_idx += 1
            else:
                break
            

            if direction in ['left', 'up']:
                current_word_str = "".join([seg[0] for seg in formed_word_segments[::-1]])
            else:
                current_word_str = "".join([seg[0] for seg in formed_word_segments])
                
            if not self.could_form_valid_word(current_word_str):
                return
            
            if direction == 'right':
                current_col += 1
            elif direction == 'down':
                current_row += 1
            elif direction == 'left':
                current_col -= 1
            elif direction == 'up':
                current_row -= 1
            
            next_pos = f"{chr(current_row)}{current_col}"
            if (not temp_rack_letters and 
                (current_row < ord('a') or current_row > ord('o') or 
                current_col < 1 or current_col > 15 or 
                next_pos not in self.is_tile_present or
                not self.is_tile_present[next_pos][0])):
                break
        

        self.check_tiles_after_end(current_row, current_col, direction, formed_word_segments)
        
        if direction in ['left', 'up']:
            final_word = "".join([seg[0] for seg in formed_word_segments[::-1]])
            final_positions = [seg[1] for seg in formed_word_segments[::-1]]
            actual_start_space = final_positions[0]
            display_direction = direction
        else:
            final_word = "".join([seg[0] for seg in formed_word_segments])
            final_positions = [seg[1] for seg in formed_word_segments]
            actual_start_space = final_positions[0]
            display_direction = direction
        
        if len(final_word) < 2:
            return
            
        if final_word.lower() not in self.words_set:
            return
            
        if tiles_placed_count == 0:
            return
        
        is_first_move = not any(tile_info[0] for pos, tile_info in self.is_tile_present.items() if pos != 'h8')
        if is_first_move and 'h8' not in final_positions:
            return
            
        has_existing_tile_in_word = any(not segment[2] for segment in formed_word_segments)
        
        if not has_existing_tile_in_word and not is_first_move:
            has_valid_connection = False
            
            for i, (letter, pos, is_from_rack) in enumerate(formed_word_segments):
                if not is_from_rack:
                    continue
                    
                row_char = pos[0]
                col_num = int(pos[1:])
                
                for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    adj_row = chr(ord(row_char) + dr)
                    adj_col = col_num + dc
                    
                    if 'a' <= adj_row <= 'o' and 1 <= adj_col <= 15:
                        adj_pos = f"{adj_row}{adj_col}"
                        if adj_pos in self.is_tile_present and self.is_tile_present[adj_pos][0]:
                            has_valid_connection = True
                            break
                
                if has_valid_connection:
                    break
                    
            if not has_valid_connection:
                return

        cross_words_valid, cross_words_details = self.check_cross_words(
            final_positions, final_word, formed_word_segments,
            direction in ['left', 'up']
        )
        if not cross_words_valid:
            return
        
        score = self.point_calculation(
            final_word, final_positions, formed_word_segments, 
            move_blank_assignments, cross_words_details
        )
        
        move_key = f"{final_word}_{actual_start_space}_{display_direction}"
        
        if move_key not in seen_moves_local:
            seen_moves_local.add(move_key)
            move_info = {
                'word': final_word,
                'start_space': actual_start_space,
                'direction': display_direction,
                'score': score,
                'positions': final_positions,
                'cross_words': cross_words_details
            }
            if move_blank_assignments:
                move_info['blank_assignments'] = move_blank_assignments
            valid_moves_local.append(move_info)
    
    def check_tiles_before_start(self, start, direction, formed_word_segments):
        """
        Check for existing tiles before the start position.
        """
        row_char = start[0]
        col_num = int(''.join(filter(str.isdigit, start)))
        
        temp_row, temp_col = ord(row_char), col_num
        
        while True:
            if direction == 'right':
                temp_col -= 1
            elif direction == 'down':
                temp_row -= 1
            elif direction == 'left':
                temp_col += 1
            elif direction == 'up':
                temp_row += 1
                
            if (temp_row < ord('a') or temp_row > ord('o') or 
                temp_col < 1 or temp_col > 15):
                break
                
            prev_pos_str = f"{chr(temp_row)}{temp_col}"
            if prev_pos_str not in self.is_tile_present:
                break
                
            if self.is_tile_present[prev_pos_str][0]:
                if direction in ['right', 'down']:
                    formed_word_segments.insert(0, (self.is_tile_present[prev_pos_str][1], prev_pos_str, False))
                else:
                    formed_word_segments.append((self.is_tile_present[prev_pos_str][1], prev_pos_str, False))
            else:
                break
    
    def check_tiles_after_end(self, current_row, current_col, direction, formed_word_segments):
        """
        Check for existing tiles after the end position.
        """
        while True:
            if (current_row < ord('a') or current_row > ord('o') or 
                current_col < 1 or current_col > 15):
                break
                
            next_pos_str = f"{chr(current_row)}{current_col}"
            if next_pos_str not in self.is_tile_present:
                break
                
            if self.is_tile_present[next_pos_str][0]:
                formed_word_segments.append((self.is_tile_present[next_pos_str][1], next_pos_str, False))
                
                if direction == 'right':
                    current_col += 1
                elif direction == 'down':
                    current_row += 1
                elif direction == 'left':
                    current_col -= 1
                elif direction == 'up':
                    current_row -= 1
            else:
                break

    def check_cross_words(self, main_word_positions, main_word_str, formed_word_segments, is_reversed=False):
        """ 
        Checks words formed perpendicular to the main word.
        Always validates words in standard reading direction (left-to-right or top-to-bottom).
        """
        cross_words_details = []
        temp_board = copy.deepcopy(self.is_tile_present)

        position_to_segment = {}
        for i, (letter, pos, is_from_rack) in enumerate(formed_word_segments):
            position_to_segment[pos] = (letter, is_from_rack)

        for i, pos in enumerate(main_word_positions):
            letter = main_word_str[i]
            if pos in position_to_segment and position_to_segment[pos][1]:
                temp_board[pos] = [True, letter.lower()]

        for pos in main_word_positions:
            if pos not in position_to_segment or not position_to_segment[pos][1]:
                continue

            row_char = pos[0]
            col_num = int(pos[1:])
            
            is_horizontal = False
            if len(main_word_positions) > 1:
                is_horizontal = main_word_positions[0][0] == main_word_positions[1][0]
            
            if is_horizontal:
                start_r_ord = ord(row_char)
                while start_r_ord > ord('a') and temp_board[f"{chr(start_r_ord - 1)}{col_num}"][0]:
                    start_r_ord -= 1
                end_r_ord = ord(row_char)
                while end_r_ord < ord('o') and temp_board[f"{chr(end_r_ord + 1)}{col_num}"][0]:
                    end_r_ord += 1

                if start_r_ord != end_r_ord:
                    cross_word_list = [temp_board[f"{chr(r)}{col_num}"][1] for r in range(start_r_ord, end_r_ord + 1)]
                    cross_word = "".join(cross_word_list)
                    if cross_word.lower() not in self.words_set:
                        return False, []
                    cross_words_details.append({
                        'word': cross_word,
                        'start': f"{chr(start_r_ord)}{col_num}",
                        'direction': 'down',
                        'positions': [f"{chr(r)}{col_num}" for r in range(start_r_ord, end_r_ord + 1)],
                        'intersection_pos': pos
                    })
            else:
                start_c = col_num
                while start_c > 1 and temp_board[f"{row_char}{start_c - 1}"][0]:
                    start_c -= 1
                end_c = col_num
                while end_c < 15 and temp_board[f"{row_char}{end_c + 1}"][0]:
                    end_c += 1

                if start_c != end_c:
                    cross_word_list = [temp_board[f"{row_char}{c}"][1] for c in range(start_c, end_c + 1)]
                    cross_word = "".join(cross_word_list)
                    if cross_word.lower() not in self.words_set:
                        return False, []
                    cross_words_details.append({
                        'word': cross_word,
                        'start': f"{row_char}{start_c}",
                        'direction': 'right',
                        'positions': [f"{row_char}{c}" for c in range(start_c, end_c + 1)],
                        'intersection_pos': pos
                    })

        return True, cross_words_details

    def could_form_valid_word(self, prefix):
        """ Checks if a prefix could form valid word. """
        return prefix.lower() in self.word_prefixes

    def point_calculation(self, main_word_str, main_word_positions, formed_word_segments, blank_assignments, cross_words_details):
        total_score = 0
        main_word_score = 0
        word_multipliers = []

        for i, (letter, pos, is_from_rack) in enumerate(formed_word_segments):
            letter_lower = letter.lower()
            letter_base_score = self.letter_point_values.get(letter_lower, 0)
            letter_multiplier = 1

            is_blank = pos in blank_assignments
            if is_blank:
                letter_base_score = 0

            if is_from_rack and pos in self.special_spaces:
                special = self.special_spaces[pos]
                if special == 'double_letter': letter_multiplier = 2
                elif special == 'triple_letter': letter_multiplier = 3
                elif special == 'double_word': word_multipliers.append(2)
                elif special == 'triple_word': word_multipliers.append(3)

            main_word_score += letter_base_score * letter_multiplier

        for mult in word_multipliers:
            main_word_score *= mult
        total_score += main_word_score

        for cross_info in cross_words_details:
            cross_word_score = 0
            cross_multipliers = []
            intersection_pos = cross_info['intersection_pos']

            for j, cross_pos in enumerate(cross_info['positions']):
                cross_letter = cross_info['word'][j].lower()
                cross_letter_base_score = self.letter_point_values.get(cross_letter, 0)
                cross_letter_multiplier = 1

                is_intersecting_blank = intersection_pos == cross_pos and intersection_pos in blank_assignments
                if is_intersecting_blank:
                    cross_letter_base_score = 0

                if cross_pos == intersection_pos and cross_pos in self.special_spaces:
                     special = self.special_spaces[cross_pos]
                     if special == 'double_letter': cross_letter_multiplier = 2
                     elif special == 'triple_letter': cross_letter_multiplier = 3
                     elif special == 'double_word': cross_multipliers.append(2)
                     elif special == 'triple_word': cross_multipliers.append(3)

                cross_word_score += cross_letter_base_score * cross_letter_multiplier

            for mult in cross_multipliers:
                cross_word_score *= mult
            total_score += cross_word_score

        tiles_from_rack_count = sum(1 for _, _, is_from_rack in formed_word_segments if is_from_rack)
        if tiles_from_rack_count == 7:
            total_score += 50

        return total_score

    def get_anchor_squares(self):
        """ Finds all empty squares adjacent to existing tiles. 'h8' if board is empty. """
        if not any(tile_info[0] for tile_info in self.is_tile_present.values()):
            return ['h8']

        anchors = set()
        for pos, (is_present, _) in self.is_tile_present.items():
             if not is_present: continue

             row = pos[0]
             col = int(pos[1:])

             for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                 nr_ord = ord(row) + dr
                 nc = col + dc
                 if ord('a') <= nr_ord <= ord('o') and 1 <= nc <= 15:
                     npos = f"{chr(nr_ord)}{nc}"
                     if npos in self.is_tile_present and not self.is_tile_present[npos][0]:
                         anchors.add(npos)
        return list(anchors)

    def get_all_valid_moves(self):
        """ Gets all valid moves using anchor squares and cpu multiprocessing. """
        global DICT_LOADED
        if not DICT_LOADED:
             print("Error: Dictionary failed to load. Cannot find moves.")
             load_dictionary()
             if not DICT_LOADED:
                 return []

        anchor_squares = self.get_anchor_squares()
        if not anchor_squares:
            return []

        num_processes = max(1, os.cpu_count() - 1) if os.cpu_count() else 1

        pool_args = [(copy.deepcopy(self.is_tile_present),
                      self.player_hand[:],
                      anchor)
                     for anchor in anchor_squares]

        results = []
        try:
            with multiprocessing.Pool(processes=num_processes, initializer=init_worker) as pool:
                 results = pool.map(process_anchor_wrapper, pool_args)
        except Exception as e:
            print(f"Multiprocessing error: {e}")
            results = []

        valid_moves = []
        seen_move_keys = set()
        for sublist in results:
            for move in sublist:
                move_key = (move['word'], move['start_space'], move['direction'])
                if move_key not in seen_move_keys:
                    seen_move_keys.add(move_key)
                    valid_moves.append(move)

        valid_moves.sort(key=lambda x: x['score'], reverse=True)

        return valid_moves
    



def process_anchor_wrapper(args):
    board_state_copy, player_hand_copy, anchor = args
    temp_instance = ScrabbleAllValidMoves(board_state_copy, player_hand_copy)
    return temp_instance.process_starting_position(anchor)


# for testing
if __name__ == "__main__":
    multiprocessing.freeze_support()

    # debug
    profiler = cProfile.Profile()
    profiler.enable()

    load_dictionary()
    if not DICT_LOADED:
        print("Exiting: Dictionary failed to load")
        exit()
    else:
         print(f"Dictionary loaded successfully: {len(WORDS_SET)} words, {len(WORD_PREFIXES)} prefixes.")

    start_time = time.time()

    board_state = {
        'a1': [False, False], 'a2': [False, False], 'a3': [False, False], 'a4': [False, False], 'a5': [False, False], 'a6': [False, False], 'a7': [False, False], 'a8': [False, False], 'a9': [False, False], 'a10': [False, False], 'a11': [False, False], 'a12': [False, False], 'a13': [False, False], 'a14': [False, False], 'a15': [False, False],
        'b1': [False, False], 'b2': [False, False], 'b3': [False, False], 'b4': [False, False], 'b5': [False, False], 'b6': [False, False], 'b7': [False, False], 'b8': [False, False], 'b9': [False, False], 'b10': [False, False], 'b11': [False, False], 'b12': [False, False], 'b13': [False, False], 'b14': [False, False], 'b15': [False, False],
        'c1': [False, False], 'c2': [False, False], 'c3': [False, False], 'c4': [False, False], 'c5': [False, False], 'c6': [False, False], 'c7': [False, False], 'c8': [False, False], 'c9': [False, False], 'c10': [False, False], 'c11': [False, False], 'c12': [False, False], 'c13': [False, False], 'c14': [False, False], 'c15': [False, False],
        'd1': [False, False], 'd2': [False, False], 'd3': [False, False], 'd4': [False, False], 'd5': [False, False], 'd6': [False, False], 'd7': [False, False], 'd8': [False, False], 'd9': [False, False], 'd10': [False, False], 'd11': [False, False], 'd12': [False, False], 'd13': [False, False], 'd14': [False, False], 'd15': [False, False],
        'e1': [False, False], 'e2': [False, False], 'e3': [False, False], 'e4': [False, False], 'e5': [False, False], 'e6': [False, False], 'e7': [False, False], 'e8': [False, False], 'e9': [False, False], 'e10': [False, False], 'e11': [False, False], 'e12': [False, False], 'e13': [False, False], 'e14': [False, False], 'e15': [False, False],
        'f1': [False, False], 'f2': [False, False], 'f3': [False, False], 'f4': [False, False], 'f5': [False, False], 'f6': [False, False], 'f7': [False, False], 'f8': [False, False], 'f9': [False, False], 'f10': [False, False], 'f11': [False, False], 'f12': [False, False], 'f13': [False, False], 'f14': [False, False], 'f15': [False, False],
        'g1': [False, False], 'g2': [False, False], 'g3': [False, False], 'g4': [False, False], 'g5': [False, False], 'g6': [False, False], 'g7': [False, False], 'g8': [False, False], 'g9': [True, 'b'], 'g10': [False, False], 'g11': [False, False], 'g12': [False, False], 'g13': [False, False], 'g14': [False, False], 'g15': [False, False],
        'h1': [False, False], 'h2': [False, False], 'h3': [False, False], 'h4': [False, False], 'h5': [False, False], 'h6': [False, False], 'h7': [False, False], 'h8': [True, 'r'], 'h9': [True, 'e'], 'h10': [True, 's'], 'h11': [True, 'e'], 'h12': [True, 't'], 'h13': [False, False], 'h14': [False, False], 'h15': [False, False],
        'i1': [False, False], 'i2': [False, False], 'i3': [False, False], 'i4': [True, 't'], 'i5': [True, 'a'], 'i6': [True, 'b'], 'i7': [False, False], 'i8': [False, False], 'i9': [True, 'l'], 'i10': [False, False], 'i11': [False, False], 'i12': [False, False], 'i13': [False, False], 'i14': [False, False], 'i15': [False, False],
        'j1': [False, False], 'j2': [False, False], 'j3': [False, False], 'j4': [False, False], 'j5': [False, False], 'j6': [True, 'e'], 'j7': [False, False], 'j8': [False, False], 'j9': [True, 'i'], 'j10': [False, False], 'j11': [False, False], 'j12': [False, False], 'j13': [False, False], 'j14': [False, False], 'j15': [False, False],
        'k1': [False, False], 'k2': [False, False], 'k3': [False, False], 'k4': [False, False], 'k5': [False, False], 'k6': [True, 'r'], 'k7': [False, False], 'k8': [False, False], 'k9': [True, 'e'], 'k10': [False, False], 'k11': [False, False], 'k12': [False, False], 'k13': [False, False], 'k14': [False, False], 'k15': [False, False],
        'l1': [False, False], 'l2': [False, False], 'l3': [False, False], 'l4': [False, False], 'l5': [False, False], 'l6': [True, 'a'], 'l7': [False, False], 'l8': [False, False], 'l9': [True, 'v'], 'l10': [False, False], 'l11': [False, False], 'l12': [False, False], 'l13': [False, False], 'l14': [False, False], 'l15': [False, False],
        'm1': [False, False], 'm2': [False, False], 'm3': [False, False], 'm4': [False, False], 'm5': [False, False], 'm6': [True, 't'], 'm7': [True, 'a'], 'm8': [True, 't'], 'm9': [True, 'e'], 'm10': [False, False], 'm11': [False, False], 'm12': [False, False], 'm13': [False, False], 'm14': [False, False], 'm15': [False, False],
        'n1': [False, False], 'n2': [False, False], 'n3': [False, False], 'n4': [False, False], 'n5': [False, False], 'n6': [True, 'e'], 'n7': [False, False], 'n8': [False, False], 'n9': [False, False], 'n10': [False, False], 'n11': [False, False], 'n12': [False, False], 'n13': [False, False], 'n14': [False, False], 'n15': [False, False],
        'o1': [False, False], 'o2': [False, False], 'o3': [False, False], 'o4': [False, False], 'o5': [False, False], 'o6': [False, False], 'o7': [False, False], 'o8': [False, False], 'o9': [False, False], 'o10': [False, False], 'o11': [False, False], 'o12': [False, False], 'o13': [False, False], 'o14': [False, False], 'o15': [False, False]
    }

    player_hand = ['a', 'e', 'd', '_', 'c', 't', 's']

    print(f"Testing with hand: {player_hand}")
    move_generator = ScrabbleAllValidMoves(board_state, player_hand)

    valid_moves = move_generator.get_all_valid_moves()

    end_time = time.time()
    execution_time_s = end_time - start_time
    print(f"\nFound {len(valid_moves)} valid moves.")
    print(f"Execution time: {execution_time_s:.3f} seconds")

    print("\nTop 20 Moves (by score):")
    for i, move in enumerate(valid_moves[:20]):
        print(f"  {i+1}. {move['word']} @ {move['start_space']} ({move['direction']}) - Score: {move['score']}", end="")
        if 'blank_assignments' in move:
            blanks_str = ", ".join([f"{pos}={letter}" for pos, letter in move['blank_assignments'].items()])
            print(f" (Blanks: {blanks_str})")
        else:
            print()
    
    # debug
    profiler.disable()
    profiler.dump_stats('python_valid_cprofile.prof')


    