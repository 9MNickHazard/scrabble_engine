import pandas as pd
from collections import Counter
from itertools import permutations, product
import copy
import time
import multiprocessing
import os

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

        words_df = pd.read_csv(csv_path)
        WORDS_SET = set(str(word).lower() for word in words_df['legal_scrabble_words'].values)
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


    def process_starting_position(self, start):
        valid_moves_local = []
        seen_moves_local = set()

        rack_letters = self.player_hand
        rack_length = len(rack_letters)
        blank_count = rack_letters.count('_')
        non_blank_rack = [letter for letter in rack_letters if letter != '_']

        for direction in ['right', 'down']:
            max_len_this_direction = self.get_max_line_length(start, direction)

            for num_rack_tiles_used in range(1, rack_length + 1):
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
                        blank_indices_in_combo = []

                        for idx, tile in enumerate(combo_list_original):
                            if tile == '_':
                                assigned_letter = next(sub_iter)
                                combo_list_modified.append(assigned_letter)
                                current_blank_info[idx] = assigned_letter
                                blank_indices_in_combo.append(idx)
                            else:
                                combo_list_modified.append(tile)

                        self.try_place_combo(start, direction, combo_list_modified,
                                             combo_list_original, current_blank_info,
                                             valid_moves_local, seen_moves_local)

        return valid_moves_local

    def try_place_combo(self, start, direction, combo_list_modified,
                        combo_list_original, current_blank_info,
                        valid_moves_local, seen_moves_local):
        """ Tries to place a specific combo starting at 'start' in 'direction'. """
        row_char = start[0]
        col_num = int(''.join(filter(str.isdigit, start)))

        temp_rack_letters = combo_list_modified[:]
        formed_word_segments = []
        current_pos_str = start
        word_len = 0
        move_blank_assignments = {}

        temp_r, temp_c = ord(row_char), col_num
        seg_before = []
        while True:
            if direction == 'right': temp_c -= 1
            else: temp_r -= 1

            if temp_c < 1 or temp_r < ord('a'): break

            prev_pos_str = f"{chr(temp_r)}{temp_c}"
            if prev_pos_str not in self.is_tile_present:
                 break
            if self.is_tile_present[prev_pos_str][0]:
                seg_before.insert(0, (self.is_tile_present[prev_pos_str][1], prev_pos_str, False))
            else:
                break
        formed_word_segments.extend(seg_before)

        tiles_placed_count = 0
        original_combo_idx = 0
        while True:
            if ord(row_char) < ord('a') or ord(row_char) > ord('o') or col_num < 1 or col_num > 15:
                return

            current_pos_str = f"{row_char}{col_num}"
            if current_pos_str not in self.is_tile_present:
                 return

            if self.is_tile_present[current_pos_str][0]:
                formed_word_segments.append((self.is_tile_present[current_pos_str][1], current_pos_str, False))
            elif temp_rack_letters:
                letter_to_place = temp_rack_letters.pop(0)
                formed_word_segments.append((letter_to_place, current_pos_str, True))
                tiles_placed_count += 1
                if original_combo_idx in current_blank_info:
                    move_blank_assignments[current_pos_str] = current_blank_info[original_combo_idx]
                original_combo_idx += 1
            else:
                break

            current_word_str = "".join([seg[0] for seg in formed_word_segments])
            if not self.could_form_valid_word(current_word_str):
                 return

            if direction == 'right': col_num += 1
            else: row_char = chr(ord(row_char) + 1)

            if not temp_rack_letters and (ord(row_char) > ord('o') or col_num > 15 or not self.is_tile_present[f"{row_char}{col_num}"][0]):
                break


        while True:
             if ord(row_char) > ord('o') or col_num > 15: break

             next_pos_str = f"{row_char}{col_num}"
             if next_pos_str not in self.is_tile_present: break

             if self.is_tile_present[next_pos_str][0]:
                 formed_word_segments.append((self.is_tile_present[next_pos_str][1], next_pos_str, False))
                 if direction == 'right': col_num += 1
                 else: row_char = chr(ord(row_char) + 1)
             else:
                 break

        final_word = "".join([seg[0] for seg in formed_word_segments])
        final_positions = [seg[1] for seg in formed_word_segments]

        if len(final_word) < 2: return

        if final_word.lower() not in self.words_set: return

        if tiles_placed_count == 0: return

        is_first_move = not any(tile_info[0] for pos, tile_info in self.is_tile_present.items() if pos != 'h8')
        if is_first_move and 'h8' not in final_positions: return

        cross_words_valid, cross_words_details = self.check_cross_words(final_positions, final_word, formed_word_segments)
        if not cross_words_valid: return

        score = self.point_calculation(final_word, final_positions, formed_word_segments, move_blank_assignments, cross_words_details)

        actual_start_space = final_positions[0]
        move_key = f"{final_word}_{actual_start_space}_{direction}"

        if move_key not in seen_moves_local:
            seen_moves_local.add(move_key)
            move_info = {
                'word': final_word,
                'start_space': actual_start_space,
                'direction': direction,
                'score': score,
                'positions': final_positions,
                'cross_words': cross_words_details
            }
            if move_blank_assignments:
                move_info['blank_assignments'] = move_blank_assignments
            valid_moves_local.append(move_info)


    def check_cross_words(self, main_word_positions, main_word_str, formed_word_segments):
        """ Checks words formed perpendicular to the main word. """
        cross_words_details = []
        temp_board = copy.deepcopy(self.is_tile_present)

        for i, pos in enumerate(main_word_positions):
             is_from_rack = formed_word_segments[i][2]
             if is_from_rack:
                 temp_board[pos] = [True, main_word_str[i].lower()]

        for i, (letter, pos, is_from_rack) in enumerate(formed_word_segments):
            if not is_from_rack: continue

            row_char = pos[0]
            col_num = int(pos[1:])
            main_direction = 'right' if len(main_word_positions) > 1 and main_word_positions[0][0] == main_word_positions[1][0] else 'down'

            if main_direction == 'right':
                start_r_ord = ord(row_char)
                while start_r_ord > ord('a') and temp_board[f"{chr(start_r_ord - 1)}{col_num}"][0]:
                    start_r_ord -= 1
                end_r_ord = ord(row_char)
                while end_r_ord < ord('o') and temp_board[f"{chr(end_r_ord + 1)}{col_num}"][0]:
                    end_r_ord += 1

                if start_r_ord != end_r_ord:
                    cross_word_list = [temp_board[f"{chr(r)}{col_num}"][1] for r in range(start_r_ord, end_r_ord + 1)]
                    cross_word = "".join(cross_word_list)
                    if cross_word.lower() not in self.words_set: return False, []
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
                    if cross_word.lower() not in self.words_set: return False, []
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

    def get_max_line_length(self, start_space: str, direction: str) -> int:
        """ Calculates the maximum number of squares available in a line from start_space. """
        row_char = start_space[0]
        col_num = int(''.join(filter(str.isdigit, start_space)))

        if direction == 'right':
            return 15 - col_num + 1
        else:
            return ord('o') - ord(row_char) + 1


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
    load_dictionary()
    if not DICT_LOADED:
        print("Exiting: Dictionary failed to load")
        exit()
    else:
         print(f"Dictionary loaded successfully: {len(WORDS_SET)} words, {len(WORD_PREFIXES)} prefixes.")


    # start_time = time.time()

    board_state = {
        'a1': [False, False], 'a2': [False, False], 'a3': [False, False], 'a4': [False, False], 'a5': [False, False], 'a6': [False, False], 'a7': [False, False], 'a8': [False, False], 'a9': [False, False], 'a10': [False, False], 'a11': [False, False], 'a12': [False, False], 'a13': [False, False], 'a14': [False, False], 'a15': [False, False],
        'b1': [False, False], 'b2': [False, False], 'b3': [False, False], 'b4': [False, False], 'b5': [False, False], 'b6': [False, False], 'b7': [False, False], 'b8': [False, False], 'b9': [False, False], 'b10': [False, False], 'b11': [False, False], 'b12': [False, False], 'b13': [False, False], 'b14': [False, False], 'b15': [False, False],
        'c1': [False, False], 'c2': [False, False], 'c3': [False, False], 'c4': [False, False], 'c5': [False, False], 'c6': [False, False], 'c7': [False, False], 'c8': [False, False], 'c9': [False, False], 'c10': [False, False], 'c11': [False, False], 'c12': [False, False], 'c13': [False, False], 'c14': [False, False], 'c15': [False, False],
        'd1': [False, False], 'd2': [False, False], 'd3': [False, False], 'd4': [False, False], 'd5': [False, False], 'd6': [False, False], 'd7': [False, False], 'd8': [False, False], 'd9': [False, False], 'd10': [False, False], 'd11': [False, False], 'd12': [False, False], 'd13': [False, False], 'd14': [False, False], 'd15': [False, False],
        'e1': [False, False], 'e2': [False, False], 'e3': [False, False], 'e4': [False, False], 'e5': [False, False], 'e6': [False, False], 'e7': [False, False], 'e8': [False, False], 'e9': [False, False], 'e10': [False, False], 'e11': [False, False], 'e12': [False, False], 'e13': [False, False], 'e14': [False, False], 'e15': [False, False],
        'f1': [False, False], 'f2': [False, False], 'f3': [False, False], 'f4': [False, False], 'f5': [False, False], 'f6': [False, False], 'f7': [False, False], 'f8': [False, False], 'f9': [False, False], 'f10': [False, False], 'f11': [False, False], 'f12': [False, False], 'f13': [False, False], 'f14': [False, False], 'f15': [False, False],
        'g1': [False, False], 'g2': [False, False], 'g3': [False, False], 'g4': [False, False], 'g5': [False, False], 'g6': [False, False], 'g7': [False, False], 'g8': [False, False], 'g9': [False, False], 'g10': [True, 'd'], 'g11': [False, False], 'g12': [True, 'z'], 'g13': [False, False], 'g14': [False, False], 'g15': [True, 't'],
        'h1': [False, False], 'h2': [False, False], 'h3': [False, False], 'h4': [False, False], 'h5': [False, False], 'h6': [False, False], 'h7': [False, False], 'h8': [True, 's'], 'h9': [True, 'a'], 'h10': [True, 'u'], 'h11': [True, 't'], 'h12': [True, 'o'], 'h13': [True, 'i'], 'h14': [True, 'r'], 'h15': [True, 'e'], # Example "word" at h8
        'i1': [False, False], 'i2': [False, False], 'i3': [False, False], 'i4': [False, False], 'i5': [False, False], 'i6': [False, False], 'i7': [False, False], 'i8': [True, 'a'], 'i9': [False, False], 'i10': [True, 'r'], 'i11': [True, 'o'], 'i12': [True, 'o'], 'i13': [False, False], 'i14': [True, 'o'], 'i15': [False, False],
        'j1': [False, False], 'j2': [False, False], 'j3': [False, False], 'j4': [False, False], 'j5': [False, False], 'j6': [False, False], 'j7': [True, 'g'], 'j8': [True, 'i'], 'j9': [False, False], 'j10': [True, 'i'], 'j11': [False, False], 'j12': [True, 'g'], 'j13': [True, 'e'], 'j14': [True, 'n'], 'j15': [True, 'u'],
        'k1': [False, False], 'k2': [False, False], 'k3': [False, False], 'k4': [False, False], 'k5': [False, False], 'k6': [False, False], 'k7': [False, False], 'k8': [True, 't'], 'k9': [True, 'w'], 'k10': [True, 'a'], 'k11': [False, False], 'k12': [True, 'l'], 'k13': [True, 'e'], 'k14': [True, 'd'], 'k15': [False, False],
        'l1': [False, False], 'l2': [False, False], 'l3': [False, False], 'l4': [False, False], 'l5': [False, False], 'l6': [False, False], 'l7': [True, 'w'], 'l8': [True, 'h'], 'l9': [True, 'i'], 'l10': [True, 'n'], 'l11': [False, False], 'l12': [True, 'e'], 'l13': [True, 'k'], 'l14': [True, 'e'], 'l15': [False, False],
        'm1': [False, False], 'm2': [False, False], 'm3': [False, False], 'm4': [False, False], 'm5': [False, False], 'm6': [True, 'z'], 'm7': [True, 'a'], 'm8': [False, False], 'm9': [True, 'v'], 'm10': [False, False], 'm11': [True, 'a'], 'm12': [True, 'a'], 'm13': [False, False], 'm14': [True, 'l'], 'm15': [True, 'o'],
        'n1': [False, False], 'n2': [False, False], 'n3': [False, False], 'n4': [False, False], 'n5': [True, 'h'], 'n6': [True, 'e'], 'n7': [True, 'l'], 'n8': [True, 'v'], 'n9': [True, 'e'], 'n10': [False, False], 'n11': [True, 'n'], 'n12': [False, False], 'n13': [True, 'l'], 'n14': [False, False], 'n15': [True, 'i'],
        'o1': [False, False], 'o2': [False, False], 'o3': [False, False], 'o4': [False, False], 'o5': [False, False], 'o6': [True, 'p'], 'o7': [True, 'e'], 'o8': [False, False], 'o9': [True, 'r'], 'o10': [False, False], 'o11': [True, 'd'], 'o12': [True, 'u'], 'o13': [True, 'a'], 'o14': [True, 'd'], 'o15': [False, False]
    }
    player_hand = ['j', 'c', 'i', 't', 'e', 'y', 'n']

    print(f"Testing with hand: {player_hand}")
    move_generator = ScrabbleAllValidMoves(board_state, player_hand)

    valid_moves = move_generator.get_all_valid_moves()

    # end_time = time.time()
    # execution_time_s = end_time - start_time
    print(f"\nFound {len(valid_moves)} valid moves.")
    # print(f"Execution time: {execution_time_s:.3f} seconds")

    print("\nTop 10 Moves (by score):")
    for i, move in enumerate(valid_moves[:10]):
        print(f"  {i+1}. {move['word']} @ {move['start_space']} ({move['direction']}) - Score: {move['score']}", end="")
        if 'blank_assignments' in move:
            blanks_str = ", ".join([f"{pos}={letter}" for pos, letter in move['blank_assignments'].items()])
            print(f" (Blanks: {blanks_str})")
        else:
            print()