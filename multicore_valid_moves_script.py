import pandas as pd
from collections import Counter
from itertools import permutations
import copy
import time
import multiprocessing


class Board_and_Variables:
    special_spaces = {
        'a1': 'triple_word', 'a2': None, 'a3': None, 'a4': 'double_letter', 'a5': None, 'a6': None, 'a7': None, 'a8': 'triple_word', 'a9': None, 'a10': None, 'a11': None, 'a12': 'double_letter', 'a13': None, 'a14': None, 'a15': 'triple_word',
        'b1': None, 'b2': 'double_word', 'b3': None, 'b4': None, 'b5': None, 'b6': 'triple_letter', 'b7': None, 'b8': None, 'b9': None, 'b10': 'triple_letter', 'b11': None, 'b12': None, 'b13': None, 'b14': 'double_word', 'b15': None,
        'c1': None, 'c2': None, 'c3': 'double_word', 'c4': None, 'c5': None, 'c6': None, 'c7': 'double_letter', 'c8': None, 'c9': 'double_letter', 'c10': None, 'c11': None, 'c12': None, 'c13': 'double_word', 'c14': None, 'c15': None,
        'd1': 'double_letter', 'd2': None, 'd3': None, 'd4': 'double_word', 'd5': None, 'd6': None, 'd7': None, 'd8': 'double_letter', 'd9': None, 'd10': None, 'd11': None, 'd12': 'double_word', 'd13': None, 'd14': None, 'd15': 'double_letter',
        'e1': None, 'e2': None, 'e3': None, 'e4': None, 'e5': 'double_word', 'e6': None, 'e7': None, 'e8': None, 'e9': None, 'e10': None, 'e11': 'double_word', 'e12': None, 'e13': None, 'e14': None, 'e15': None,
        'f1': None, 'f2': 'triple_letter', 'f3': None, 'f4': None, 'f5': None, 'f6': 'triple_letter', 'f7': None, 'f8': None, 'f9': None, 'f10': 'triple_letter', 'f11': None, 'f12': None, 'f13': None, 'f14': 'triple_letter', 'f15': None,
        'g1': None, 'g2': None, 'g3': 'double_letter', 'g4': None, 'g5': None, 'g6': None, 'g7': 'double_letter', 'g8': None, 'g9': 'double_letter', 'g10': None, 'g11': None, 'g12': None, 'g13': 'double_letter', 'g14': None, 'g15': None,
        'h1': 'triple_word', 'h2': None, 'h3': None, 'h4': 'double_letter', 'h5': None, 'h6': None, 'h7': None, 'h8': 'double_word', 'h9': [True, 'i'], 'h10': [True, 'n'], 'h11': [True, 'e'], 'h12': None, 'h13': None, 'h14': None, 'h15': 'triple_word',
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
        'n': 6, 'o': 8, 'p': 2, 'q': 1, 'r': 6, 's': 4, 't': 6, 'u': 4, 'v': 2, 'w': 2, 'x': 1, 'y': 2, 'z': 1
    }

    letter_point_values = {
        'a': 1, 'b': 3, 'c': 3, 'd': 2, 'e': 1, 'f': 4, 'g': 2, 'h': 4, 'i': 1, 'j': 8, 'k': 5, 'l': 1, 'm': 3, 
        'n': 1, 'o': 1, 'p': 3, 'q': 10, 'r': 1, 's': 1, 't': 1, 'u': 1, 'v': 4, 'w': 4, 'x': 8, 'y': 4, 'z': 10
    }

class ScrabbleAllValidMoves:
    def __init__(self, board_state, player_hand):
        self.special_spaces = Board_and_Variables.special_spaces.copy()
        self.is_tile_present = board_state
        self.letter_counts = Board_and_Variables.letter_counts.copy()
        self.letter_point_values = Board_and_Variables.letter_point_values.copy()
        self.player_hand = list(player_hand.lower())


        self.words_df = pd.read_csv("word_dictionary.csv")


        self.words_set = set(str(word).lower() for word in self.words_df['legal_scrabble_words'].values)
        self.preprocess_dictionary()

    def preprocess_dictionary(self):
        self.word_prefixes = set()
        for word in self.words_set:
            for i in range(len(word)):
                self.word_prefixes.add(word[:i+1])

    # multicore processing
    def process_starting_position(self, start):
        valid_moves_local = []
        directions = ['right', 'down']
        rack_letters = self.player_hand
        rack_length = len(rack_letters)
        seen_moves_local = set()
        
        for direction in directions:
            max_length_for_line = self.get_max_line_length(start, direction)
            for word_length in range(2, min(max_length_for_line + 1, 16)):
                if word_length > rack_length + 15:
                    break
                for used_letters_count in range(1, min(word_length, rack_length) + 1):
                    combos = set(permutations(rack_letters, used_letters_count))
                    for combo in combos:
                        combo_list = list(combo)
                        row_char = start[0]
                        col_num = int(''.join(filter(str.isdigit, start)))
                        
                        if direction == 'right':
                            end_col = col_num + word_length - 1
                            if end_col > 15:
                                continue
                            spaces_word_occupies = [f"{row_char}{c}" for c in range(col_num, col_num + word_length)]
                        else:
                            end_row_ord = ord(row_char) + word_length - 1
                            if end_row_ord > ord('o'):
                                continue
                            spaces_word_occupies = [f"{chr(r)}{col_num}" for r in range(ord(row_char), ord(row_char) + word_length)]
                        
                        formed_word = []
                        temp_combo_list = combo_list[:]
                        valid_placement = True
                        for sp in spaces_word_occupies:
                            if self.is_tile_present[sp][0]:
                                formed_word.append(self.is_tile_present[sp][1])
                            else:
                                if temp_combo_list:
                                    formed_word.append(temp_combo_list.pop(0))
                                else:
                                    valid_placement = False
                                    break
                            partial_word = ''.join(formed_word)
                            if not self.could_form_valid_word(partial_word):
                                valid_placement = False
                                break
                        if not valid_placement:
                            continue
                        
                        candidate_word = ''.join(formed_word)
                        candidate_letter_count = Counter(candidate_word)
                        for sp in spaces_word_occupies:
                            if self.is_tile_present[sp][0]:
                                board_letter = self.is_tile_present[sp][1]
                                candidate_letter_count[board_letter] -= 1
                                if candidate_letter_count[board_letter] <= 0:
                                    del candidate_letter_count[board_letter]
                        hand_count = Counter(self.player_hand)
                        if any(candidate_letter_count[l] > hand_count[l] for l in candidate_letter_count):
                            continue
                        
                        is_valid, word_spaces, cwords, cwords_spaces, used_7 = self.validate_move(
                            starting_space=start,
                            word=candidate_word,
                            direction=direction,
                            is_first_turn=False
                        )
                        
                        if is_valid:
                            backup_board = copy.deepcopy(self.is_tile_present)
                            for i, sp in enumerate(word_spaces):
                                backup_board[sp][0] = True
                                backup_board[sp][1] = candidate_word[i]
                            real_board = self.is_tile_present
                            self.is_tile_present = backup_board
                            move_score = self.point_calculation(
                                main_word=candidate_word,
                                main_word_spaces=word_spaces,
                                used_all_letters=used_7,
                                attached_words=cwords,
                                attached_words_spaces=cwords_spaces
                            )
                            self.is_tile_present = real_board
                            
                            move_info = {
                                'starting_space': start,
                                'word': candidate_word,
                                'direction': direction,
                                'score': move_score,
                                'connected_words': cwords
                            }
                            move_key = (start, candidate_word, direction)
                            if move_key not in seen_moves_local:
                                seen_moves_local.add(move_key)
                                valid_moves_local.append(move_info)
        return valid_moves_local


    def could_form_valid_word(self, prefix):
        return prefix in self.word_prefixes

    def check_word(self, word: str):
        return word.lower() in self.words_set

    def find_same_space(self, dict1: dict, dict2: dict):
        for key1, value1 in dict1.items():
            for key2, value2 in dict2.items():
                if value1 == value2:
                    return value1
                

    def get_candidate_starting_positions(self, is_first_turn: bool):
        if is_first_turn:
            return ['h8']
        
        candidates = set()
        for pos, tile_info in self.is_tile_present.items():
            if tile_info[0]:
                row = pos[0]
                col = int(pos[1:])
                # for each filled square, add all empty squares within Manhattan distance 7
                for dr in range(-7, 8):
                    for dc in range(-7, 8):
                        if abs(dr) + abs(dc) > 7:
                            continue  # outside reach of a 7–letter move
                        new_row_ord = ord(row) + dr
                        if new_row_ord < ord('a') or new_row_ord > ord('o'):
                            continue  # off the board
                        new_col = col + dc
                        if new_col < 1 or new_col > 15:
                            continue
                        candidate_pos = f"{chr(new_row_ord)}{new_col}"
                        if not self.is_tile_present[candidate_pos][0]:
                            candidates.add(candidate_pos)
        return list(candidates)


    def point_calculation(self, main_word: str, main_word_spaces: list, used_all_letters: bool = False, attached_words: list = [], attached_words_spaces: list = []):
        point_total = 0
        double_word_scores = 0
        triple_word_scores = 0
        word_point_total = 0

        main_word_dict = dict(zip(main_word, main_word_spaces))

        for letter, space in main_word_dict.items():
            if self.special_spaces[space] == 'double_word':
                double_word_scores += 1
                word_point_total += self.letter_point_values[letter]
            elif self.special_spaces[space] == 'triple_word':
                triple_word_scores += 1
                word_point_total += self.letter_point_values[letter]
            elif self.special_spaces[space] == 'double_letter':
                word_point_total += self.letter_point_values[letter] * 2
            elif self.special_spaces[space] == 'triple_letter':
                word_point_total += self.letter_point_values[letter] * 3
            else:
                word_point_total += self.letter_point_values[letter]

        if double_word_scores:
            point_total += word_point_total * (2 ** double_word_scores)
        elif triple_word_scores:
            point_total += word_point_total * (3 ** triple_word_scores)
        else:
            point_total += word_point_total

        if used_all_letters:
            point_total += 50

        all_dictionaries = []
        if attached_words:
            for index, word in enumerate(attached_words):
                word_dict = {}
                for letter_index, letter in enumerate(word):
                    word_dict[letter] = attached_words_spaces[index][letter_index]
                all_dictionaries.append(word_dict)

            for dictionary in all_dictionaries:
                double_word_scores = 0
                triple_word_scores = 0
                word_point_total = 0
                shared_space = self.find_same_space(dictionary, main_word_dict)

                for letter, space in dictionary.items():
                    if letter == shared_space:
                        if self.special_spaces[space] == 'double_word':
                            double_word_scores += 1
                            word_point_total += self.letter_point_values[letter]
                        elif self.special_spaces[space] == 'triple_word':
                            triple_word_scores += 1
                            word_point_total += self.letter_point_values[letter]
                        elif self.special_spaces[space] == 'double_letter':
                            word_point_total += self.letter_point_values[letter] * 2
                        elif self.special_spaces[space] == 'triple_letter':
                            word_point_total += self.letter_point_values[letter] * 3
                        else:
                            word_point_total += self.letter_point_values[letter]
                    else:
                        word_point_total += self.letter_point_values[letter]

                if double_word_scores:
                    point_total += word_point_total * 2
                elif triple_word_scores:
                    point_total += word_point_total * 3
                else:
                    point_total += word_point_total

        return point_total

    def get_max_line_length(self, start_space: str, direction: str) -> int:
        row_char = start_space[0]
        col_num = int(''.join(filter(str.isdigit, start_space)))
        length = 1

        if direction == 'right':
            c = col_num + 1
            while c <= 15:
                pos = f"{row_char}{c}"
                if pos in self.is_tile_present:
                    length += 1
                    c += 1
                else:
                    break
        else:
            r_ord = ord(row_char) + 1
            while r_ord <= ord('o'):
                pos = f"{chr(r_ord)}{col_num}"
                if pos in self.is_tile_present:
                    length += 1
                    r_ord += 1
                else:
                    break

        return length

    def is_adjacent_to_filled(self, pos: str) -> bool:
        row_char = pos[0]
        col_num = int(''.join(filter(str.isdigit, pos)))
        adjacent_positions = []

        if ord(row_char) > ord('a'):
            adjacent_positions.append(f"{chr(ord(row_char)-1)}{col_num}")
        if ord(row_char) < ord('o'):
            adjacent_positions.append(f"{chr(ord(row_char)+1)}{col_num}")
        if col_num > 1:
            adjacent_positions.append(f"{row_char}{col_num-1}")
        if col_num < 15:
            adjacent_positions.append(f"{row_char}{col_num+1}")

        for a_pos in adjacent_positions:
            if self.is_tile_present[a_pos][0]:
                return True
        return False

    def validate_move(self, starting_space: str, word: str, direction: str, is_first_turn: bool = False):
        hand_counter = Counter(self.player_hand)
        word_list = list(word)
        word_counter = Counter(word_list)

        is_valid_space = (starting_space in self.is_tile_present)
        is_valid_word = self.check_word(word)
        is_valid_direction = direction in ('right', 'down')

        if not (is_valid_space and is_valid_word and is_valid_direction):
            return False, [], [], [], False

        word_length = len(word)
        spaces_word_occupies = [starting_space]
        number = int(starting_space[1:])
        letter_char = starting_space[0]

        if direction == 'right':
            for i in range(word_length - 1):
                new_space = f"{letter_char}{number + i + 1}"
                spaces_word_occupies.append(new_space)
        elif direction == 'down':
            letter_ord = ord(letter_char)
            for i in range(word_length - 1):
                new_space = f"{chr(letter_ord + i + 1)}{number}"
                spaces_word_occupies.append(new_space)

        if not all(space in self.is_tile_present for space in spaces_word_occupies):
            return False, [], [], [], False

        if is_first_turn and 'h8' not in spaces_word_occupies:
            return False, [], [], [], False

        complete_word = list(word)
        complete_word_spaces = spaces_word_occupies.copy()

        if direction == 'right':
            temp_col = number - 1
            while temp_col >= 1:
                temp_space = f"{letter_char}{temp_col}"
                if self.is_tile_present[temp_space][0]:
                    complete_word.insert(0, self.is_tile_present[temp_space][1])
                    complete_word_spaces.insert(0, temp_space)
                    temp_col -= 1
                else:
                    break

            temp_col = number + word_length
            while temp_col <= 15:
                temp_space = f"{letter_char}{temp_col}"
                if self.is_tile_present[temp_space][0]:
                    complete_word.append(self.is_tile_present[temp_space][1])
                    complete_word_spaces.append(temp_space)
                    temp_col += 1
                else:
                    break
        else:
            temp_row_ord = ord(letter_char) - 1
            while temp_row_ord >= ord('a'):
                temp_space = f"{chr(temp_row_ord)}{number}"
                if self.is_tile_present[temp_space][0]:
                    complete_word.insert(0, self.is_tile_present[temp_space][1])
                    complete_word_spaces.insert(0, temp_space)
                    temp_row_ord -= 1
                else:
                    break

            temp_row_ord = ord(letter_char) + word_length
            while temp_row_ord <= ord('o'):
                temp_space = f"{chr(temp_row_ord)}{number}"
                if self.is_tile_present[temp_space][0]:
                    complete_word.append(self.is_tile_present[temp_space][1])
                    complete_word_spaces.append(temp_space)
                    temp_row_ord += 1
                else:
                    break

        complete_word_str = ''.join(complete_word)
        if complete_word_str != word and not self.check_word(complete_word_str):
            return False, [], [], [], False

        word_is_adjacent_to_existing_tile = False
        pass_through_existing_tile = False
        word_list_temp = word_list.copy()

        are_connected_words_valid = []
        connected_words = []
        all_connected_word_spaces = []

        for index, space in enumerate(spaces_word_occupies):
            if self.is_tile_present[space][0]:
                if self.is_tile_present[space][1] == word_list[index]:
                    pass_through_existing_tile = True
                    if word_list[index] in word_list_temp:
                        word_list_temp.remove(word_list[index])
                else:
                    return False, [], [], [], False
            else:
                letter_temp = ord(space[0])
                number_temp = int(space[1:])

                connected_word = []
                connected_word_spaces = []

                if direction == 'right':
                    up_letter = letter_temp - 1
                    down_letter = letter_temp + 1

                    if (up_letter >= ord('a') and self.is_tile_present[f"{chr(up_letter)}{number_temp}"][0]) or \
                       (down_letter <= ord('o') and self.is_tile_present[f"{chr(down_letter)}{number_temp}"][0]):
                        word_is_adjacent_to_existing_tile = True
                        connected_word = [word_list[index]]
                        connected_word_spaces = [space]

                        temp_up = up_letter
                        while temp_up >= ord('a'):
                            up_space = f"{chr(temp_up)}{number_temp}"
                            if self.is_tile_present[up_space][0]:
                                connected_word.insert(0, self.is_tile_present[up_space][1])
                                connected_word_spaces.insert(0, up_space)
                                temp_up -= 1
                            else:
                                break

                        temp_down = down_letter
                        while temp_down <= ord('o'):
                            down_space = f"{chr(temp_down)}{number_temp}"
                            if self.is_tile_present[down_space][0]:
                                connected_word.append(self.is_tile_present[down_space][1])
                                connected_word_spaces.append(down_space)
                                temp_down += 1
                            else:
                                break

                        if len(connected_word) > 1:
                            connected_words.append(''.join(connected_word))
                            all_connected_word_spaces.append(connected_word_spaces)
                else:
                    left_num = number_temp - 1
                    right_num = number_temp + 1

                    if (left_num >= 1 and self.is_tile_present[f"{chr(letter_temp)}{left_num}"][0]) or \
                       (right_num <= 15 and self.is_tile_present[f"{chr(letter_temp)}{right_num}"][0]):
                        word_is_adjacent_to_existing_tile = True
                        connected_word = [word_list[index]]
                        connected_word_spaces = [space]

                        temp_left = left_num
                        while temp_left >= 1:
                            left_space = f"{chr(letter_temp)}{temp_left}"
                            if self.is_tile_present[left_space][0]:
                                connected_word.insert(0, self.is_tile_present[left_space][1])
                                connected_word_spaces.insert(0, left_space)
                                temp_left -= 1
                            else:
                                break

                        temp_right = right_num
                        while temp_right <= 15:
                            right_space = f"{chr(letter_temp)}{temp_right}"
                            if self.is_tile_present[right_space][0]:
                                connected_word.append(self.is_tile_present[right_space][1])
                                connected_word_spaces.append(right_space)
                                temp_right += 1
                            else:
                                break

                        if len(connected_word) > 1:
                            connected_words.append(''.join(connected_word))
                            all_connected_word_spaces.append(connected_word_spaces)

        if not is_first_turn and not (word_is_adjacent_to_existing_tile or pass_through_existing_tile):
            return False, [], [], [], False

        for w in connected_words:
            if not self.check_word(w):
                return False, [], [], [], False

        if pass_through_existing_tile:
            word_counter_temp = Counter(word_list_temp)
            letters_are_in_hand = all(word_counter_temp[l] <= hand_counter[l] for l in word_counter_temp)
        else:
            letters_are_in_hand = all(word_counter[l] <= hand_counter[l] for l in word_counter)

        if not letters_are_in_hand:
            return False, [], [], [], False

        used_7_letters = False
        if pass_through_existing_tile:
            if len(word_list_temp) == 7:
                used_7_letters = True
        else:
            if len(word_list) == 7:
                used_7_letters = True

        return True, spaces_word_occupies, connected_words, all_connected_word_spaces, used_7_letters

    def get_all_valid_moves(self, is_first_turn: bool = False):
        valid_moves = []
        anchor_squares = []
        
        starting_positions = self.get_candidate_starting_positions(is_first_turn)
        if not is_first_turn and not starting_positions:
            return valid_moves

        # multicore processing
        with multiprocessing.Pool() as pool:
            args = [(self, s) for s in starting_positions]
            results = pool.map(process_starting_position_wrapper, args)


        valid_moves = []
        for sublist in results:
            valid_moves.extend(sublist)

        valid_moves.sort(key=lambda x: x['score'], reverse=True)
        return valid_moves

# multicore processing
def process_starting_position_wrapper(args):
    instance, start = args
    return instance.process_starting_position(start)

if __name__ == "__main__":
    from scrabble_display import display_loop
    start_time = time.time() # for timing

    # blank board state
    # board_state = {
    #     'a1': [False, False], 'a2': [False, False], 'a3': [False, False], 'a4': [False, False], 'a5': [False, False], 'a6': [False, False], 'a7': [False, False], 'a8': [False, False], 'a9': [False, False], 'a10': [False, False], 'a11': [False, False], 'a12': [False, False], 'a13': [False, False], 'a14': [False, False], 'a15': [False, False],
    #     'b1': [False, False], 'b2': [False, False], 'b3': [False, False], 'b4': [False, False], 'b5': [False, False], 'b6': [False, False], 'b7': [False, False], 'b8': [False, False], 'b9': [False, False], 'b10': [False, False], 'b11': [False, False], 'b12': [False, False], 'b13': [False, False], 'b14': [False, False], 'b15': [False, False],
    #     'c1': [False, False], 'c2': [False, False], 'c3': [False, False], 'c4': [False, False], 'c5': [False, False], 'c6': [False, False], 'c7': [False, False], 'c8': [False, False], 'c9': [False, False], 'c10': [False, False], 'c11': [False, False], 'c12': [False, False], 'c13': [False, False], 'c14': [False, False], 'c15': [False, False],
    #     'd1': [False, False], 'd2': [False, False], 'd3': [False, False], 'd4': [False, False], 'd5': [False, False], 'd6': [False, False], 'd7': [False, False], 'd8': [False, False], 'd9': [False, False], 'd10': [False, False], 'd11': [False, False], 'd12': [False, False], 'd13': [False, False], 'd14': [False, False], 'd15': [False, False],
    #     'e1': [False, False], 'e2': [False, False], 'e3': [False, False], 'e4': [False, False], 'e5': [False, False], 'e6': [False, False], 'e7': [False, False], 'e8': [False, False], 'e9': [False, False], 'e10': [False, False], 'e11': [False, False], 'e12': [False, False], 'e13': [False, False], 'e14': [False, False], 'e15': [False, False],
    #     'f1': [False, False], 'f2': [False, False], 'f3': [False, False], 'f4': [False, False], 'f5': [False, False], 'f6': [False, False], 'f7': [False, False], 'f8': [False, False], 'f9': [False, False], 'f10': [False, False], 'f11': [False, False], 'f12': [False, False], 'f13': [False, False], 'f14': [False, False], 'f15': [False, False],
    #     'g1': [False, False], 'g2': [False, False], 'g3': [False, False], 'g4': [False, False], 'g5': [False, False], 'g6': [False, False], 'g7': [False, False], 'g8': [False, False], 'g9': [False, False], 'g10': [False, False], 'g11': [False, False], 'g12': [False, False], 'g13': [False, False], 'g14': [False, False], 'g15': [False, False],
    #     'h1': [False, False], 'h2': [False, False], 'h3': [False, False], 'h4': [False, False], 'h5': [False, False], 'h6': [False, False], 'h7': [False, False], 'h8': [False, False], 'h9': [False, False], 'h10': [False, False], 'h11': [False, False], 'h12': [False, False], 'h13': [False, False], 'h14': [False, False], 'h15': [False, False],
    #     'i1': [False, False], 'i2': [False, False], 'i3': [False, False], 'i4': [False, False], 'i5': [False, False], 'i6': [False, False], 'i7': [False, False], 'i8': [False, False], 'i9': [False, False], 'i10': [False, False], 'i11': [False, False], 'i12': [False, False], 'i13': [False, False], 'i14': [False, False], 'i15': [False, False],
    #     'j1': [False, False], 'j2': [False, False], 'j3': [False, False], 'j4': [False, False], 'j5': [False, False], 'j6': [False, False], 'j7': [False, False], 'j8': [False, False], 'j9': [False, False], 'j10': [False, False], 'j11': [False, False], 'j12': [False, False], 'j13': [False, False], 'j14': [False, False], 'j15': [False, False],
    #     'k1': [False, False], 'k2': [False, False], 'k3': [False, False], 'k4': [False, False], 'k5': [False, False], 'k6': [False, False], 'k7': [False, False], 'k8': [False, False], 'k9': [False, False], 'k10': [False, False], 'k11': [False, False], 'k12': [False, False], 'k13': [False, False], 'k14': [False, False], 'k15': [False, False],
    #     'l1': [False, False], 'l2': [False, False], 'l3': [False, False], 'l4': [False, False], 'l5': [False, False], 'l6': [False, False], 'l7': [False, False], 'l8': [False, False], 'l9': [False, False], 'l10': [False, False], 'l11': [False, False], 'l12': [False, False], 'l13': [False, False], 'l14': [False, False], 'l15': [False, False],
    #     'm1': [False, False], 'm2': [False, False], 'm3': [False, False], 'm4': [False, False], 'm5': [False, False], 'm6': [False, False], 'm7': [False, False], 'm8': [False, False], 'm9': [False, False], 'm10': [False, False], 'm11': [False, False], 'm12': [False, False], 'm13': [False, False], 'm14': [False, False], 'm15': [False, False],
    #     'n1': [False, False], 'n2': [False, False], 'n3': [False, False], 'n4': [False, False], 'n5': [False, False], 'n6': [False, False], 'n7': [False, False], 'n8': [False, False], 'n9': [False, False], 'n10': [False, False], 'n11': [False, False], 'n12': [False, False], 'n13': [False, False], 'n14': [False, False], 'n15': [False, False],
    #     'o1': [False, False], 'o2': [False, False], 'o3': [False, False], 'o4': [False, False], 'o5': [False, False], 'o6': [False, False], 'o7': [False, False], 'o8': [False, False], 'o9': [False, False], 'o10': [False, False], 'o11': [False, False], 'o12': [False, False], 'o13': [False, False], 'o14': [False, False], 'o15': [False, False]
    # }

    # board_state = {
    #     'a1': [False, False], 'a2': [False, False], 'a3': [False, False], 'a4': [False, False], 'a5': [False, False], 'a6': [False, False], 'a7': [False, False], 'a8': [False, False], 'a9': [False, False], 'a10': [False, False], 'a11': [False, False], 'a12': [False, False], 'a13': [False, False], 'a14': [False, False], 'a15': [False, False],
    #     'b1': [False, False], 'b2': [False, False], 'b3': [False, False], 'b4': [False, False], 'b5': [False, False], 'b6': [False, False], 'b7': [False, False], 'b8': [False, False], 'b9': [False, False], 'b10': [False, False], 'b11': [False, False], 'b12': [False, False], 'b13': [False, False], 'b14': [False, False], 'b15': [False, False],
    #     'c1': [False, False], 'c2': [False, False], 'c3': [False, False], 'c4': [False, False], 'c5': [False, False], 'c6': [False, False], 'c7': [False, False], 'c8': [False, False], 'c9': [False, False], 'c10': [False, False], 'c11': [False, False], 'c12': [False, False], 'c13': [False, False], 'c14': [False, False], 'c15': [False, False],
    #     'd1': [False, False], 'd2': [False, False], 'd3': [False, False], 'd4': [False, False], 'd5': [False, False], 'd6': [False, False], 'd7': [False, False], 'd8': [False, False], 'd9': [False, False], 'd10': [False, False], 'd11': [False, False], 'd12': [False, False], 'd13': [False, False], 'd14': [False, False], 'd15': [False, False],
    #     'e1': [False, False], 'e2': [False, False], 'e3': [False, False], 'e4': [False, False], 'e5': [False, False], 'e6': [False, False], 'e7': [False, False], 'e8': [False, False], 'e9': [False, False], 'e10': [False, False], 'e11': [False, False], 'e12': [False, False], 'e13': [False, False], 'e14': [False, False], 'e15': [False, False],
    #     'f1': [False, False], 'f2': [False, False], 'f3': [False, False], 'f4': [False, False], 'f5': [False, False], 'f6': [False, False], 'f7': [False, False], 'f8': [False, False], 'f9': [False, False], 'f10': [False, False], 'f11': [False, False], 'f12': [False, False], 'f13': [False, False], 'f14': [False, False], 'f15': [False, False],
    #     'g1': [False, False], 'g2': [False, False], 'g3': [False, False], 'g4': [False, False], 'g5': [False, False], 'g6': [False, False], 'g7': [False, False], 'g8': [False, False], 'g9': [False, False], 'g10': [False, False], 'g11': [False, False], 'g12': [False, False], 'g13': [False, False], 'g14': [False, False], 'g15': [False, False],
    #     'h1': [False, False], 'h2': [False, False], 'h3': [False, False], 'h4': [False, False], 'h5': [False, False], 'h6': [False, False], 'h7': [False, False], 'h8': [True, 'l'], 'h9': [True, 'i'], 'h10': [True, 'n'], 'h11': [True, 'e'], 'h12': [False, False], 'h13': [False, False], 'h14': [False, False], 'h15': [False, False],
    #     'i1': [False, False], 'i2': [False, False], 'i3': [False, False], 'i4': [False, False], 'i5': [False, False], 'i6': [False, False], 'i7': [False, False], 'i8': [False, False], 'i9': [False, False], 'i10': [True, 'e'], 'i11': [True, 'd'], 'i12': [False, False], 'i13': [False, False], 'i14': [False, False], 'i15': [False, False],
    #     'j1': [False, False], 'j2': [False, False], 'j3': [False, False], 'j4': [False, False], 'j5': [False, False], 'j6': [False, False], 'j7': [False, False], 'j8': [False, False], 'j9': [False, False], 'j10': [False, False], 'j11': [True, 's'], 'j12': [True, 'a'], 'j13': [True, 'w'], 'j14': [False, False], 'j15': [False, False],
    #     'k1': [False, False], 'k2': [False, False], 'k3': [False, False], 'k4': [False, False], 'k5': [False, False], 'k6': [False, False], 'k7': [False, False], 'k8': [False, False], 'k9': [False, False], 'k10': [False, False], 'k11': [False, False], 'k12': [False, False], 'k13': [False, False], 'k14': [False, False], 'k15': [False, False],
    #     'l1': [False, False], 'l2': [False, False], 'l3': [False, False], 'l4': [False, False], 'l5': [False, False], 'l6': [False, False], 'l7': [False, False], 'l8': [False, False], 'l9': [False, False], 'l10': [False, False], 'l11': [False, False], 'l12': [False, False], 'l13': [False, False], 'l14': [False, False], 'l15': [False, False],
    #     'm1': [False, False], 'm2': [False, False], 'm3': [False, False], 'm4': [False, False], 'm5': [False, False], 'm6': [False, False], 'm7': [False, False], 'm8': [False, False], 'm9': [False, False], 'm10': [False, False], 'm11': [False, False], 'm12': [False, False], 'm13': [False, False], 'm14': [False, False], 'm15': [False, False],
    #     'n1': [False, False], 'n2': [False, False], 'n3': [False, False], 'n4': [False, False], 'n5': [False, False], 'n6': [False, False], 'n7': [False, False], 'n8': [False, False], 'n9': [False, False], 'n10': [False, False], 'n11': [False, False], 'n12': [False, False], 'n13': [False, False], 'n14': [False, False], 'n15': [False, False],
    #     'o1': [False, False], 'o2': [False, False], 'o3': [False, False], 'o4': [False, False], 'o5': [False, False], 'o6': [False, False], 'o7': [False, False], 'o8': [False, False], 'o9': [False, False], 'o10': [False, False], 'o11': [False, False], 'o12': [False, False], 'o13': [False, False], 'o14': [False, False], 'o15': [False, False]
    # }

    # edit to test
    board_state = {
        'a1': [False, False], 'a2': [False, False], 'a3': [False, False], 'a4': [False, False], 'a5': [False, False], 'a6': [False, False], 'a7': [False, False], 'a8': [False, False], 'a9': [False, False], 'a10': [False, False], 'a11': [False, False], 'a12': [False, False], 'a13': [False, False], 'a14': [False, False], 'a15': [False, False],
        'b1': [False, False], 'b2': [False, False], 'b3': [False, False], 'b4': [False, False], 'b5': [False, False], 'b6': [False, False], 'b7': [False, False], 'b8': [False, False], 'b9': [False, False], 'b10': [False, False], 'b11': [False, False], 'b12': [False, False], 'b13': [False, False], 'b14': [False, False], 'b15': [False, False],
        'c1': [False, False], 'c2': [False, False], 'c3': [False, False], 'c4': [False, False], 'c5': [False, False], 'c6': [False, False], 'c7': [False, False], 'c8': [False, False], 'c9': [False, False], 'c10': [False, False], 'c11': [False, False], 'c12': [False, False], 'c13': [True, 'o'], 'c14': [True, 'l'], 'c15': [True, 'e'],
        'd1': [False, False], 'd2': [False, False], 'd3': [False, False], 'd4': [False, False], 'd5': [False, False], 'd6': [False, False], 'd7': [False, False], 'd8': [False, False], 'd9': [True, 'r'], 'd10': [True, 'i'], 'd11': [True, 'v'], 'd12': [True, 'e'], 'd13': [True, 'n'], 'd14': [False, False], 'd15': [False, False],
        'e1': [False, False], 'e2': [False, False], 'e3': [False, False], 'e4': [False, False], 'e5': [False, False], 'e6': [False, False], 'e7': [False, False], 'e8': [False, False], 'e9': [False, False], 'e10': [False, False], 'e11': [True, 'i'], 'e12': [False, False], 'e13': [False, False], 'e14': [False, False], 'e15': [False, False],
        'f1': [False, False], 'f2': [False, False], 'f3': [True, 'b'], 'f4': [False, False], 'f5': [False, False], 'f6': [False, False], 'f7': [False, False], 'f8': [False, False], 'f9': [False, False], 'f10': [False, False], 'f11': [True, 'x'], 'f12': [False, False], 'f13': [False, False], 'f14': [False, False], 'f15': [False, False],
        'g1': [False, False], 'g2': [False, False], 'g3': [True, 'o'], 'g4': [False, False], 'g5': [True, 'p'], 'g6': [False, False], 'g7': [True, 'f'], 'g8': [True, 'o'], 'g9': [True, 'r'], 'g10': [True, 't'], 'g11': [True, 'e'], 'g12': [True, 'd'], 'g13': [False, False], 'g14': [False, False], 'g15': [False, False],
        'h1': [True, 'w'], 'h2': [True, 'a'], 'h3': [True, 'r'], 'h4': [True, 'z'], 'h5': [True, 'o'], 'h6': [True, 'n'], 'h7': [True, 'e'], 'h8': [True, 's'], 'h9': [False, False], 'h10': [False, False], 'h11': [True, 'n'], 'h12': [True, 'e'], 'h13': [False, False], 'h14': [False, False], 'h15': [False, False],
        'i1': [False, False], 'i2': [False, False], 'i3': [True, 'e'], 'i4': [False, False], 'i5': [True, 'g'], 'i6': [False, False], 'i7': [True, 'w'], 'i8': [True, 'e'], 'i9': [True, 'b'], 'i10': [False, False], 'i11': [False, False], 'i12': [True, 's'], 'i13': [False, False], 'i14': [False, False], 'i15': [False, False],
        'j1': [False, False], 'j2': [False, False], 'j3': [True, 'a'], 'j4': [False, False], 'j5': [True, 'g'], 'j6': [False, False], 'j7': [False, False], 'j8': [False, False], 'j9': [False, False], 'j10': [True, 'k'], 'j11': [True, 'u'], 'j12': [True, 't'], 'j13': [True, 'c'], 'j14': [True, 'h'], 'j15': [False, False],
        'k1': [False, False], 'k2': [False, False], 'k3': [True, 'l'], 'k4': [False, False], 'k5': [True, 'e'], 'k6': [False, False], 'k7': [False, False], 'k8': [False, False], 'k9': [False, False], 'k10': [False, False], 'k11': [False, False], 'k12': [True, 'a'], 'k13': [False, False], 'k14': [False, False], 'k15': [False, False],
        'l1': [False, False], 'l2': [False, False], 'l3': [True, 'i'], 'l4': [False, False], 'l5': [False, False], 'l6': [False, False], 'l7': [False, False], 'l8': [True, 'y'], 'l9': [False, False], 'l10': [False, False], 'l11': [False, False], 'l12': [True, 'i'], 'l13': [False, False], 'l14': [False, False], 'l15': [False, False],
        'm1': [False, False], 'm2': [False, False], 'm3': [True, 's'], 'm4': [False, False], 'm5': [False, False], 'm6': [True, 'n'], 'm7': [True, 'a'], 'm8': [True, 'u'], 'm9': [True, 's'], 'm10': [True, 'e'], 'm11': [True, 'a'], 'm12': [True, 'n'], 'm13': [True, 't'], 'm14': [False, False], 'm15': [False, False],
        'n1': [False, False], 'n2': [False, False], 'n3': [False, False], 'n4': [False, False], 'n5': [False, False], 'n6': [False, False], 'n7': [False, False], 'n8': [True, 'f'], 'n9': [False, False], 'n10': [False, False], 'n11': [False, False], 'n12': [False, False], 'n13': [False, False], 'n14': [False, False], 'n15': [False, False],
        'o1': [False, False], 'o2': [False, False], 'o3': [False, False], 'o4': [False, False], 'o5': [False, False], 'o6': [False, False], 'o7': [False, False], 'o8': [True, 't'], 'o9': [False, False], 'o10': [False, False], 'o11': [False, False], 'o12': [False, False], 'o13': [False, False], 'o14': [False, False], 'o15': [False, False],
    }

    # edit to test
    player_hand = 'zsattli'

    allValidMoves = ScrabbleAllValidMoves(board_state, player_hand)

    validwords = allValidMoves.get_all_valid_moves(False)
    print(validwords)

    end_time = time.time()
    execution_time_ms = (end_time - start_time) * 1000
    print(f"\nExecution time: {execution_time_ms:.2f} milliseconds")

    if validwords:
        best_move = validwords[0]
        
        display_board = copy.deepcopy(allValidMoves.is_tile_present)
        
        word = best_move['word']
        start_space = best_move['starting_space']
        direction = best_move['direction']
        
        row = start_space[0]
        col = int(start_space[1:])
        
        for i, letter in enumerate(word):
            if direction == 'right':
                pos = f"{row}{col + i}"
            else:  # down
                pos = f"{chr(ord(row) + i)}{col}"
            display_board[pos] = [True, letter]

        while True:
            display_loop(
                display_board,
                allValidMoves.player_hand,
                [],  # empty second hand
                best_move['score'],
                0 # 2nd players score
            )