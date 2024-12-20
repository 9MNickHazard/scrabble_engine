import pandas as pd
import time
import random
from collections import Counter
from scrabble_display import display_loop

def sep():
    print("=========----------=========")

# Scrabble Game

class Board_and_Variables:
    board = (
    ('a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8', 'a9', 'a10', 'a11', 'a12', 'a13', 'a14', 'a15'),
    ('b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8', 'b9', 'b10', 'b11', 'b12', 'b13', 'b14', 'b15'),
    ('c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9', 'c10', 'c11', 'c12', 'c13', 'c14', 'c15'),
    ('d1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7', 'd8', 'd9', 'd10', 'd11', 'd12', 'd13', 'd14', 'd15'),
    ('e1', 'e2', 'e3', 'e4', 'e5', 'e6', 'e7', 'e8', 'e9', 'e10', 'e11', 'e12', 'e13', 'e14', 'e15'),
    ('f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11', 'f12', 'f13', 'f14', 'f15'),
    ('g1', 'g2', 'g3', 'g4', 'g5', 'g6', 'g7', 'g8', 'g9', 'g10', 'g11', 'g12', 'g13', 'g14', 'g15'),
    ('h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'h7', 'h8', 'h9', 'h10', 'h11', 'h12', 'h13', 'h14', 'h15'),
    ('i1', 'i2', 'i3', 'i4', 'i5', 'i6', 'i7', 'i8', 'i9', 'i10', 'i11', 'i12', 'i13', 'i14', 'i15'),
    ('j1', 'j2', 'j3', 'j4', 'j5', 'j6', 'j7', 'j8', 'j9', 'j10', 'j11', 'j12', 'j13', 'j14', 'j15'),
    ('k1', 'k2', 'k3', 'k4', 'k5', 'k6', 'k7', 'k8', 'k9', 'k10', 'k11', 'k12', 'k13', 'k14', 'k15'),
    ('l1', 'l2', 'l3', 'l4', 'l5', 'l6', 'l7', 'l8', 'l9', 'l10', 'l11', 'l12', 'l13', 'l14', 'l15'),
    ('m1', 'm2', 'm3', 'm4', 'm5', 'm6', 'm7', 'm8', 'm9', 'm10', 'm11', 'm12', 'm13', 'm14', 'm15'),
    ('n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'n7', 'n8', 'n9', 'n10', 'n11', 'n12', 'n13', 'n14', 'n15'),
    ('o1', 'o2', 'o3', 'o4', 'o5', 'o6', 'o7', 'o8', 'o9', 'o10', 'o11', 'o12', 'o13', 'o14', 'o15')
    )


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

    # [is there a tile here, False or which tile, e.g. 'g']
    is_tile_present = {
        'a1': [False, False], 'a2': [False, False], 'a3': [False, False], 'a4': [False, False], 'a5': [False, False], 'a6': [False, False], 'a7': [False, False], 'a8': [False, False], 'a9': [False, False], 'a10': [False, False], 'a11': [False, False], 'a12': [False, False], 'a13': [False, False], 'a14': [False, False], 'a15': [False, False],
        'b1': [False, False], 'b2': [False, False], 'b3': [False, False], 'b4': [False, False], 'b5': [False, False], 'b6': [False, False], 'b7': [False, False], 'b8': [False, False], 'b9': [False, False], 'b10': [False, False], 'b11': [False, False], 'b12': [False, False], 'b13': [False, False], 'b14': [False, False], 'b15': [False, False],
        'c1': [False, False], 'c2': [False, False], 'c3': [False, False], 'c4': [False, False], 'c5': [False, False], 'c6': [False, False], 'c7': [False, False], 'c8': [False, False], 'c9': [False, False], 'c10': [False, False], 'c11': [False, False], 'c12': [False, False], 'c13': [False, False], 'c14': [False, False], 'c15': [False, False],
        'd1': [False, False], 'd2': [False, False], 'd3': [False, False], 'd4': [False, False], 'd5': [False, False], 'd6': [False, False], 'd7': [False, False], 'd8': [False, False], 'd9': [False, False], 'd10': [False, False], 'd11': [False, False], 'd12': [False, False], 'd13': [False, False], 'd14': [False, False], 'd15': [False, False],
        'e1': [False, False], 'e2': [False, False], 'e3': [False, False], 'e4': [False, False], 'e5': [False, False], 'e6': [False, False], 'e7': [False, False], 'e8': [False, False], 'e9': [False, False], 'e10': [False, False], 'e11': [False, False], 'e12': [False, False], 'e13': [False, False], 'e14': [False, False], 'e15': [False, False],
        'f1': [False, False], 'f2': [False, False], 'f3': [False, False], 'f4': [False, False], 'f5': [False, False], 'f6': [False, False], 'f7': [False, False], 'f8': [False, False], 'f9': [False, False], 'f10': [False, False], 'f11': [False, False], 'f12': [False, False], 'f13': [False, False], 'f14': [False, False], 'f15': [False, False],
        'g1': [False, False], 'g2': [False, False], 'g3': [False, False], 'g4': [False, False], 'g5': [False, False], 'g6': [False, False], 'g7': [False, False], 'g8': [False, False], 'g9': [False, False], 'g10': [False, False], 'g11': [False, False], 'g12': [False, False], 'g13': [False, False], 'g14': [False, False], 'g15': [False, False],
        'h1': [False, False], 'h2': [False, False], 'h3': [False, False], 'h4': [False, False], 'h5': [False, False], 'h6': [False, False], 'h7': [False, False], 'h8': [False, False], 'h9': [False, False], 'h10': [False, False], 'h11': [False, False], 'h12': [False, False], 'h13': [False, False], 'h14': [False, False], 'h15': [False, False],
        'i1': [False, False], 'i2': [False, False], 'i3': [False, False], 'i4': [False, False], 'i5': [False, False], 'i6': [False, False], 'i7': [False, False], 'i8': [False, False], 'i9': [False, False], 'i10': [False, False], 'i11': [False, False], 'i12': [False, False], 'i13': [False, False], 'i14': [False, False], 'i15': [False, False],
        'j1': [False, False], 'j2': [False, False], 'j3': [False, False], 'j4': [False, False], 'j5': [False, False], 'j6': [False, False], 'j7': [False, False], 'j8': [False, False], 'j9': [False, False], 'j10': [False, False], 'j11': [False, False], 'j12': [False, False], 'j13': [False, False], 'j14': [False, False], 'j15': [False, False],
        'k1': [False, False], 'k2': [False, False], 'k3': [False, False], 'k4': [False, False], 'k5': [False, False], 'k6': [False, False], 'k7': [False, False], 'k8': [False, False], 'k9': [False, False], 'k10': [False, False], 'k11': [False, False], 'k12': [False, False], 'k13': [False, False], 'k14': [False, False], 'k15': [False, False],
        'l1': [False, False], 'l2': [False, False], 'l3': [False, False], 'l4': [False, False], 'l5': [False, False], 'l6': [False, False], 'l7': [False, False], 'l8': [False, False], 'l9': [False, False], 'l10': [False, False], 'l11': [False, False], 'l12': [False, False], 'l13': [False, False], 'l14': [False, False], 'l15': [False, False],
        'm1': [False, False], 'm2': [False, False], 'm3': [False, False], 'm4': [False, False], 'm5': [False, False], 'm6': [False, False], 'm7': [False, False], 'm8': [False, False], 'm9': [False, False], 'm10': [False, False], 'm11': [False, False], 'm12': [False, False], 'm13': [False, False], 'm14': [False, False], 'm15': [False, False],
        'n1': [False, False], 'n2': [False, False], 'n3': [False, False], 'n4': [False, False], 'n5': [False, False], 'n6': [False, False], 'n7': [False, False], 'n8': [False, False], 'n9': [False, False], 'n10': [False, False], 'n11': [False, False], 'n12': [False, False], 'n13': [False, False], 'n14': [False, False], 'n15': [False, False],
        'o1': [False, False], 'o2': [False, False], 'o3': [False, False], 'o4': [False, False], 'o5': [False, False], 'o6': [False, False], 'o7': [False, False], 'o8': [False, False], 'o9': [False, False], 'o10': [False, False], 'o11': [False, False], 'o12': [False, False], 'o13': [False, False], 'o14': [False, False], 'o15': [False, False]
    }

    letter_counts = {
        'a': 9, 'b': 2, 'c': 2, 'd': 4, 'e': 12, 'f': 2, 'g': 3, 'h': 2, 'i': 9, 'j': 1, 'k': 1, 'l': 4, 'm': 2,
        'n': 6, 'o': 8, 'p': 2, 'q': 1, 'r': 6, 's': 4, 't': 6, 'u': 4, 'v': 2, 'w': 2, 'x': 1, 'y': 2, 'z': 1
    }

    letter_point_values = {
        'a': 1, 'b': 3, 'c': 3, 'd': 2, 'e': 1, 'f': 4, 'g': 2, 'h': 4, 'i': 1,  'j': 8, 'k': 5, 'l': 1, 'm': 3, 
        'n': 1, 'o': 1, 'p': 3, 'q': 10, 'r': 1, 's': 1, 't': 1, 'u': 1, 'v': 4, 'w': 4, 'x': 8, 'y': 4, 'z': 10
    }

class Scrabble_Game:
    def __init__(self):
        self.board = Board_and_Variables.board
        self.special_spaces = Board_and_Variables.special_spaces
        self.is_tile_present = Board_and_Variables.is_tile_present
        self.letter_counts = Board_and_Variables.letter_counts
        self.letter_point_values = Board_and_Variables.letter_point_values
        self.player_1_hand = []
        self.player_2_hand = []
        self.player_1_score = 0
        self.player_2_score = 0
        self.words_df = pd.read_csv("word_dictionary.csv")

    # returns 1 or 2 for which player goes first
    def who_goes_first(self):
        print("Both players will roll a 20 sided die. The player who gets the highest roll will go first. If it is a tie, both players will roll again.")
        sep()
        # time.sleep(2)
        while True:
            while True:
                confirm = input("Press 'r' and hit enter to roll both dies: ")
                sep()
                if confirm.strip().lower() == 'r':
                    break
                else:
                    print("Please enter a valid response.")
                    sep()
                    # time.sleep(1)

            p1_roll = random.randint(1, 20)
            p2_roll = random.randint(1, 20)

            print("Rolling for P1...")
            # time.sleep(2)
            print(f"Player 1 rolled a: {p1_roll}")
            sep()
            # time.sleep(2)
            print("Rolling for P2...")
            # time.sleep(2)
            print(f"Player 2 rolled a: {p2_roll}")
            sep()
            # time.sleep(2)

            if p1_roll != p2_roll:
                if p1_roll > p2_roll:
                    return 1
                else:
                    return 2
            else:
                print("It's a tie! Time to reroll...")
                sep()
                # time.sleep(1.5)


    def draw_tiles(self, player: int) -> list:
        available_tiles = []

        for letter, count in self.letter_counts.items():
            available_tiles.extend([letter] * count)

        if player == 1:
            tiles_to_pick = 7 - len(self.player_1_hand)
        elif player == 2:
            tiles_to_pick = 7 - len(self.player_2_hand)
        else:
            print("Invalid Player")

        if tiles_to_pick > len(available_tiles):
            tiles_to_pick = len(available_tiles)
            if tiles_to_pick == 0:
                return

        selected_tiles = random.sample(available_tiles, tiles_to_pick)

        for tile in selected_tiles:
            self.letter_counts[tile] -= 1

        if player == 1:
            self.player_1_hand += selected_tiles
        elif player == 2:
            self.player_2_hand += selected_tiles
        else:
            print("Invalid Player")

    def exchange_tiles(self, player: int) -> bool:
        while True:
            letter_choice = input("Input the letters you would like to exchange (input 0 to cancel), separated by a single space.\nFor example: g z o a i\nWhich letters would you like to exchange: ")
            sep()

            letter_list = letter_choice.split()
            exchange_counter = Counter(letter_list)

            if player == 1:
                if letter_choice.strip() == '0':
                    return False
                
                hand_counter = Counter(self.player_1_hand)

                valid_exchange = all(exchange_counter[letter] <= hand_counter[letter] for letter in exchange_counter)

                if valid_exchange:
                    for tile in letter_list:
                        self.player_1_hand.remove(tile)
                    self.draw_tiles(1)
                    break
                else:
                    continue

            elif player == 2:
                if letter_choice.strip() == '0':
                    return False
                
                hand_counter = Counter(self.player_2_hand)

                valid_exchange = all(exchange_counter[letter] <= hand_counter[letter] for letter in exchange_counter)

                if valid_exchange:
                    for tile in letter_list:
                        self.player_2_hand.remove(tile)
                    self.draw_tiles(2)
                    break
                else:
                    continue

            else:
                sep()
                print("Invalid Player")

        for tile in letter_list:
            self.letter_counts[tile] += 1
        
        return True

    def check_word(self, word: str, words_df: pd.DataFrame) -> bool:
        word = word.lower()
        return word in words_df['legal_scrabble_words'].values
    
    def find_same_space(self, dict1: dict, dict2: dict):
        for key1, value1 in dict1.items():
            for key2, value2 in dict2.items():
                if value1 == value2:
                    return value1
    
    def point_calculation(self, player: int, main_word: str, main_word_spaces: list, used_all_letters: bool = False, attached_words: list = [], attached_words_spaces: list = []):
        point_total = 0
        double_word_scores = 0
        triple_word_scores = 0
        word_point_total = 0

        # main word
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

        # attached words
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


        if player == 1:
            self.player_1_score += point_total
        else:
            self.player_2_score += point_total

        



    def place_word(self, player: int, is_first_turn: bool = False) -> bool:
        used_7_letters = False

        if player == 1:
            hand_counter = Counter(self.player_1_hand)
            # letters_in_hand = len(self.player_1_hand)
        else:
            hand_counter = Counter(self.player_2_hand)
            # letters_in_hand = len(self.player_2_hand)

        while True:
            choice = input("Pick the starting tile, the word you want to play and the direction (down or right). Enter 0 to cancel.\nExample: h8 dog right\nEnter your choice: ")
            sep()

            if choice == '0':
                return False
            
            choice = choice.split()

            if len(choice) == 3:
                starting_space = choice[0]
                word = choice[1]
                direction = choice[2]

                is_valid_space = any(starting_space in row for row in self.board)
                # print(is_valid_space) # Debug print
                is_valid_word = self.check_word(word, self.words_df)
                # print(is_valid_word) # Debug print
                is_valid_direction = direction in ('right', 'down')
                # print(is_valid_direction) # Debug print

                word_list = list(word)

                word_counter = Counter(word_list)

                if is_valid_space and is_valid_word and is_valid_direction:
                    word_length = len(word)
                    spaces_word_occupies = []
                    spaces_word_occupies.append(starting_space)

                    number = int(starting_space[1:])
                    
                    if direction == 'right':
                        letter = starting_space[0]
                        for i in range(word_length - 1):
                            new_space = f"{letter}{number + i + 1}"
                            spaces_word_occupies.append(new_space)
                            # print(f"Added space: {new_space}")  # Debug print

                    elif direction == 'down':
                        letter = ord(starting_space[0])
                        for i in range(word_length - 1):
                            new_space = f"{chr(letter + i + 1)}{number}"
                            spaces_word_occupies.append(new_space)
                            # print(f"Added space: {new_space}")  # Debug print

                    # print(f"All spaces to occupy: {spaces_word_occupies}")  # Debug print
                    is_word_within_board = all(space in self.is_tile_present for space in spaces_word_occupies)
                    # print(f"Is word within board: {is_word_within_board}")  # Debug print
                    

                    is_word_within_board = all(space in self.is_tile_present for space in spaces_word_occupies)

                    if is_word_within_board:
                        if is_first_turn:
                            if 'h8' in spaces_word_occupies:
                                letters_are_in_hand = all(word_counter[letter1] <= hand_counter[letter1] for letter1 in word_counter)

                                if letters_are_in_hand:
                                    for index, space in enumerate(spaces_word_occupies):
                                        self.is_tile_present[space][0] = True
                                        self.is_tile_present[space][1] = word[index]

                                    if len(word_list) == 7:
                                        used_7_letters = True

                                    if player == 1:
                                        for tile in word_list:
                                            self.player_1_hand.remove(tile)

                                        self.draw_tiles(1)
                                    else:
                                        for tile in word_list:
                                            self.player_2_hand.remove(tile)

                                        self.draw_tiles(2)

                                    self.point_calculation(player, word, spaces_word_occupies, used_7_letters)
                                    return True

                                else:
                                    print("All letters in your chosen word must be in your hand.")
                                    sep()
                                    continue

                            else:
                                print("Your first word must go through the space 'h8'")
                                sep()
                                continue
                        else:
                            word_is_adjacent_to_existing_tile = False
                            pass_through_existing_tile = False
                            word_list_temp = word_list.copy()
                            # list of bools for all connected words if they are words or not
                            are_connected_words_valid = []
                            # list of strings of connected words
                            connected_words = []
                            all_connected_word_spaces = []

                            for index, space in enumerate(spaces_word_occupies):
                                if self.is_tile_present[space][0] == True:
                                    if self.is_tile_present[space][1] == word_list[index]:
                                        pass_through_existing_tile = True
                                        word_list_temp.remove(self.is_tile_present[space][1])
                                    else:
                                        print("Mismatch in your chosen word and existing tiles on the board.")
                                        sep()
                                        return False
                                else:
                                    letter_temp = ord(space[0])
                                    number_temp = int(space[1:])

                                    connected_word = []
                                    connected_word_spaces = []

                                    if direction == 'right':
                                        if (self.is_tile_present[f"{chr(letter_temp + 1)}{number_temp}"][0] == True) or (self.is_tile_present[f"{chr(letter_temp - 1)}{number_temp}"][0] == True):
                                            word_is_adjacent_to_existing_tile = True
                                            connected_word = [word_list[index]]  
                                            connected_word_spaces = [space]      
                                                                    
                                            # check up
                                            temp_letter = letter_temp
                                            while temp_letter > ord('a'):
                                                if self.is_tile_present[f"{chr(temp_letter - 1)}{number_temp}"][0]:
                                                    temp_word = self.is_tile_present[f"{chr(temp_letter - 1)}{number_temp}"][1]
                                                    connected_word.insert(0, temp_word)
                                                    connected_word_spaces.insert(0, f"{chr(temp_letter - 1)}{number_temp}")
                                                    temp_letter -= 1
                                                else:
                                                    break
                                                                        
                                            # check down
                                            temp_letter = letter_temp
                                            while temp_letter < ord('o'):
                                                if self.is_tile_present[f"{chr(temp_letter + 1)}{number_temp}"][0]:
                                                    temp_word = self.is_tile_present[f"{chr(temp_letter + 1)}{number_temp}"][1]
                                                    connected_word.append(temp_word)
                                                    connected_word_spaces.append(f"{chr(temp_letter + 1)}{number_temp}")
                                                    temp_letter += 1
                                                else:
                                                    break
                                                                        
                                            if len(connected_word) > 1:
                                                all_connected_word_spaces.append(connected_word_spaces)
                                                word_str = ''.join(connected_word)
                                                connected_words.append(word_str)


                                    elif direction == 'down':
                                        if (self.is_tile_present[f"{chr(letter_temp)}{number_temp + 1}"][0] == True) or (self.is_tile_present[f"{chr(letter_temp)}{number_temp - 1}"][0] == True):
                                            word_is_adjacent_to_existing_tile = True
                                            connected_word = [word_list[index]]
                                            connected_word_spaces = [space]
                                            
                                            # check left
                                            temp_number = number_temp
                                            while temp_number > 1:
                                                if self.is_tile_present[f"{chr(letter_temp)}{temp_number - 1}"][0]:
                                                    temp_word = self.is_tile_present[f"{chr(letter_temp)}{temp_number - 1}"][1]
                                                    connected_word.insert(0, temp_word)
                                                    connected_word_spaces.insert(0, f"{chr(letter_temp)}{temp_number - 1}")
                                                    temp_number -= 1
                                                else:
                                                    break
                                            
                                            # check right
                                            temp_number = number_temp
                                            while temp_number < 15:
                                                if self.is_tile_present[f"{chr(letter_temp)}{temp_number + 1}"][0]:
                                                    temp_word = self.is_tile_present[f"{chr(letter_temp)}{temp_number + 1}"][1]
                                                    connected_word.append(temp_word)
                                                    connected_word_spaces.append(f"{chr(letter_temp)}{temp_number + 1}")
                                                    temp_number += 1
                                                else:
                                                    break
                                            

                                            if len(connected_word) > 1:
                                                all_connected_word_spaces.append(connected_word_spaces)
                                                word_str = ''.join(connected_word)
                                                connected_words.append(word_str)


                            for word1 in connected_words:
                                are_connected_words_valid.append(self.check_word(word1, self.words_df))

                            all_connected_words_valid = all(are_connected_words_valid)

                            # # Debug prints
                            # print(f"Word is adjacent: {word_is_adjacent_to_existing_tile}")
                            # print(f"Connected words: {connected_words}")
                            # print(f"All connected words valid: {all_connected_words_valid}")
                            # print(f"Pass through existing: {pass_through_existing_tile}")

                            if pass_through_existing_tile:
                                word_counter_temp = Counter(word_list_temp)
                                letters_are_in_hand = all(word_counter_temp[letter] <= hand_counter[letter] for letter in word_counter_temp)
                                # print(f"Letters in hand (pass through): {letters_are_in_hand}") # Debug print
                                if letters_are_in_hand:
                                    if len(word_list_temp) == 7:
                                        used_7_letters = True

                                    if player == 1:
                                        for tile in word_list_temp:
                                            self.player_1_hand.remove(tile)

                                        self.draw_tiles(1)
                                    else:
                                        for tile in word_list_temp:
                                            self.player_2_hand.remove(tile)

                                        self.draw_tiles(2)
                            
                            elif word_is_adjacent_to_existing_tile and all_connected_words_valid and not pass_through_existing_tile:
                                letters_are_in_hand = all(word_counter[letter] <= hand_counter[letter] for letter in word_counter)
                                # print(f"Letters in hand (adjacent): {letters_are_in_hand}") # Debug print
                                if letters_are_in_hand:
                                    if len(word_list) == 7:
                                        used_7_letters = True

                                    if player == 1:
                                        for tile in word_list:
                                            self.player_1_hand.remove(tile)

                                        self.draw_tiles(1)
                                    else:
                                        for tile in word_list:
                                            self.player_2_hand.remove(tile)

                                        self.draw_tiles(2)

                            else:
                                continue

                            # # Debug prints
                            # print(f"Main word being placed: {word}")
                            # print(f"Spaces it will occupy: {spaces_word_occupies}")
                            # print(f"Letters in the main word: {word_list}")
                            # print(f"Connected words found: {connected_words}")
                            # print(f"Connected word spaces: {all_connected_word_spaces}")

                            if letters_are_in_hand:
                                for index, space in enumerate(spaces_word_occupies):
                                    self.is_tile_present[space][0] = True
                                    self.is_tile_present[space][1] = word[index]
                                # point calculation
                                self.point_calculation(player, word, spaces_word_occupies, used_7_letters, connected_words, all_connected_word_spaces)
                                return True
  
                            else:
                                continue

                    else:
                        print("Your word was too long!")
                        sep()
                        continue
                elif not is_valid_word:
                    print("That is not a valid word.")
                    sep()
                    continue
                elif not is_valid_direction:
                    print("Please choose right or down.")
                    sep()
                    continue
                elif not is_valid_space:
                    print("Please enter a valid starting space for your word.")
                    sep()
                    continue
                else:
                    continue
            else:
                print("Please enter a valid response.")
                sep()
                # time.sleep(1)
        


    def game_loop(self):
        # while True:
        #     current_player = self.who_goes_first()
        #     if current_player:
        #         break
        
        game_over = False
        game_turn = 0
        conceded = False

        # preset hands and player for testing
        current_player = 1
        self.player_1_hand = ['l', 'i', 'n', 'e', 's', 'w', 'a']
        self.player_2_hand = ['p', 'i', 'n', 'e', 's', 'd', 'e']

        # if current_player == 1:
        #     self.draw_tiles(1)
        #     self.draw_tiles(2)
        # else:
        #     self.draw_tiles(2)
        #     self.draw_tiles(1)

        while not game_over:
            game_turn += 1
            display_loop(self.is_tile_present, self.player_1_hand, self.player_2_hand, self.player_1_score, self.player_2_score)
            while True:
                display_loop(self.is_tile_present, self.player_1_hand, self.player_2_hand, self.player_1_score, self.player_2_score)

                choice = input(f"Player {current_player}, Would you like to:\n1. Play a word.\n2. Exchange tiles.\n3. Concede the game.\n\nEnter the number associated with your choice: ")
                sep()

                if game_turn == 1:
                    first_turn = True
                else:
                    first_turn = False


                if choice =='1':
                    is_turn_finished = self.place_word(current_player, first_turn)
                    if is_turn_finished:
                        break

                elif choice == '2':
                    is_turn_finished = self.exchange_tiles(current_player)
                    if is_turn_finished:
                        break

                elif choice == '3':
                    game_over = True
                    conceded = True

                else:
                    print("Please enter a valid option.")
                    sep()

                display_loop(self.is_tile_present, self.player_1_hand, self.player_2_hand, self.player_1_score, self.player_2_score)

            if current_player == 1:
                current_player = 2
            else:
                current_player = 1

            if all(value == 0 for value in self.letter_counts.values()) and (len(self.player_1_hand) == 0 or len(self.player_2_hand) == 0):
                game_over = True

        if conceded:
            sep()
            print(f"GAME OVER!\nPlayer {current_player} wins!!\nThe game lasted {game_turn} turns.\nThe score was: P1: {self.player_1_score} to P2: {self.player_2_score}")
        else:
            which_player_won = 0

            if len(self.player_1_hand) == 0:
                additional_points = 0

                for tile in self.player_2_hand:
                    additional_points += self.letter_point_values[tile]

                self.player_1_score += additional_points
                self.player_2_score -= additional_points
            else:
                additional_points = 0

                for tile in self.player_1_hand:
                    additional_points += self.letter_point_values[tile]

                self.player_1_score -= additional_points
                self.player_2_score += additional_points

            if self.player_1_score > self.player_2_score:
                which_player_won = 1
                sep()
                print(f"GAME OVER!\nPlayer {which_player_won} wins!!\nThe game lasted {game_turn} turns.\nThe score was: P1: {self.player_1_score} to P2: {self.player_2_score}")
                sep()
            elif self.player_1_score == self.player_2_score:
                sep()
                print(f"GAME OVER!\nIt's a tie!!\nThe game lasted {game_turn} turns.\nThe score was: P1: {self.player_1_score} to P2: {self.player_2_score}")
                sep()
            else:
                which_player_won = 2
                sep()
                print(f"GAME OVER!\nPlayer {which_player_won} wins!!\nThe game lasted {game_turn} turns.\nThe score was: P1: {self.player_1_score} to P2: {self.player_2_score}")
                sep()
            
                






if __name__ == "__main__":
    game = Scrabble_Game()

    game.game_loop()