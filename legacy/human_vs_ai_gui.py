import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import os
import sys
import tensorflow as tf
import copy
from valid_moves_script import ScrabbleAllValidMoves, load_dictionary as load_move_gen_dictionary
from game_api import Board_and_Variables
from game_api import Scrabble_Game
from dqn_state import ScrabbleDQNState
from dqn_reward_functions import ScrabbleDQNReward
from dqn_model import ScrabbleDQN
import pygame
from PIL import Image, ImageTk

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

pygame.init()

CELL_SIZE = 50
HAND_CELL_SIZE = 40
BOARD_SIZE = 15
BOARD_PIXELS = 750
HAND_HEIGHT = 120
LABEL_MARGIN = 30
SCORE_HEIGHT = 40
WINDOW_SIZE = (
    BOARD_PIXELS + (LABEL_MARGIN * 2),
    SCORE_HEIGHT + LABEL_MARGIN + BOARD_PIXELS + HAND_HEIGHT
)

TILE_COLOR = (219, 190, 143)
BOARD_COLOR = (245, 245, 220)
GRID_COLOR = (139, 69, 19)
FONT_COLOR = (0, 0, 0)
HAND_BG_COLOR = (230, 230, 230)
LETTER_POINT_VALUES = {
    'a': 1, 'b': 3, 'c': 3, 'd': 2, 'e': 1, 'f': 4, 'g': 2, 'h': 4, 'i': 1, 'j': 8,
    'k': 5, 'l': 1, 'm': 3, 'n': 1, 'o': 1, 'p': 3, 'q': 10, 'r': 1, 's': 1, 't': 1,
    'u': 1, 'v': 4, 'w': 4, 'x': 8, 'y': 4, 'z': 10
}

TRIPLE_WORD = (255, 0, 0)
DOUBLE_WORD = (255, 182, 193)
TRIPLE_LETTER = (0, 0, 255)
DOUBLE_LETTER = (173, 216, 230)

board_font = pygame.font.Font(None, 36)
hand_font = pygame.font.Font(None, 32)
player_font = pygame.font.Font(None, 32)
point_font = pygame.font.Font(None, 20)


load_move_gen_dictionary()

class ScrabbleGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Scrabble DQN Visualizer")
        self.root.geometry("1200x800")
        self.root.resizable(True, True)

        self.game = Scrabble_Game()
        self.state_encoder = ScrabbleDQNState()
        self.reward_calculator = ScrabbleDQNReward()
        self.dqn_agent = None

        self.current_player = 1

        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        self.create_menu_bar()

        self.create_board_frame()

        self.create_control_panel()

        self.create_info_panel()

        self.initialize_game()

    def create_menu_bar(self):
        menu_bar = tk.Menu(self.root)

        file_menu = tk.Menu(menu_bar, tearoff=0)
        file_menu.add_command(label="New Game", command=self.initialize_game)
        file_menu.add_command(label="Load Model", command=self.load_model)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        menu_bar.add_cascade(label="File", menu=file_menu)

        help_menu = tk.Menu(menu_bar, tearoff=0)
        help_menu.add_command(label="Instructions", command=self.show_instructions)
        menu_bar.add_cascade(label="Help", menu=help_menu)

        self.root.config(menu=menu_bar)

    def create_board_frame(self):
        self.board_frame = ttk.LabelFrame(self.main_frame, text="Scrabble Board", padding="10")
        self.board_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        self.board_image_label = tk.Label(self.board_frame)
        self.board_image_label.pack(fill=tk.BOTH, expand=True)

        self.board_surface = pygame.Surface((BOARD_PIXELS + (LABEL_MARGIN * 2), BOARD_PIXELS + (LABEL_MARGIN * 2) + SCORE_HEIGHT ), pygame.SRCALPHA, 32)

        self.main_frame.columnconfigure(0, weight=3)
        self.main_frame.rowconfigure(0, weight=1)

    def create_control_panel(self):
        self.control_frame = ttk.LabelFrame(self.main_frame, text="Control Panel", padding="10")
        self.control_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)

        self.player1_hand_frame = ttk.LabelFrame(self.control_frame, text="Your Hand (P1)", padding="10")
        self.player1_hand_frame.pack(fill=tk.X, pady=(5, 2))
        self.player1_hand_tiles = []

        self.player2_hand_frame = ttk.LabelFrame(self.control_frame, text="AI Hand (P2)", padding="10")
        self.player2_hand_frame.pack(fill=tk.X, pady=(2, 5))
        self.player2_hand_tiles = []

        self.hand_tiles = []

        self.buttons_frame = ttk.Frame(self.control_frame, padding="10")
        self.buttons_frame.pack(fill=tk.X, pady=5)

        self.ai_move_btn = ttk.Button(self.buttons_frame, text="AI Move", command=self.ai_move)
        self.ai_move_btn.pack(fill=tk.X, pady=2)

        self.pass_btn = ttk.Button(self.buttons_frame, text="Pass Turn", command=self.pass_turn)
        self.pass_btn.pack(fill=tk.X, pady=2)

        self.play_btn = ttk.Button(self.buttons_frame, text="Play Move", command=self.play_move)
        self.play_btn.pack(fill=tk.X, pady=2)

        self.move_frame = ttk.LabelFrame(self.control_frame, text="Move Input", padding="10")
        self.move_frame.pack(fill=tk.X, pady=5)

        ttk.Label(self.move_frame, text="Word:").grid(row=0, column=0, sticky="w", pady=2)
        self.word_var = tk.StringVar()
        self.word_entry = ttk.Entry(self.move_frame, textvariable=self.word_var)
        self.word_entry.grid(row=0, column=1, sticky="ew", pady=2)

        ttk.Label(self.move_frame, text="Start Position:").grid(row=1, column=0, sticky="w", pady=2)
        self.start_var = tk.StringVar()
        self.start_entry = ttk.Entry(self.move_frame, textvariable=self.start_var)
        self.start_entry.grid(row=1, column=1, sticky="ew", pady=2)

        ttk.Label(self.move_frame, text="Direction:").grid(row=2, column=0, sticky="w", pady=2)
        self.direction_var = tk.StringVar(value="right")
        self.direction_combo = ttk.Combobox(self.move_frame, textvariable=self.direction_var, values=["right", "down"])
        self.direction_combo.grid(row=2, column=1, sticky="ew", pady=2)

        ttk.Label(self.move_frame, text="Blank Assignments:").grid(row=3, column=0, sticky="w", pady=2)
        self.blank_var = tk.StringVar()
        self.blank_entry = ttk.Entry(self.move_frame, textvariable=self.blank_var)
        self.blank_entry.grid(row=3, column=1, sticky="ew", pady=2)
        ttk.Label(self.move_frame, text="Format: pos1=a,pos2=b").grid(row=4, column=1, sticky="w", pady=0)

        self.move_frame.columnconfigure(1, weight=1)

        self.valid_moves_frame = ttk.LabelFrame(self.control_frame, text="Valid Moves", padding="10")
        self.valid_moves_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        self.valid_moves_list = tk.Listbox(self.valid_moves_frame, height=10)
        self.valid_moves_list.pack(fill=tk.BOTH, expand=True)
        self.valid_moves_list.bind('<<ListboxSelect>>', self.on_move_selected)

        scrollbar = ttk.Scrollbar(self.valid_moves_list, orient="vertical", command=self.valid_moves_list.yview)
        self.valid_moves_list.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.main_frame.columnconfigure(1, weight=1)

    def create_info_panel(self):
        self.info_frame = ttk.LabelFrame(self.main_frame, text="Game Information", padding="10")
        self.info_frame.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=5, pady=5)

        self.status_frame = ttk.Frame(self.info_frame, padding="5")
        self.status_frame.pack(fill=tk.X)

        self.player1_score_var = tk.StringVar(value="Player 1: 0")
        self.player2_score_var = tk.StringVar(value="Player 2: 0")

        ttk.Label(self.status_frame, textvariable=self.player1_score_var, font=("TkDefaultFont", 12, "bold")).pack(side=tk.LEFT, padx=10)
        ttk.Label(self.status_frame, textvariable=self.player2_score_var, font=("TkDefaultFont", 12, "bold")).pack(side=tk.LEFT, padx=10)

        self.current_player_var = tk.StringVar(value="Current Player: 1")
        ttk.Label(self.status_frame, textvariable=self.current_player_var, font=("TkDefaultFont", 12, "bold")).pack(side=tk.LEFT, padx=10)

        self.tiles_remaining_var = tk.StringVar(value="Tiles Remaining: 100")
        ttk.Label(self.status_frame, textvariable=self.tiles_remaining_var).pack(side=tk.LEFT, padx=10)

        self.log_frame = ttk.LabelFrame(self.info_frame, text="Game Log", padding="5")
        self.log_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        self.game_log = tk.Text(self.log_frame, height=5, width=50)
        self.game_log.pack(fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(self.game_log, orient="vertical", command=self.game_log.yview)
        self.game_log.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.main_frame.rowconfigure(1, weight=1)

    def initialize_game(self):
        self.game = Scrabble_Game()
        self.game.initialize_game()
        self.current_player = 1

        self.update_board_display()

        self.update_player_hands_display()

        self.update_game_info()

        self.update_valid_moves()

        self.game_log.delete(1.0, tk.END)
        self.log_game_event("New game started. P1 turn.")

        self.update_button_states()

    def get_square_color(self, row, col):
        if (row, col) in [(0, 0), (0, 7), (0, 14), (7, 0), (7, 14), (14, 0), (14, 7), (14, 14)]:
            return TRIPLE_WORD

        if (row, col) in [(1, 1), (2, 2), (3, 3), (4, 4), (7, 7), (13, 13), (12, 12), (11, 11), (10, 10),
                          (1, 13), (2, 12), (3, 11), (4, 10), (13, 1), (12, 2), (11, 3), (10, 4)]:
            return DOUBLE_WORD

        if (row, col) in [(1, 5), (1, 9), (5, 1), (5, 5), (5, 9), (5, 13),
                          (9, 1), (9, 5), (9, 9), (9, 13), (13, 5), (13, 9)]:
            return TRIPLE_LETTER

        if (row, col) in [(0, 3), (0, 11), (2, 6), (2, 8), (3, 0), (3, 7), (3, 14),
                          (6, 2), (6, 6), (6, 8), (6, 12), (7, 3), (7, 11),
                          (8, 2), (8, 6), (8, 8), (8, 12), (11, 0), (11, 7), (11, 14),
                          (12, 6), (12, 8), (14, 3), (14, 11)]:
            return DOUBLE_LETTER

        return BOARD_COLOR

    def draw_grid_labels(self):
        board_surface = self.board_surface
        for i in range(15):
            label = player_font.render(str(i + 1), True, FONT_COLOR)
            label_rect = label.get_rect(center=(LABEL_MARGIN + (i * CELL_SIZE) + CELL_SIZE//2, LABEL_MARGIN//2 + SCORE_HEIGHT))
            board_surface.blit(label, label_rect)

        for i in range(15):
            label = player_font.render(chr(65 + i), True, FONT_COLOR)
            label_rect = label.get_rect(center=(LABEL_MARGIN//2, LABEL_MARGIN + (i * CELL_SIZE) + CELL_SIZE//2 + SCORE_HEIGHT))
            board_surface.blit(label, label_rect)


    def draw_scores(self, player_1_score, player_2_score):
        board_surface = self.board_surface
        pygame.draw.rect(board_surface, HAND_BG_COLOR, (0, 0, WINDOW_SIZE[0], SCORE_HEIGHT))
        pygame.draw.line(board_surface, GRID_COLOR, (0, SCORE_HEIGHT), (WINDOW_SIZE[0], SCORE_HEIGHT), 2)

        p1_score = player_font.render(f"Player 1 Score: {player_1_score}", True, FONT_COLOR)
        p2_score = player_font.render(f"Player 2 Score: {player_2_score}", True, FONT_COLOR)

        board_surface.blit(p1_score, (10, 10))
        board_surface.blit(p2_score, (WINDOW_SIZE[0] - p2_score.get_rect().width - 10, 10))


    def draw_tile(self, surface, x, y, letter, size, font):
        pygame.draw.rect(surface, TILE_COLOR, (x, y, size, size))
        pygame.draw.rect(surface, GRID_COLOR, (x, y, size, size), 1)

        if letter:
            text = font.render(letter.upper(), True, FONT_COLOR)
            text_rect = text.get_rect(center=(x + size//2, y + size//2))
            surface.blit(text, text_rect)

            point_value = LETTER_POINT_VALUES[letter.lower()] if letter.lower() in LETTER_POINT_VALUES else 0
            point_text = point_font.render(str(point_value), True, FONT_COLOR)
            point_rect = point_text.get_rect(bottomright=(x + size - 2, y + size - 2))
            surface.blit(point_text, point_rect)


    def draw_board(self, is_tile_present):
        board_surface = self.board_surface
        board_surface.fill(BOARD_COLOR)

        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                x = col * CELL_SIZE + LABEL_MARGIN
                y = row * CELL_SIZE + LABEL_MARGIN + SCORE_HEIGHT
                color = self.get_square_color(row, col)
                pygame.draw.rect(board_surface, color, (x, y, CELL_SIZE, CELL_SIZE))
                pygame.draw.rect(board_surface, GRID_COLOR, (x, y, CELL_SIZE, CELL_SIZE), 1)

        for position, (is_present, letter) in is_tile_present.items():
            if is_present:
                row = ord(position[0].lower()) - ord('a')
                col = int(position[1:]) - 1
                x = col * CELL_SIZE + LABEL_MARGIN
                y = row * CELL_SIZE + LABEL_MARGIN + SCORE_HEIGHT
                self.draw_tile(board_surface, x, y, letter, CELL_SIZE, board_font)

        self.draw_grid_labels()
        self.draw_scores(self.game.player_1_score, self.game.player_2_score)


    def update_board_display(self):
        self.draw_board(self.game.is_tile_present)
        
        pygame_surface = self.board_surface
        width, height = pygame_surface.get_size()
        
        raw_str = pygame.image.tostring(pygame_surface, "RGB")
        
        image = Image.frombytes("RGB", (width, height), raw_str)
        
        tk_image = ImageTk.PhotoImage(image=image)
        
        self.board_image_label.configure(image=tk_image)
        self.board_image_label.image = tk_image


    def update_player_hands_display(self):
        for widget in self.player1_hand_frame.winfo_children():
            widget.destroy()
        hand1 = self.game.player_1_hand
        tiles_frame1 = ttk.Frame(self.player1_hand_frame)
        tiles_frame1.pack(fill=tk.X)
        self.player1_hand_tiles = []
        for i, tile in enumerate(hand1):
            display_tile = tile.upper() if tile != '_' else '□'
            tile_lbl = ttk.Label(tiles_frame1, text=display_tile, relief=tk.RAISED, padding=5, width=3, anchor=tk.CENTER)
            tile_lbl.grid(row=0, column=i, padx=2, pady=2)
            self.player1_hand_tiles.append(tile_lbl)

        for widget in self.player2_hand_frame.winfo_children():
            widget.destroy()
        hand2 = self.game.player_2_hand
        tiles_frame2 = ttk.Frame(self.player2_hand_frame)
        tiles_frame2.pack(fill=tk.X)
        self.player2_hand_tiles = []
        for i, tile in enumerate(hand2):
            display_tile = tile.upper() if tile != '_' else '□'
            tile_lbl = ttk.Label(tiles_frame2, text=display_tile, relief=tk.SUNKEN, padding=5, width=3, anchor=tk.CENTER)
            tile_lbl.grid(row=0, column=i, padx=2, pady=2)
            self.player2_hand_tiles.append(tile_lbl)

    def update_game_info(self):
        self.player1_score_var.set(f"Player 1: {self.game.player_1_score}")
        self.player2_score_var.set(f"Player 2: {self.game.player_2_score}")

        self.current_player_var.set(f"Current Player: {self.game.current_player}")

        tiles_remaining = sum(self.game.letter_counts.values())
        self.tiles_remaining_var.set(f"Tiles Remaining: {tiles_remaining}")


    def update_valid_moves(self):
        self.valid_moves_list.delete(0, tk.END)

        self.valid_moves = self.game.get_all_valid_moves(self.current_player)

        self.valid_moves.sort(key=lambda x: x['score'], reverse=True)

        for i, move in enumerate(self.valid_moves):
            move_str = f"{move['word']} at {move['start_space']} {move['direction']} ({move['score']} pts)"
            if 'blank_assignments' in move and move['blank_assignments']:
                blanks = []
                for pos, letter in move['blank_assignments'].items():
                    blanks.append(f"{pos}='{letter}'")
                move_str += f" Blanks: {', '.join(blanks)}"
            self.valid_moves_list.insert(tk.END, move_str)


    def update_button_states(self):
        self.ai_move_btn.config(state=tk.NORMAL if self.dqn_agent else tk.DISABLED)

        self.pass_btn.config(state=tk.NORMAL)

        self.play_btn.config(state=tk.NORMAL if self.valid_moves else tk.DISABLED)

    def load_model(self):
        model_path = filedialog.askopenfilename(
            title="Select Model File",
            filetypes=[("Weights Files", "*.weights.h5"), ("All Files", "*.*")]
        )

        if not model_path:
            return

        try:
            self.dqn_agent = ScrabbleDQN()

            self.dqn_agent.load(model_path)

            self.dqn_agent.epsilon = 0

            messagebox.showinfo("Model Loaded", f"Model loaded successfully from:\n{model_path}")

            self.update_button_states()

            self.log_game_event(f"Loaded model from {os.path.basename(model_path)}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model:\n{str(e)}")

    def ai_move(self):
        if self.current_player != 2:
             messagebox.showwarning("Wait", "It's not the AI's turn.")
             return
        if not self.dqn_agent:
            messagebox.showerror("Error", "No model loaded")
            return

        state = self.state_encoder.get_state_representation(self.game, self.current_player)

        valid_moves = self.game.get_all_valid_moves(self.current_player)

        if not valid_moves:
            messagebox.showinfo("No Valid Moves", "AI has no valid moves. Passing turn.")
            self.pass_turn()
            return

        try:
            ai_player_id = 2
            move = self.dqn_agent.act(state, valid_moves, self.reward_calculator, self.game, ai_player_id)

            success = self.game.play_move(ai_player_id, move['word'], move['start_space'], move['direction'],
                               blank_assignments=move.get('blank_assignments', None))


            if not success:
                 messagebox.showerror("Error", "AI attempted an invalid move.")
                 return

            move_str = f"AI played '{move['word']}' at {move['start_space']} going {move['direction']} for {move['score']} points"
            if 'blank_assignments' in move and move['blank_assignments']:
                blanks = []
                for pos, letter in move['blank_assignments'].items():
                    blanks.append(f"{pos}='{letter}'")
                move_str += f" using blanks: {', '.join(blanks)}"

            self.log_game_event(move_str + f" (AI - Player {ai_player_id})")

            self.current_player = 1
            self.update_board_display()
            self.update_player_hands_display()
            self.update_game_info()
            self.update_valid_moves()
            self.update_button_states()
            self.log_game_event(f"Player {self.current_player}'s turn.")
        except Exception as e:
            messagebox.showerror("Error", f"AI move failed:\n{str(e)}")

    def pass_turn(self):
        passed_player = self.current_player
        success = self.game.pass_turn(passed_player)
        if not success:
             messagebox.showerror("Error", "Failed to pass turn.")
             return

        self.log_game_event(f"Player {passed_player} passed")

        self.current_player = 3 - passed_player
        self.update_game_info()
        self.update_player_hands_display()
        self.update_valid_moves()
        self.update_button_states()
        self.log_game_event(f"Player {self.current_player}'s turn.")


        if self.game.game_over:
            self.handle_game_over()

    def play_move(self):
        if self.current_player != 1:
             messagebox.showwarning("Wait", "It's not your turn.")
             return

        word = self.word_var.get().strip().lower()
        start = self.start_var.get().strip().lower()
        direction = self.direction_var.get().strip().lower()
        blank_str = self.blank_var.get().strip()

        if not word:
            messagebox.showerror("Error", "Please enter a word")
            return

        if not start:
            messagebox.showerror("Error", "Please enter a starting position")
            return

        if not direction or direction not in ["right", "down"]:
            messagebox.showerror("Error", "Direction must be 'right' or 'down'")
            return

        blank_assignments = {}
        if blank_str:
            try:
                assignments = blank_str.split(',')
                for assignment in assignments:
                    pos, letter = assignment.split('=')
                    pos = pos.strip()
                    letter = letter.strip().strip("'\"").lower()
                    blank_assignments[pos] = letter
            except:
                messagebox.showerror("Error", "Invalid blank assignments format")
                return

        try:
            human_player_id = 1
            if not self.game.validate_move(human_player_id, word, start, direction, blank_assignments):
                messagebox.showerror("Error", "Invalid move (check placement, letters, or words).")
                return

            success = self.game.play_move(human_player_id, word, start, direction, blank_assignments)

            if not success:
                 messagebox.showerror("Error", "Failed to play move (Internal Error).")
                 return

            last_move_score = 0
            if self.game.move_history:
                 last_move_data = self.game.move_history[-1]
                 if last_move_data.get('player') == human_player_id and last_move_data.get('type') != 'pass':
                     last_move_score = last_move_data.get('score', 0)

            move_str = f"Player {human_player_id} played '{word}' at {start} going {direction} for {last_move_score} points."

            if blank_assignments:
                blanks = []
                for pos, letter in blank_assignments.items():
                    blanks.append(f"{pos}='{letter}'")
                move_str += f" using blanks: {', '.join(blanks)}"

            self.log_game_event(move_str)

            self.word_var.set("")
            self.start_var.set("")
            self.blank_var.set("")

            self.current_player = 2
            self.update_board_display()
            self.update_player_hands_display()
            self.update_game_info()
            self.update_valid_moves()
            self.update_button_states()
            self.log_game_event(f"Player {self.current_player}'s turn (AI).")

            if self.game.game_over:
                self.handle_game_over()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to play move:\n{str(e)}")

    def on_move_selected(self, event):
        selection = self.valid_moves_list.curselection()
        if not selection:
            return

        index = selection[0]
        if index < 0 or index >= len(self.valid_moves):
            return

        move = self.valid_moves[index]

        self.word_var.set(move['word'])
        self.start_var.set(move['start_space'])
        self.direction_var.set(move['direction'])

        if 'blank_assignments' in move and move['blank_assignments']:
            blank_str = []
            for pos, letter in move['blank_assignments'].items():
                blank_str.append(f"{pos}={letter}")
            self.blank_var.set(','.join(blank_str))
        else:
            self.blank_var.set("")

    def log_game_event(self, event_str):
        self.game_log.insert(tk.END, f"{event_str}\n")
        self.game_log.see(tk.END)

    def handle_game_over(self):
        self.log_game_event("Game Over!")
        self.log_game_event(f"Final scores: Player 1: {self.game.player_1_score}, Player 2: {self.game.player_2_score}")

        if self.game.player_1_score > self.game.player_2_score:
            winner = "Player 1"
        elif self.game.player_1_score < self.game.player_2_score:
            winner = "Player 2"
        else:
            winner = "It's a tie!"

        self.log_game_event(f"Winner: {winner}")

        messagebox.showinfo("Game Over", f"Game over!\nFinal scores:\nPlayer 1: {self.game.player_1_score}\nPlayer 2: {self.game.player_2_score}\n\nWinner: {winner}")

    def show_instructions(self):
        instructions = """
        Scrabble Human vs AI GUI

        Play against a trained DQN Model

        Instructions:
        1. Load a trained model using the "Load Model" option under File.
        2. Play against the AI by either:
           - Selecting a move from the Valid Moves list
           - Entering a move manually in the Move Input fields
        3. The game log shows the history of moves and events.
        4. The board displays the current state of the game.

        Controls:
        - AI Move: Let the AI make a move
        - Pass Turn: Skip your turn
        - Play Move: Play the move entered in the Move Input fields

        Move Input:
        - Word: The word to play
        - Start Position: The starting position (e.g., b3)
        - Direction: The direction (right or down)
        - Blank Assignments: Format: pos1=a,pos2=b

        For example, if using a blank tile as 'r' in position b3, enter: b3=r
        """

        instructions_window = tk.Toplevel(self.root)
        instructions_window.title("Instructions")
        instructions_window.geometry("600x500")

        text = tk.Text(instructions_window, wrap=tk.WORD, padx=10, pady=10)
        text.pack(fill=tk.BOTH, expand=True)
        text.insert(tk.END, instructions)
        text.config(state=tk.DISABLED)


def main():
    root = tk.Tk()
    app = ScrabbleGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()