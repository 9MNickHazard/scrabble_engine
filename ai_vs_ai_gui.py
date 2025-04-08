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

BLANK_FONT_COLOR = (0, 128, 0)
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

class ScrabbleAIvsAIGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Scrabble AI vs AI Visualizer")
        self.root.geometry("1200x800")
        self.root.resizable(True, True)

        self.game = Scrabble_Game()
        self.state_encoder = ScrabbleDQNState()
        self.reward_calculator = ScrabbleDQNReward()
        
        self.dqn_agent1 = None
        self.dqn_agent2 = None
        
        self.current_player = 1
        
        self.autoplay = False
        self.paused = False
        self.autoplay_speed = 1000
        self.autoplay_job_id = None
        
        self.current_top_moves = []

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
        file_menu.add_command(label="Load Model 1", command=lambda: self.load_model(1))
        file_menu.add_command(label="Load Model 2", command=lambda: self.load_model(2))
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        menu_bar.add_cascade(label="File", menu=file_menu)

        help_menu = tk.Menu(menu_bar, tearoff=0)
        help_menu.add_command(label="Instructions", command=self.show_instructions)
        help_menu.add_command(label="About", command=self.show_about)
        menu_bar.add_cascade(label="Help", menu=help_menu)

        self.root.config(menu=menu_bar)

    def create_board_frame(self):
        self.board_frame = ttk.LabelFrame(self.main_frame, text="Scrabble Board", padding="10")
        self.board_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        self.board_image_label = tk.Label(self.board_frame)
        self.board_image_label.pack(fill=tk.BOTH, expand=True)

        self.board_surface = pygame.Surface((BOARD_PIXELS + (LABEL_MARGIN * 2), BOARD_PIXELS + (LABEL_MARGIN * 2) + SCORE_HEIGHT), pygame.SRCALPHA, 32)

        self.main_frame.columnconfigure(0, weight=3)
        self.main_frame.rowconfigure(0, weight=1)

    def create_control_panel(self):
        self.control_frame = ttk.LabelFrame(self.main_frame, text="Control Panel", padding="10")
        self.control_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)

        self.player1_hand_frame = ttk.LabelFrame(self.control_frame, text="AI 1 Hand (P1)", padding="10")
        self.player1_hand_frame.pack(fill=tk.X, pady=(5, 2))
        self.player1_hand_tiles = []

        self.player2_hand_frame = ttk.LabelFrame(self.control_frame, text="AI 2 Hand (P2)", padding="10")
        self.player2_hand_frame.pack(fill=tk.X, pady=(2, 5))
        self.player2_hand_tiles = []

        self.buttons_frame = ttk.Frame(self.control_frame, padding="10")
        self.buttons_frame.pack(fill=tk.X, pady=5)

        self.start_autoplay_btn = ttk.Button(self.buttons_frame, text="Start AI vs AI Game", command=self.toggle_autoplay)
        self.start_autoplay_btn.pack(fill=tk.X, pady=2)

        self.pause_btn = ttk.Button(self.buttons_frame, text="Pause", command=self.pause_resume_game)
        self.pause_btn.pack(fill=tk.X, pady=2)
        self.pause_btn.config(state=tk.DISABLED)

        self.model_info_frame = ttk.LabelFrame(self.control_frame, text="Model Information", padding="10")
        self.model_info_frame.pack(fill=tk.X, pady=5)
        
        self.model1_info_var = tk.StringVar(value="Model 1: Not loaded")
        ttk.Label(self.model_info_frame, textvariable=self.model1_info_var).pack(anchor=tk.W, pady=2)
        
        self.model2_info_var = tk.StringVar(value="Model 2: Not loaded")
        ttk.Label(self.model_info_frame, textvariable=self.model2_info_var).pack(anchor=tk.W, pady=2)

        self.top_moves_frame = ttk.LabelFrame(self.control_frame, text="Top AI Moves Considered", padding="10")
        self.top_moves_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        self.top_moves_list = tk.Listbox(self.top_moves_frame, height=10)
        self.top_moves_list.pack(fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(self.top_moves_list, orient="vertical", command=self.top_moves_list.yview)
        self.top_moves_list.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.main_frame.columnconfigure(1, weight=1)

    def create_info_panel(self):
        self.info_frame = ttk.LabelFrame(self.main_frame, text="Game Information", padding="10")
        self.info_frame.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=5, pady=5)

        self.status_frame = ttk.Frame(self.info_frame, padding="5")
        self.status_frame.pack(fill=tk.X)

        self.player1_score_var = tk.StringVar(value="AI 1 (P1): 0")
        self.player2_score_var = tk.StringVar(value="AI 2 (P2): 0")

        ttk.Label(self.status_frame, textvariable=self.player1_score_var, font=("TkDefaultFont", 12, "bold")).pack(side=tk.LEFT, padx=10)
        ttk.Label(self.status_frame, textvariable=self.player2_score_var, font=("TkDefaultFont", 12, "bold")).pack(side=tk.LEFT, padx=10)

        self.current_player_var = tk.StringVar(value="Current Player: AI 1 (P1)")
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
        if self.autoplay_job_id:
            self.root.after_cancel(self.autoplay_job_id)
            self.autoplay_job_id = None
        
        self.autoplay = False
        self.paused = False
        
        self.start_autoplay_btn.config(text="Start AI vs AI Game")
        self.pause_btn.config(text="Pause", state=tk.DISABLED)

        self.game = Scrabble_Game()
        self.game.initialize_game()
        self.current_player = 1

        self.update_board_display()
        self.update_player_hands_display()
        self.update_game_info()
        self.update_top_moves_display()

        self.game_log.delete(1.0, tk.END)
        self.log_game_event("New game initialized. Ready to start AI vs AI game.")

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

        p1_score = player_font.render(f"AI 1 Score: {player_1_score}", True, FONT_COLOR)
        p2_score = player_font.render(f"AI 2 Score: {player_2_score}", True, FONT_COLOR)

        board_surface.blit(p1_score, (10, 10))
        board_surface.blit(p2_score, (WINDOW_SIZE[0] - p2_score.get_rect().width - 10, 10))

    def draw_tile(self, surface, x, y, letter, size, font, is_blank=False):
        pygame.draw.rect(surface, TILE_COLOR, (x, y, size, size))
        pygame.draw.rect(surface, GRID_COLOR, (x, y, size, size), 1)

        if letter:
            text_color = BLANK_FONT_COLOR if is_blank else FONT_COLOR
            
            text = font.render(letter.upper(), True, text_color)
            text_rect = text.get_rect(center=(x + size//2, y + size//2))
            surface.blit(text, text_rect)

            point_value = 0 if is_blank else LETTER_POINT_VALUES.get(letter.lower(), 0)
            point_text = point_font.render(str(point_value), True, FONT_COLOR)
            point_rect = point_text.get_rect(bottomright=(x + size - 2, y + size - 2))
            surface.blit(point_text, point_rect)

    def draw_board_pygame(self, is_tile_present):
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
                
                is_blank = position in self.game.blank_tile_assignments
                
                self.draw_tile(board_surface, x, y, letter, CELL_SIZE, board_font, is_blank)

        self.draw_grid_labels()
        self.draw_scores(self.game.player_1_score, self.game.player_2_score)

    def update_board_display(self):
        self.draw_board_pygame(self.game.is_tile_present)
        
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
        self.player1_score_var.set(f"AI 1 (P1): {self.game.player_1_score}")
        self.player2_score_var.set(f"AI 2 (P2): {self.game.player_2_score}")

        player_name = "AI 1 (P1)" if self.current_player == 1 else "AI 2 (P2)"
        self.current_player_var.set(f"Current Player: {player_name}")

        tiles_remaining = sum(self.game.letter_counts.values())
        self.tiles_remaining_var.set(f"Tiles Remaining: {tiles_remaining}")

    def update_top_moves_display(self):
        self.top_moves_list.delete(0, tk.END)
        
        for i, move in enumerate(self.current_top_moves[:10]):
            move_str = f"{move['word']} at {move['start_space']} {move['direction']} ({move['score']} pts)"
            if 'blank_assignments' in move and move['blank_assignments']:
                blanks = []
                for pos, letter in move['blank_assignments'].items():
                    blanks.append(f"{pos}='{letter}'")
                move_str += f" Blanks: {', '.join(blanks)}"
            self.top_moves_list.insert(tk.END, move_str)

    def update_button_states(self):
        if self.dqn_agent1 and self.dqn_agent2:
            self.start_autoplay_btn.config(state=tk.NORMAL)
        else:
            self.start_autoplay_btn.config(state=tk.DISABLED)
            
        if self.autoplay:
            self.pause_btn.config(state=tk.NORMAL)
        else:
            self.pause_btn.config(state=tk.DISABLED)

    def load_model(self, player_id):
        model_path = filedialog.askopenfilename(
            title=f"Select Model for AI {player_id}",
            filetypes=[("Weights Files", "*.weights.h5"), ("All Files", "*.*")]
        )

        if not model_path:
            return

        try:
            if player_id == 1:
                self.dqn_agent1 = ScrabbleDQN()
                self.dqn_agent1.load(model_path)
                self.dqn_agent1.epsilon = 0
                self.model1_info_var.set(f"Model 1: {os.path.basename(model_path)}")
            else:
                self.dqn_agent2 = ScrabbleDQN()
                self.dqn_agent2.load(model_path)
                self.dqn_agent2.epsilon = 0
                self.model2_info_var.set(f"Model 2: {os.path.basename(model_path)}")

            messagebox.showinfo("Model Loaded", f"Model for AI {player_id} loaded successfully from:\n{model_path}")

            self.update_button_states()

            self.log_game_event(f"Loaded model for AI {player_id} from {os.path.basename(model_path)}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model:\n{str(e)}")

    def toggle_autoplay(self):
        if not self.autoplay:
            if not self.dqn_agent1 or not self.dqn_agent2:
                messagebox.showerror("Error", "Both AI models must be loaded before starting autoplay")
                return
                
            self.autoplay = True
            self.paused = False
            self.start_autoplay_btn.config(text="Stop AI vs AI Game")
            self.pause_btn.config(text="Pause", state=tk.NORMAL)
            
            self.log_game_event("Starting AI vs AI autoplay")
            
            self.autoplay_job_id = self.root.after(self.autoplay_speed, self.auto_play_move)
        else:
            self.autoplay = False
            self.paused = False
            self.start_autoplay_btn.config(text="Start AI vs AI Game")
            self.pause_btn.config(text="Pause", state=tk.DISABLED)
            
            if self.autoplay_job_id:
                self.root.after_cancel(self.autoplay_job_id)
                self.autoplay_job_id = None
                
            self.log_game_event("Stopped AI vs AI autoplay")

    def pause_resume_game(self):
        if not self.autoplay:
            return
            
        if self.paused:
            self.paused = False
            self.pause_btn.config(text="Pause")
            
            self.autoplay_job_id = self.root.after(self.autoplay_speed, self.auto_play_move)
            
            self.log_game_event("Resumed Game")
        else:
            self.paused = True
            self.pause_btn.config(text="Resume")
            
            if self.autoplay_job_id:
                self.root.after_cancel(self.autoplay_job_id)
                self.autoplay_job_id = None
                
            self.log_game_event("Paused Game")

    def auto_play_move(self):
        if not self.autoplay or self.paused or self.game.game_over:
            return
            
        current_agent = self.dqn_agent1 if self.current_player == 1 else self.dqn_agent2
        
        state = self.state_encoder.get_state_representation(self.game, self.current_player)
        
        valid_moves = self.game.get_all_valid_moves(self.current_player)
        
        self.current_top_moves = valid_moves[:10] if valid_moves else []
        self.update_top_moves_display()
        
        if not valid_moves:
            self.log_game_event(f"AI {self.current_player} has no valid moves. Passing turn.")
            self.pass_turn()
        else:
            try:
                move = current_agent.act(state, valid_moves, self.reward_calculator, self.game, self.current_player)
                
                success = self.game.play_move(
                    self.current_player, 
                    move['word'], 
                    move['start_space'], 
                    move['direction'],
                    blank_assignments=move.get('blank_assignments', None)
                )
                
                if not success:
                    self.log_game_event(f"AI {self.current_player} attempted an invalid move. Passing turn.")
                    self.pass_turn()
                    return
                
                move_str = f"AI {self.current_player} played '{move['word']}' at {move['start_space']} going {move['direction']} for {move['score']} points"
                if 'blank_assignments' in move and move['blank_assignments']:
                    blanks = []
                    for pos, letter in move['blank_assignments'].items():
                        blanks.append(f"{pos}='{letter}'")
                    move_str += f" using blanks: {', '.join(blanks)}"
                
                self.log_game_event(move_str)
                
                self.current_player = 3 - self.current_player
                
                self.update_board_display()
                self.update_player_hands_display()
                self.update_game_info()
                
                self.log_game_event(f"AI {self.current_player}'s turn")
                
            except Exception as e:
                self.log_game_event(f"Error during AI {self.current_player} move: {str(e)}. Passing turn.")
                self.pass_turn()
        
        if self.game.game_over:
            self.handle_game_over()
            return
            
        self.autoplay_job_id = self.root.after(self.autoplay_speed, self.auto_play_move)

    def pass_turn(self):
        self.game.pass_turn(self.current_player)
        
        self.log_game_event(f"AI {self.current_player} passed")
        
        self.current_player = 3 - self.current_player
        
        self.update_game_info()
        self.update_player_hands_display()
        
        self.log_game_event(f"AI {self.current_player}'s turn")
        
        if self.game.game_over:
            self.handle_game_over()

    def log_game_event(self, event_str):
        self.game_log.insert(tk.END, f"{event_str}\n")
        self.game_log.see(tk.END)

    def handle_game_over(self):
        self.autoplay = False
        if self.autoplay_job_id:
            self.root.after_cancel(self.autoplay_job_id)
            self.autoplay_job_id = None
            
        self.start_autoplay_btn.config(text="Start Game")
        self.pause_btn.config(state=tk.DISABLED)
        
        self.log_game_event("Game Over!")
        self.log_game_event(f"Final scores: AI 1: {self.game.player_1_score}, AI 2: {self.game.player_2_score}")

        if self.game.player_1_score > self.game.player_2_score:
            winner = "AI 1 (P1)"
        elif self.game.player_1_score < self.game.player_2_score:
            winner = "AI 2 (P2)"
        else:
            winner = "It's a tie!"

        self.log_game_event(f"Winner: {winner}")

        messagebox.showinfo("Game Over", f"Game over!\nFinal scores:\nAI 1: {self.game.player_1_score}\nAI 2: {self.game.player_2_score}\n\nWinner: {winner}")

    def show_instructions(self):
        instructions = """
        Scrabble AI vs AI Visualizer

        Instructions:
        1. Load both AI models using the options in the File menu
        2. Click "Start AI vs AI Game" to begin automatic play
        3. Use the "Pause" button to pause/resume the game
        4. Adjust the speed slider to control how fast the AIs play
        5. The game log shows the history of moves and events
        6. The top moves list shows the moves being considered by the current AI

        Controls:
        - Start/Stop: Begin or end the AI vs AI game
        - Pause/Resume: Temporarily stop or continue the game

        The game ends when:
        - Both AIs pass consecutively
        - One AI uses all it's tiles and no tiles remain in the bag
        """

        instructions_window = tk.Toplevel(self.root)
        instructions_window.title("Instructions")
        instructions_window.geometry("600x500")

        text = tk.Text(instructions_window, wrap=tk.WORD, padx=10, pady=10)
        text.pack(fill=tk.BOTH, expand=True)
        text.insert(tk.END, instructions)
        text.config(state=tk.DISABLED)

    def show_about(self):
        about_text = """
        Scrabble AI vs AI Visualizer

        A GUI for visualizing two Deep Q-Network (DQN) models
        playing Scrabble against each other.

        This application allows you to:
        - Load two different trained DQN models
        - Watch them play automatically against each other
        - Visualize the game board and moves
        - See what moves each AI is considering
        - Track the game's progress with detailed logging

        The visualization uses Pygame embedded in a Tkinter interface.
        """

        messagebox.showinfo("About", about_text)

def main():
    root = tk.Tk()
    app = ScrabbleAIvsAIGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()