import numpy as np
import matplotlib.pyplot as plt
import time
import os
import random
import copy
from game_api import Board_and_Variables

class ScrabbleDQNEvaluator:
    """
    Class for evaluating the Scrabble DQN model.
    Includes methods for comparing against baselines and analyzing move patterns.
    """
    
    def __init__(self, game_api_class, state_encoder, reward_calculator, results_dir="./results"):
        self.game_api_class = game_api_class
        self.state_encoder = state_encoder
        self.reward_calculator = reward_calculator
        
        os.makedirs(results_dir, exist_ok=True)
        self.results_dir = results_dir
    
    def evaluate_against_baseline(self, dqn_agent, baseline_type="highest_score", num_games=50, verbose=True):
        """
        Evaluate DQN agent against a baseline strategy
        
        Args:
            dqn_agent: Trained DQN agent
            baseline_type: Type of baseline ("random", "highest_score", "strategic")
            num_games: Number of games to play
            verbose: Whether to print progress
            
        Returns:
            dict: Evaluation results
        """
        original_epsilon = dqn_agent.epsilon
        
        dqn_agent.epsilon = 0
        
        wins = 0
        losses = 0
        draws = 0
        dqn_scores = []
        baseline_scores = []
        dqn_blank_usage = 0
        baseline_blank_usage = 0
        game_lengths = []
        
        for game_num in range(num_games):
            if verbose and game_num % 10 == 0:
                print(f"Playing game {game_num + 1}/{num_games}")
                
            game = self.game_api_class()
            game.initialize_game()
            
            moves_count = 0
            
            while not game.game_over:
                if game.current_player == 1:
                    state = self.state_encoder.get_state_representation(game, 1)
                    
                    valid_moves = game.get_all_valid_moves(1)
                    
                    if not valid_moves:
                        game.pass_turn(1)
                        continue
                    
                    move = dqn_agent.act(state, valid_moves, self.reward_calculator, game, 1)
                    
                    if 'blank_assignments' in move and move['blank_assignments']:
                        dqn_blank_usage += 1
                    
                    game.play_move(1, move['word'], move['start_space'], move['direction'],
                                  blank_assignments=move.get('blank_assignments', None))
                    
                    moves_count += 1
                
                else:
                    valid_moves = game.get_all_valid_moves(2)
                    
                    if not valid_moves:
                        game.pass_turn(2)
                        continue
                    
                    if baseline_type == "random":
                        move = random.choice(valid_moves)
                    elif baseline_type == "highest_score":
                        move = max(valid_moves, key=lambda m: m['score'])
                    elif baseline_type == "strategic":
                        scored_moves = self.reward_calculator.estimate_move_quality(game, 2, valid_moves)
                        move = scored_moves[0]['move']
                    
                    if 'blank_assignments' in move and move['blank_assignments']:
                        baseline_blank_usage += 1
                    
                    game.play_move(2, move['word'], move['start_space'], move['direction'],
                                  blank_assignments=move.get('blank_assignments', None))
                    
                    moves_count += 1
            
            dqn_score = game.player_1_score
            baseline_score = game.player_2_score
            
            dqn_scores.append(dqn_score)
            baseline_scores.append(baseline_score)
            game_lengths.append(moves_count)
            
            if dqn_score > baseline_score:
                wins += 1
            elif dqn_score < baseline_score:
                losses += 1
            else:
                draws += 1
        
        dqn_agent.epsilon = original_epsilon
        
        win_rate = wins / num_games
        avg_dqn_score = np.mean(dqn_scores)
        avg_baseline_score = np.mean(baseline_scores)
        avg_score_diff = np.mean(np.array(dqn_scores) - np.array(baseline_scores))
        avg_game_length = np.mean(game_lengths)
        avg_dqn_blank_usage = dqn_blank_usage / num_games
        avg_baseline_blank_usage = baseline_blank_usage / num_games
        
        if verbose:
            print(f"\nEvaluation against {baseline_type} baseline:")
            print(f"Win rate: {win_rate:.2f} ({wins} wins, {losses} losses, {draws} draws)")
            print(f"Average DQN score: {avg_dqn_score:.2f}")
            print(f"Average baseline score: {avg_baseline_score:.2f}")
            print(f"Average score difference: {avg_score_diff:.2f}")
            print(f"Average game length: {avg_game_length:.2f} moves")
            print(f"Average DQN blank usage: {avg_dqn_blank_usage:.2f} per game")
            print(f"Average baseline blank usage: {avg_baseline_blank_usage:.2f} per game")
        
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 2, 1)
        plt.hist(dqn_scores, alpha=0.5, label='DQN')
        plt.hist(baseline_scores, alpha=0.5, label=baseline_type)
        plt.title('Score Distribution')
        plt.xlabel('Score')
        plt.ylabel('Frequency')
        plt.legend()
        
        plt.subplot(2, 2, 2)
        score_diffs = np.array(dqn_scores) - np.array(baseline_scores)
        plt.hist(score_diffs, bins=20)
        plt.title('Score Difference (DQN - Baseline)')
        plt.xlabel('Score Difference')
        plt.ylabel('Frequency')
        
        plt.subplot(2, 2, 3)
        plt.hist(game_lengths, bins=20)
        plt.title('Game Length Distribution')
        plt.xlabel('Number of Moves')
        plt.ylabel('Frequency')
        
        plt.subplot(2, 2, 4)
        plt.bar(['DQN', baseline_type], [avg_dqn_blank_usage, avg_baseline_blank_usage])
        plt.title('Average Blank Tile Usage per Game')
        plt.ylabel('Blank Tiles Used')
        
        plt.tight_layout()
        plt.savefig(f"{self.results_dir}/evaluation_vs_{baseline_type}.png")
        
        if verbose:
            plt.show()
        
        results = {
            'win_rate': win_rate,
            'wins': wins,
            'losses': losses,
            'draws': draws,
            'avg_dqn_score': avg_dqn_score,
            'avg_baseline_score': avg_baseline_score,
            'avg_score_diff': avg_score_diff,
            'avg_game_length': avg_game_length,
            'dqn_scores': dqn_scores,
            'baseline_scores': baseline_scores,
            'game_lengths': game_lengths,
            'dqn_blank_usage': dqn_blank_usage,
            'baseline_blank_usage': baseline_blank_usage
        }
        
        return results
    
    def analyze_move_patterns(self, dqn_agent, num_games=20, verbose=True):
        """
        Analyze patterns in moves chosen by the DQN agent
        
        Args:
            dqn_agent: Trained DQN agent
            num_games: Number of games to analyze
            verbose: Whether to print progress
            
        Returns:
            dict: Analysis results
        """
        original_epsilon = dqn_agent.epsilon
        
        dqn_agent.epsilon = 0
        
        word_lengths = []
        score_distribution = []
        premium_square_usage = {
            'triple_word': 0,
            'double_word': 0,
            'triple_letter': 0,
            'double_letter': 0
        }
        blank_usage = 0
        total_moves = 0
        
        for game_num in range(num_games):
            if verbose and game_num % 5 == 0:
                print(f"Analyzing game {game_num + 1}/{num_games}")
                
            game = self.game_api_class()
            game.initialize_game()
            
            while not game.game_over:
                if game.current_player == 1:
                    state = self.state_encoder.get_state_representation(game, 1)
                    
                    valid_moves = game.get_all_valid_moves(1)
                    
                    if not valid_moves:
                        game.pass_turn(1)
                        continue
                    
                    move = dqn_agent.act(state, valid_moves, self.reward_calculator, game, 1)
                    
                    word_lengths.append(len(move['word']))
                    score_distribution.append(move['score'])
                    total_moves += 1
                    
                    if 'blank_assignments' in move and move['blank_assignments']:
                        blank_usage += len(move['blank_assignments'])
                    
                    for pos in move['positions']:
                        if pos in game.special_spaces and game.special_spaces[pos]:
                            premium_square_usage[game.special_spaces[pos]] += 1
                    
                    game.play_move(1, move['word'], move['start_space'], move['direction'],
                                  blank_assignments=move.get('blank_assignments', None))
                
                else:
                    valid_moves = game.get_all_valid_moves(2)
                    
                    if not valid_moves:
                        game.pass_turn(2)
                        continue
                    
                    move = random.choice(valid_moves)
                    
                    game.play_move(2, move['word'], move['start_space'], move['direction'],
                                  blank_assignments=move.get('blank_assignments', None))
        
        dqn_agent.epsilon = original_epsilon
        
        avg_word_length = np.mean(word_lengths)
        avg_score = np.mean(score_distribution)
        blank_usage_rate = blank_usage / total_moves
        
        for key in premium_square_usage:
            premium_square_usage[key] /= total_moves
        
        if verbose:
            print("\nMove Pattern Analysis:")
            print(f"Average word length: {avg_word_length:.2f}")
            print(f"Average move score: {avg_score:.2f}")
            print(f"Blank tile usage rate: {blank_usage_rate:.2f} per move")
            print("Premium square usage rates:")
            for square_type, rate in premium_square_usage.items():
                print(f"  {square_type}: {rate:.2f}")
        
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 2, 1)
        plt.hist(word_lengths, bins=range(2, 16), alpha=0.7)
        plt.title('Word Length Distribution')
        plt.xlabel('Word Length')
        plt.ylabel('Frequency')
        plt.xticks(range(2, 16))
        
        plt.subplot(2, 2, 2)
        plt.hist(score_distribution, bins=20, alpha=0.7)
        plt.title('Score Distribution')
        plt.xlabel('Score')
        plt.ylabel('Frequency')
        
        plt.subplot(2, 2, 3)
        square_types = list(premium_square_usage.keys())
        usage_rates = [premium_square_usage[t] for t in square_types]
        plt.bar(square_types, usage_rates)
        plt.title('Premium Square Usage Rate')
        plt.ylabel('Usage Rate')
        plt.xticks(rotation=45)
        
        plt.subplot(2, 2, 4)
        plt.scatter(word_lengths, score_distribution, alpha=0.5)
        plt.title('Word Length vs. Score')
        plt.xlabel('Word Length')
        plt.ylabel('Score')
        
        plt.tight_layout()
        plt.savefig(f"{self.results_dir}/move_pattern_analysis.png")
        
        if verbose:
            plt.show()
        
        results = {
            'avg_word_length': avg_word_length,
            'avg_score': avg_score,
            'blank_usage_rate': blank_usage_rate,
            'premium_square_usage': premium_square_usage,
            'word_lengths': word_lengths,
            'score_distribution': score_distribution,
            'total_moves': total_moves
        }
        
        return results
    
    def visualize_game(self, dqn_agent, opponent_type="random", save_path=None, verbose=True):
        """
        Play a single game and visualize the board state after each move
        
        Args:
            dqn_agent: Trained DQN agent
            opponent_type: Type of opponent ("random", "highest_score", "strategic")
            save_path: Path to save the visualization
            verbose: Whether to print progress
            
        Returns:
            list: Move history
        """
        original_epsilon = dqn_agent.epsilon
        
        dqn_agent.epsilon = 0
        
        game = self.game_api_class()
        game.initialize_game()
        
        move_history = []
        
        while not game.game_over:
            if game.current_player == 1:
                state = self.state_encoder.get_state_representation(game, 1)
                
                valid_moves = game.get_all_valid_moves(1)
                
                if not valid_moves:
                    game.pass_turn(1)
                    move_data = {
                        'player': 1,
                        'type': 'pass',
                        'turn': len(move_history) + 1,
                        'player_1_score': game.player_1_score,
                        'player_2_score': game.player_2_score
                    }
                    move_history.append(move_data)
                    continue
                
                move = dqn_agent.act(state, valid_moves, self.reward_calculator, game, 1)
                
                move_data = {
                    'player': 1,
                    'type': 'move',
                    'turn': len(move_history) + 1,
                    'word': move['word'],
                    'start_space': move['start_space'],
                    'direction': move['direction'],
                    'score': move['score'],
                    'positions': move.get('positions', []),
                    'blank_assignments': move.get('blank_assignments', {}),
                    'player_1_score': game.player_1_score,
                    'player_2_score': game.player_2_score
                }
                move_history.append(move_data)
                
                game.play_move(1, move['word'], move['start_space'], move['direction'],
                              blank_assignments=move.get('blank_assignments', None))
            
            else:
                valid_moves = game.get_all_valid_moves(2)
                
                if not valid_moves:
                    game.pass_turn(2)
                    move_data = {
                        'player': 2,
                        'type': 'pass',
                        'turn': len(move_history) + 1,
                        'player_1_score': game.player_1_score,
                        'player_2_score': game.player_2_score
                    }
                    move_history.append(move_data)
                    continue
                
                if opponent_type == "random":
                    move = random.choice(valid_moves)
                elif opponent_type == "highest_score":
                    move = max(valid_moves, key=lambda m: m['score'])
                elif opponent_type == "strategic":
                    scored_moves = self.reward_calculator.estimate_move_quality(game, 2, valid_moves)
                    move = scored_moves[0]['move']
                else:
                    move = max(valid_moves, key=lambda m: m['score'])
                
                move_data = {
                    'player': 2,
                    'type': 'move',
                    'turn': len(move_history) + 1,
                    'word': move['word'],
                    'start_space': move['start_space'],
                    'direction': move['direction'],
                    'score': move['score'],
                    'positions': move.get('positions', []),
                    'blank_assignments': move.get('blank_assignments', {}),
                    'player_1_score': game.player_1_score,
                    'player_2_score': game.player_2_score
                }
                move_history.append(move_data)
                
                game.play_move(2, move['word'], move['start_space'], move['direction'],
                              blank_assignments=move.get('blank_assignments', None))
        
        self.create_game_visualization(game, move_history, save_path)
        
        dqn_agent.epsilon = original_epsilon
        
        return move_history
    
    def create_game_visualization(self, game, move_history, save_path=None):
        """
        Create a visualization of the game
        
        Args:
            game: Final game state
            move_history: History of moves
            save_path: Path to save the visualization
        """
        fig = plt.figure(figsize=(15, 10))
        
        ax_board = plt.subplot2grid((2, 3), (0, 0), colspan=2, rowspan=2)
        self.visualize_board(ax_board, game.is_tile_present, game.blank_tile_assignments)
        
        ax_history = plt.subplot2grid((2, 3), (0, 2), rowspan=1)
        self.visualize_move_history(ax_history, move_history)
        
        ax_scores = plt.subplot2grid((2, 3), (1, 2), rowspan=1)
        self.visualize_score_progression(ax_scores, move_history)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
        
        plt.close(fig)
    
    def visualize_board(self, ax, is_tile_present, blank_tile_assignments=None):
        """
        Visualize the Scrabble board
        
        Args:
            ax: Matplotlib axes
            is_tile_present: Dictionary mapping positions to [is_present, letter]
            blank_tile_assignments: Dictionary mapping positions to assigned letters for blank tiles
        """
        grid = np.zeros((15, 15), dtype=object)
        grid.fill('')
        
        for pos, (is_present, letter) in is_tile_present.items():
            if is_present:
                row = ord(pos[0]) - ord('a')
                col = int(pos[1:]) - 1
                
                is_blank = blank_tile_assignments and pos in blank_tile_assignments
                
                if is_blank:
                    grid[row, col] = letter.lower()
                else:
                    grid[row, col] = letter.upper()
        
        table = ax.table(cellText=grid, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        
        for pos, special in Board_and_Variables.special_spaces.items():
            if special:
                row = ord(pos[0]) - ord('a')
                col = int(pos[1:]) - 1
                
                cell = table[(row, col)]
                if special == 'triple_word':
                    cell.set_facecolor('#ff9999')
                elif special == 'double_word':
                    cell.set_facecolor('#ffcccc')
                elif special == 'triple_letter':
                    cell.set_facecolor('#99ccff')
                elif special == 'double_letter':
                    cell.set_facecolor('#ccccff')
        
        for i in range(15):
            ax.text(i/15 + 0.035, -0.02, str(i+1), transform=ax.transAxes)
            ax.text(-0.02, i/15 + 0.035, chr(ord('a') + i), transform=ax.transAxes)
        
        ax.axis('off')
        
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#ff9999', label='Triple Word'),
            Patch(facecolor='#ffcccc', label='Double Word'),
            Patch(facecolor='#99ccff', label='Triple Letter'),
            Patch(facecolor='#ccccff', label='Double Letter')
        ]
        ax.legend(handles=legend_elements, loc='upper center', 
                 bbox_to_anchor=(0.5, 1.05), ncol=4, fontsize='small')
        
        ax.set_title('Final Board State')
    
    def visualize_move_history(self, ax, move_history):
        """
        Visualize the move history
        
        Args:
            ax: Matplotlib axes
            move_history: List of move data dictionaries
        """
        move_texts = []
        
        for move in move_history:
            player = move['player']
            turn = move.get('turn', 0)
            
            if move.get('type') == 'pass':
                move_texts.append(f"Turn {turn}: Player {player} passed")
            elif move.get('type') == 'game_over':
                move_texts.append(f"Game Over")
                move_texts.append(f"Player 1: {move['player_1_score']}")
                move_texts.append(f"Player 2: {move['player_2_score']}")
            else:
                word = move.get('word', '')
                start = move.get('start_space', '')
                direction = move.get('direction', '')
                score = move.get('score', 0)
                
                blank_info = ""
                if move.get('blank_assignments'):
                    blanks = []
                    for pos, letter in move['blank_assignments'].items():
                        blanks.append(f"{pos}='{letter}'")
                    blank_info = f" (Blanks: {', '.join(blanks)})"
                
                move_texts.append(f"Turn {turn}: Player {player}")
                move_texts.append(f"  {word} at {start} {direction}")
                move_texts.append(f"  Score: +{score}{blank_info}")
        
        ax.axis('off')
        ax.set_title('Move History')
        
        y_pos = 0.95
        for text in move_texts:
            ax.text(0.05, y_pos, text, transform=ax.transAxes, 
                   fontsize=8, verticalalignment='top')
            y_pos -= 0.03
    
    def visualize_score_progression(self, ax, move_history):
        """
        Visualize the score progression
        
        Args:
            ax: Matplotlib axes
            move_history: List of move data dictionaries
        """
        turns = []
        p1_scores = []
        p2_scores = []
        
        for i, move in enumerate(move_history):
            turns.append(i + 1)
            p1_scores.append(move.get('player_1_score', 0))
            p2_scores.append(move.get('player_2_score', 0))
        
        ax.plot(turns, p1_scores, 'b-', label='Player 1 (DQN)')
        ax.plot(turns, p2_scores, 'r-', label='Player 2 (Opponent)')
        
        ax.set_xlabel('Turn')
        ax.set_ylabel('Score')
        ax.set_title('Score Progression')
        ax.legend()
        
        ax.grid(True, linestyle='--', alpha=0.7)
    
    def analyze_blank_tile_usage(self, dqn_agent, num_games=50, verbose=True):
        """
        Analyze how the DQN agent uses blank tiles
        
        Args:
            dqn_agent: Trained DQN agent
            num_games: Number of games to analyze
            verbose: Whether to print progress
            
        Returns:
            dict: Analysis results
        """
        original_epsilon = dqn_agent.epsilon
        
        dqn_agent.epsilon = 0
        
        blank_usage_by_letter = {chr(ord('a') + i): 0 for i in range(26)}
        blank_usage_by_position = {}
        blank_points_saved = []
        
        for game_num in range(num_games):
            if verbose and game_num % 10 == 0:
                print(f"Analyzing game {game_num + 1}/{num_games}")
                
            game = self.game_api_class()
            game.initialize_game()
            
            while not game.game_over:
                if game.current_player == 1:
                    state = self.state_encoder.get_state_representation(game, 1)
                    
                    valid_moves = game.get_all_valid_moves(1)
                    
                    if not valid_moves:
                        game.pass_turn(1)
                        continue
                    
                    move = dqn_agent.act(state, valid_moves, self.reward_calculator, game, 1)
                    
                    if 'blank_assignments' in move and move['blank_assignments']:
                        for pos, letter in move['blank_assignments'].items():
                            blank_usage_by_letter[letter.lower()] += 1
                            
                            if isinstance(pos, str):
                                if pos not in blank_usage_by_position:
                                    blank_usage_by_position[pos] = 0
                                blank_usage_by_position[pos] += 1
                            
                            letter_value = game.letter_point_values.get(letter.lower(), 0)
                            blank_points_saved.append(letter_value)
                    
                    game.play_move(1, move['word'], move['start_space'], move['direction'],
                                  blank_assignments=move.get('blank_assignments', None))
                
                else:
                    valid_moves = game.get_all_valid_moves(2)
                    
                    if not valid_moves:
                        game.pass_turn(2)
                        continue
                    
                    move = random.choice(valid_moves)
                    
                    game.play_move(2, move['word'], move['start_space'], move['direction'],
                                  blank_assignments=move.get('blank_assignments', None))
        
        dqn_agent.epsilon = original_epsilon
        
        total_blanks_used = sum(blank_usage_by_letter.values())
        avg_letter_value_saved = np.mean(blank_points_saved) if blank_points_saved else 0
        
        if total_blanks_used > 0:
            for letter in blank_usage_by_letter:
                blank_usage_by_letter[letter] /= total_blanks_used
        
        if verbose:
            print("\nBlank Tile Usage Analysis:")
            print(f"Total blank tiles used: {total_blanks_used}")
            print(f"Average letter value saved: {avg_letter_value_saved:.2f}")
            
            print("\nBlank usage by letter (normalized):")
            for letter, freq in sorted(blank_usage_by_letter.items(), key=lambda x: x[1], reverse=True):
                letter_value = self.reward_calculator.letter_point_values.get(letter, 0)
                print(f"  {letter.upper()} ({letter_value} pts): {freq:.2f}")
        
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 2, 1)
        letters = sorted(blank_usage_by_letter.keys())
        freqs = [blank_usage_by_letter[l] for l in letters]
        letter_values = [self.reward_calculator.letter_point_values.get(l, 0) for l in letters]
        
        bars = plt.bar(letters, freqs)
        
        for i, bar in enumerate(bars):
            value = letter_values[i]
            if value <= 1:
                bar.set_color('lightblue')
            elif value <= 3:
                bar.set_color('skyblue')
            elif value <= 5:
                bar.set_color('royalblue')
            else:
                bar.set_color('navy')
        
        plt.title('Blank Tile Usage by Letter')
        plt.xlabel('Letter')
        plt.ylabel('Usage Frequency')
        
        plt.subplot(2, 2, 2)
        plt.hist(blank_points_saved, bins=range(11), alpha=0.7)
        plt.title('Points Saved by Using Blank Tiles')
        plt.xlabel('Letter Point Value')
        plt.ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig(f"{self.results_dir}/blank_tile_analysis.png")
        
        if verbose:
            plt.show()
        
        results = {
            'total_blanks_used': total_blanks_used,
            'avg_letter_value_saved': avg_letter_value_saved,
            'blank_usage_by_letter': blank_usage_by_letter,
            'blank_usage_by_position': blank_usage_by_position,
            'blank_points_saved': blank_points_saved
        }
        
        return results
    
    def compare_models(self, models, baseline_type="highest_score", num_games=20, verbose=True):
        """
        Compare multiple DQN models against a baseline
        
        Args:
            models: List of tuples (model_name, dqn_agent)
            baseline_type: Type of baseline opponent
            num_games: Number of games to play
            verbose: Whether to print progress
            
        Returns:
            dict: Comparison results
        """
        results = {
            'model_names': [],
            'win_rates': [],
            'avg_scores': [],
            'avg_score_diffs': [],
            'avg_game_lengths': []
        }
        
        for model_name, dqn_agent in models:
            if verbose:
                print(f"\nEvaluating model: {model_name}")
            
            eval_results = self.evaluate_against_baseline(
                dqn_agent,
                baseline_type=baseline_type,
                num_games=num_games,
                verbose=verbose
            )
            
            results['model_names'].append(model_name)
            results['win_rates'].append(eval_results['win_rate'])
            results['avg_scores'].append(eval_results['avg_dqn_score'])
            results['avg_score_diffs'].append(eval_results['avg_score_diff'])
            results['avg_game_lengths'].append(eval_results['avg_game_length'])
        
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 2, 1)
        plt.bar(results['model_names'], results['win_rates'])
        plt.title(f'Win Rates vs {baseline_type}')
        plt.ylabel('Win Rate')
        plt.ylim(0, 1)
        plt.xticks(rotation=45, ha='right')
        
        plt.subplot(2, 2, 2)
        plt.bar(results['model_names'], results['avg_scores'])
        plt.title('Average Scores')
        plt.ylabel('Score')
        plt.xticks(rotation=45, ha='right')
        
        plt.subplot(2, 2, 3)
        plt.bar(results['model_names'], results['avg_score_diffs'])
        plt.title('Average Score Differences')
        plt.ylabel('Score Difference')
        plt.xticks(rotation=45, ha='right')
        
        plt.subplot(2, 2, 4)
        plt.bar(results['model_names'], results['avg_game_lengths'])
        plt.title('Average Game Lengths')
        plt.ylabel('Moves')
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(f"{self.results_dir}/model_comparison.png")
        
        if verbose:
            plt.show()
        
        return results