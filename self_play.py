import copy
import time
import numpy as np
import random
import os
from collections import deque
# import matplotlib.pyplot as plt

import pandas as pd

class ScrabbleSelfPlay:
    def __init__(self, game_api_class, state_encoder, reward_calculator, 
                 model_dir="./models", log_dir="./logs"):
        self.game_api_class = game_api_class
        self.state_encoder = state_encoder
        self.reward_calculator = reward_calculator
        
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        self.model_dir = model_dir
        self.log_dir = log_dir
        
        self.stats = {
            'episode_rewards': [],
            'episode_lengths': [],
            'win_rates': [],
            'avg_scores': [],
            'blank_tile_usage': []
        }
    
    def train(self, agent1, agent2, num_episodes=1000, target_update_freq=5,
              batch_size=64, checkpoint_freq=100, verbose=True):
        """
        Args:
            agent1: First DQN agent
            agent2: Second DQN agent
            num_episodes: # of episodes to train for
            target_update_freq: Frequency of network updates
            batch_size: Batch size for experience replay
            checkpoint_freq: Frequency of model checkpoints
            verbose: Whether to print progress
            
        Returns:
            dict: Training statistics
        """
        start_time = time.time()
        
        for episode in range(num_episodes):
            game = self.game_api_class()
            game.initialize_game()
            
            episode_reward1 = 0
            episode_reward2 = 0
            episode_length = 0
            blank_usage_count = 0
            
            while not game.game_over:
                if game.current_player == 1:
                    state = self.state_encoder.get_state_representation(game, 1)

                    # # debug
                    # print(f"Unseen tiles (Player 1's view): {state['unseen']}")
                    
                    valid_moves = game.get_all_valid_moves(1)

                    if len(valid_moves) > 0:
                        top_move_score1 = valid_moves[0]['score']
                    else:
                        top_move_score1 = 0
                    
                    if not valid_moves:
                        game.pass_turn(1)
                        continue
                    
                    move = agent1.act(state, valid_moves, self.reward_calculator, game, 1, episode)
                    
                    game_before = copy.deepcopy(game)
                    
                    game.play_move(1, move['word'], move['start_space'], move['direction'], 
                                  blank_assignments=move.get('blank_assignments', None))
                    
                    if 'blank_assignments' in move and move['blank_assignments']:
                        blank_usage_count += 1
                    
                    next_state = self.state_encoder.get_state_representation(game, 1)
                    
                    reward = self.reward_calculator.calculate_reward(game_before, 1, move, top_move_score1, game)
                    episode_reward1 += reward
                    
                    agent1.remember(state, 0, reward, next_state, game.game_over)
                    
                    agent1.replay(batch_size)

                    # # debug
                    # print(f"Player 1 hand after move: {game.player_1_hand}")
                
                elif game.current_player == 2:
                    state = self.state_encoder.get_state_representation(game, 2)

                    # # debug
                    # print(f"Unseen tiles (Player 2's view): {state['unseen']}")
                    
                    valid_moves = game.get_all_valid_moves(2)

                    if len(valid_moves) > 0:
                        top_move_score2 = valid_moves[0]['score']
                    else:
                        top_move_score2 = 0
                    
                    if not valid_moves:
                        game.pass_turn(2)
                        continue
                    
                    move = agent2.act(state, valid_moves, self.reward_calculator, game, 2, episode)
                    
                    game_before = copy.deepcopy(game)
                    
                    game.play_move(2, move['word'], move['start_space'], move['direction'],
                                  blank_assignments=move.get('blank_assignments', None))
                    
                    if 'blank_assignments' in move and move['blank_assignments']:
                        blank_usage_count += 1
                    
                    next_state = self.state_encoder.get_state_representation(game, 2)
                    
                    reward = self.reward_calculator.calculate_reward(game_before, 2, move, top_move_score2, game)
                    episode_reward2 += reward
                    
                    agent2.remember(state, 0, reward, next_state, game.game_over)
                    
                    agent2.replay(batch_size)

                    # # debug
                    # print(f"Player 2 hand after move: {game.player_2_hand}")
                
                episode_length += 1

                # # debug
                # print(f'Turn: {episode_length}')
            
            if episode % target_update_freq == 0:
                agent1.update_target_model()
                agent2.update_target_model()
            
            if episode % checkpoint_freq == 0:
                agent1.save(f"{self.model_dir}/agent1_episode_{episode}.weights.h5")
                agent2.save(f"{self.model_dir}/agent2_episode_{episode}.weights.h5")
            
            self.stats['episode_rewards'].append((episode_reward1, episode_reward2))
            self.stats['episode_lengths'].append(episode_length)
            self.stats['blank_tile_usage'].append(blank_usage_count)
            
            if verbose and episode % 5 == 0:
                progress_percent = (episode + 1) / num_episodes * 100
                elapsed_time = time.time() - start_time
                if episode > 0:
                     estimated_total_time = elapsed_time / episode * num_episodes
                     remaining_time = estimated_total_time - elapsed_time
                     remaining_str = f"{remaining_time/60:.1f} min"
                else:
                     remaining_str = "Calculating..."
                
                print(f"\n--- Episode {episode+1}/{num_episodes} ---")
                print(f"Progress: {progress_percent:.1f}%")
                print(f"Time elapsed: {elapsed_time/60:.1f} min | Estimated remaining: {remaining_str}")
                print(f"Current epsilon: {agent1.epsilon:.4f}")
                
                bar_length = 30
                filled_length = int(bar_length * (episode + 1) // num_episodes)
                bar = '█' * filled_length + '░' * (bar_length - filled_length)
                print(f"[{bar}] {progress_percent:.1f}%")

        
        agent1.save(f"{self.model_dir}/agent1_final.weights.h5")
        agent2.save(f"{self.model_dir}/agent2_final.weights.h5")

        # # debug
        # log_file_name = "move_evaluation_log_last30epsof200.csv"
        # combined_log_data = agent1.move_evaluation_log
        # if combined_log_data:
        #     try:
        #         print(f"Creating DataFrame from {len(combined_log_data)} log entries...")
        #         df_eval = pd.DataFrame(combined_log_data)
        #         save_path = os.path.join(self.log_dir, log_file_name)
        #         df_eval.to_csv(save_path, index=False)
        #         print(f"Move evaluation log saved to {save_path}")
        #     except Exception as e:
        #         print(f"Error creating or saving evaluation log DataFrame: {e}")
        # else:
        #     print("Move evaluation log is empty. No CSV file saved.")
        
        return self.stats