import copy
import time
import numpy as np
import random
import os
# from collections import deque # Not directly used in this file anymore
# import matplotlib.pyplot as plt # If you use it for plotting stats later

import pandas as pd # For saving stats if you choose to

class ScrabbleSelfPlay:
    def __init__(self, game_api_class, state_encoder, reward_calculator, 
                 model_dir="./models", log_dir="./logs"):
        self.game_api_class = game_api_class # Store the class, not an instance
        self.state_encoder = state_encoder
        self.reward_calculator = reward_calculator
        
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        self.model_dir = model_dir
        self.log_dir = log_dir
        
        self.stats = {
            'episode_rewards_p1': [],
            'episode_rewards_p2': [],
            'episode_lengths': [],
            'win_counts_p1': 0,
            'win_counts_p2': 0,
            'draw_counts': 0,
            'avg_scores_p1': [],
            'avg_scores_p2': [],
            'blank_tile_usage_per_episode': [],
            'avg_loss_per_episode': [] # To track training loss
        }
    
    def train(self, agent1, agent2, num_episodes=1000, target_update_freq=10, # episodes
              batch_size=64, checkpoint_freq=100, verbose=True):

        start_time = time.time()
        total_steps = 0 
        
        for episode in range(num_episodes):
            game_instance = self.game_api_class() # Instantiate a new game for each episode
            game_instance.initialize_game()
            
            episode_reward_p1 = 0.0
            episode_reward_p2 = 0.0
            episode_steps = 0
            episode_blank_usage = 0
            episode_losses = []
            
            game_over_for_memory = False

            while not game_instance.game_over:
                current_player_id = game_instance.current_player
                current_agent = agent1 if current_player_id == 1 else agent2
                
                # State s
                state_s_dict = self.state_encoder.get_state_representation(game_instance, current_player_id)
                game_s_obj_for_act_and_reward = copy.deepcopy(game_instance) # State before current player's move
                
                valid_moves_s = game_s_obj_for_act_and_reward.get_all_valid_moves(current_player_id)

                if not valid_moves_s:
                    game_instance.pass_turn(current_player_id) # Modifies 'game_instance'
                    episode_steps +=1
                    # Check if pass ended the game to correctly set done_for_memory
                    game_over_for_memory = game_instance.game_over 
                    # We might want to store a "pass" experience with a specific reward.
                    # For now, skipping memory addition for pass if no valid moves.
                    if game_over_for_memory: break # End inner loop if pass ends game
                    continue 
                
                # Action a (chosen_move_dict) and its features (chosen_action_s_features_np)
                # Pass reward_calculator from self_play to agent.act, which then passes to _get_normalized_action_features
                chosen_move_dict, chosen_action_s_features_np = current_agent.act(
                    state_s_dict, valid_moves_s, game_s_obj_for_act_and_reward, 
                    current_player_id, episode
                )

                if chosen_move_dict is None: # Should be caught by valid_moves_s check, but as safeguard
                    game_instance.pass_turn(current_player_id)
                    episode_steps +=1
                    game_over_for_memory = game_instance.game_over
                    if game_over_for_memory: break
                    continue
                
                # Play move on the main 'game_instance' object to transition s -> s'
                play_success = game_instance.play_move(
                    current_player_id, chosen_move_dict['word'], chosen_move_dict['start_space'],
                    chosen_move_dict['direction'], blank_assignments=chosen_move_dict.get('blank_assignments')
                )

                if not play_success:
                    print(f"CRITICAL ERROR: Play move failed for P{current_player_id}. Move: {chosen_move_dict} in episode {episode}.")
                    # This indicates a severe issue. Consider adding a large negative reward and breaking.
                    # For now, just break the inner loop for this problematic episode.
                    break 

                if 'blank_assignments' in chosen_move_dict and chosen_move_dict['blank_assignments']:
                    episode_blank_usage += 1
                
                # State s' (game_instance is now s')
                game_over_for_memory = game_instance.game_over 
                
                # State representation of s' from the perspective of the current_player_id (who took action a)
                state_s_prime_dict = self.state_encoder.get_state_representation(game_instance, current_player_id)
                
                # For calculating max_a' Q(s', a'): features of valid moves from s' for the *next* player
                next_state_s_prime_valid_moves_features_list = []
                next_player_id_in_s_prime = game_instance.current_player # Player whose turn it is in s'

                if not game_over_for_memory:
                    valid_moves_s_prime = game_instance.get_all_valid_moves(next_player_id_in_s_prime)
                    
                    # Need 'unseen' vector for next_player_id_in_s_prime from state s'
                    s_prime_dict_for_next_player_perspective = self.state_encoder.get_state_representation(game_instance, next_player_id_in_s_prime)
                    unseen_for_s_prime_next_player = s_prime_dict_for_next_player_perspective['unseen']

                    for next_s_prime_move in valid_moves_s_prime:
                        # Use current_agent instance to call its _get_normalized_action_features.
                        # This assumes if agent1 != agent2, they share the same feature normalization logic.
                        # If they can have different normalization, then agent for next_player_id_in_s_prime should be used.
                        # For simplicity, using current_agent (or self.reward_calculator if agent does not have it directly).
                        # Best is to pass self.reward_calculator to _get_normalized_action_features as it's stateless.
                        # The agent's _get_normalized_action_features now takes reward_calculator_ref.
                        features = current_agent._get_normalized_action_features( 
                            next_s_prime_move, 
                            game_instance, # This is game_state_obj_for_eval (s')
                            next_player_id_in_s_prime, 
                            unseen_for_s_prime_next_player, 
                            self.reward_calculator # Pass self_play's reward_calculator
                        )
                        next_state_s_prime_valid_moves_features_list.append(features)
                
                # Reward r for action a from state s
                top_move_score_s = valid_moves_s[0]['score'] if valid_moves_s else 0 # Score of best move from s
                reward = self.reward_calculator.calculate_reward(
                    game_s_obj_for_act_and_reward, # game_state (s)
                    current_player_id,             # player_id who made move from s
                    chosen_move_dict,              # move (a)
                    top_move_score_s,              # top_move_score available at s
                    game_instance                  # next_game_state (s')
                )
                
                if current_player_id == 1: episode_reward_p1 += reward
                else: episode_reward_p2 += reward
                
                current_agent.remember(
                    state_s_dict, chosen_action_s_features_np, reward, state_s_prime_dict,
                    game_over_for_memory, next_state_s_prime_valid_moves_features_list, 
                    next_player_id_in_s_prime
                )
                
                loss = current_agent.replay(batch_size)
                if loss > 0: episode_losses.append(loss)
                
                episode_steps += 1
                total_steps += 1
            
            # --- End of Episode ---
            if episode_steps > 0 : # Avoid division by zero if episode ended prematurely
                self.stats['avg_loss_per_episode'].append(np.mean(episode_losses) if episode_losses else 0)
            self.stats['episode_rewards_p1'].append(episode_reward_p1)
            self.stats['episode_rewards_p2'].append(episode_reward_p2)
            self.stats['episode_lengths'].append(episode_steps)
            self.stats['blank_tile_usage_per_episode'].append(episode_blank_usage)
            self.stats['avg_scores_p1'].append(game_instance.player_1_score)
            self.stats['avg_scores_p2'].append(game_instance.player_2_score)

            if game_instance.player_1_score > game_instance.player_2_score:
                self.stats['win_counts_p1'] += 1
            elif game_instance.player_2_score > game_instance.player_1_score:
                self.stats['win_counts_p2'] += 1
            else: # Scores are equal
                if game_instance.game_over : self.stats['draw_counts'] += 1 # Only count draw if game truly finished

            if episode > 0 and episode % target_update_freq == 0:
                agent1.update_target_model()
                if agent1 is not agent2:
                    agent2.update_target_model()
            
            if episode > 0 and episode % checkpoint_freq == 0:
                agent1.save(f"{self.model_dir}/agent1_episode_{episode}.weights.h5")
                if agent1 is not agent2:
                    agent2.save(f"{self.model_dir}/agent2_episode_{episode}.weights.h5")
            
            if verbose and (episode + 1) % 5 == 0:
                progress_percent = (episode + 1) / num_episodes * 100
                elapsed_time = time.time() - start_time
                avg_steps_per_sec = total_steps / elapsed_time if elapsed_time > 0 else 0
                est_total_time = (num_episodes * (elapsed_time / (episode + 1))) if episode > 0 else 0
                est_remaining_time = est_total_time - elapsed_time if episode > 0 else 0
                
                print(f"\n--- Episode {episode+1}/{num_episodes} ({progress_percent:.1f}%) ---")
                print(f"Time: {elapsed_time/60:.1f}m | Left: {est_remaining_time/60:.1f}m | Steps/sec: {avg_steps_per_sec:.1f}")
                print(f"Epsilon: {agent1.epsilon:.4f} | Avg Loss (last ep): {self.stats['avg_loss_per_episode'][-1]:.4f if self.stats['avg_loss_per_episode'] else 'N/A'}")
                print(f"P1 Wins: {self.stats['win_counts_p1']} | P2 Wins: {self.stats['win_counts_p2']} | Draws: {self.stats['draw_counts']}")
                print(f"Avg Ep Length: {np.mean(self.stats['episode_lengths'][-20:]):.1f if self.stats['episode_lengths'] else 'N/A'} steps")
                bar_length = 30
                filled_length = int(bar_length * (episode + 1) // num_episodes)
                bar = '█' * filled_length + '░' * (bar_length - filled_length)
                print(f"[{bar}]")

        agent1.save(f"{self.model_dir}/agent1_final.weights.h5")
        if agent1 is not agent2:
            agent2.save(f"{self.model_dir}/agent2_final.weights.h5")
        
        return self.stats