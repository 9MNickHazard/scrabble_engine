import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import numpy as np
import random
from collections import deque

import pandas as pd

class ScrabbleDQN:
    def __init__(self, state_size=(15, 15, 27), hand_size=27, unseen_size=27,
                 game_features_size=5, learning_rate=0.001, gamma=0.95, epsilon=1.0,
                 epsilon_min=0.01, epsilon_decay=0.995, memory_size=10000,
                 game_api_class_ref=None, state_encoder_ref=None, reward_calculator_ref=None):
        self.state_size = state_size
        self.hand_size = hand_size
        self.unseen_size = unseen_size
        self.game_features_size = game_features_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.game_api_class_ref = game_api_class_ref
        self.state_encoder_ref = state_encoder_ref
        self.reward_calculator_ref = reward_calculator_ref

        self.optimizer = optimizers.Adam(learning_rate=self.learning_rate)
        self.loss_fn = tf.keras.losses.MeanSquaredError()
        
        self.memory = deque(maxlen=memory_size)
        
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

        # debug
        self.move_evaluation_log = []
        self.log_eval_start_episode = 50
        
    def _build_model(self):
        """
        Build the neural network model for the DQN
        
        Returns:
            Keras model
        """
        board_input = layers.Input(shape=self.state_size, name='board_input')
        hand_input = layers.Input(shape=(self.hand_size,), name='hand_input')
        unseen_input = layers.Input(shape=(self.unseen_size,), name='unseen_input')
        game_features_input = layers.Input(shape=(self.game_features_size,), name='game_features_input')
        
        x_board = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(board_input)
        x_board = layers.BatchNormalization()(x_board)
        x_board = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x_board)
        x_board = layers.BatchNormalization()(x_board)
        x_board = layers.MaxPooling2D(pool_size=(2, 2))(x_board)
        x_board = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x_board)
        x_board = layers.BatchNormalization()(x_board)
        x_board = layers.Flatten()(x_board)
        x_board = layers.Dense(256, activation='relu')(x_board)
        
        x_hand = layers.Dense(64, activation='relu')(hand_input)
        x_hand = layers.BatchNormalization()(x_hand)
        x_hand = layers.Dense(64, activation='relu')(x_hand)
        
        x_unseen = layers.Dense(64, activation='relu')(unseen_input)
        x_unseen = layers.BatchNormalization()(x_unseen)
        x_unseen = layers.Dense(64, activation='relu')(x_unseen)
        
        x_game = layers.Dense(32, activation='relu')(game_features_input)
        x_game = layers.BatchNormalization()(x_game)
        x_game = layers.Dense(32, activation='relu')(x_game)
        
        combined = layers.Concatenate()([x_board, x_hand, x_unseen, x_game])
        
        x = layers.Dense(256, activation='relu')(combined)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(128, activation='relu')(x)
        
        outputs = layers.Dense(7, activation='linear')(x)
        
        model = models.Model(
            inputs=[board_input, hand_input, unseen_input, game_features_input],
            outputs=outputs
        )
        model.compile(optimizer=optimizers.Adam(learning_rate=self.learning_rate),
                     loss='mse')
        
        return model
    
    def update_target_model(self):
        """
        Update target model weights with current model weights
        """
        self.target_model.set_weights(self.model.get_weights())
    
    def remember(self, state, action, reward, next_state, done, next_state_valid_moves_features_list, next_state_player_id):
        """
        Store experience in memory
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode is finished
            next_state_valid_moves_features_list: List of feature arrays for all valid moves in next_state
            next_state_player_id: Player ID whose turn it is in next_state
        """
        self.memory.append((state, action, reward, next_state, done, next_state_valid_moves_features_list, next_state_player_id))
    
    def act(self, state_dict_s, valid_moves_s, current_game_state_obj_s, player_id_s, episode_num=0):
        """
        Choose action based on epsilon-greedy policy.
        Args:
            state_dict_s: Dictionary representing the current state (output of state_encoder).
            valid_moves_s: List of valid move dictionaries for the current state.
            current_game_state_obj_s: The Scrabble_Game object representing the current state s.
            player_id_s: ID of the player making the move.
            episode_num: Current episode number (for logging).
        Returns:
            A tuple (chosen_move_dict, chosen_action_norm_features_np_array),
            or (None, None) if no valid moves.
        """
        if not valid_moves_s:
            return None, None

        # Epsilon-greedy exploration
        if np.random.rand() <= self.epsilon:
            # Use estimate_move_quality for exploration
            # self.reward_calculator_ref should be an instance of ScrabbleDQNReward
            scored_moves_for_exploration = self.reward_calculator_ref.estimate_move_quality(
                current_game_state_obj_s, valid_moves_s
            )

            if not scored_moves_for_exploration:
                # Fallback if estimate_move_quality returns empty (should not happen if valid_moves_s is not empty)
                chosen_move_dict = random.choice(valid_moves_s)
            else:
                # Pick randomly from the top 20% (or at least 1) of heuristically good moves
                top_moves_count = max(1, len(scored_moves_for_exploration) // 5)
                chosen_move_dict = random.choice(scored_moves_for_exploration[:top_moves_count])['move']
            
            # Calculate the normalized features for this chosen exploratory move
            chosen_action_norm_features = self._get_normalized_action_features(
                chosen_move_dict,
                current_game_state_obj_s,
                player_id_s,
                state_dict_s['unseen'], # 'unseen' vector from the current state dictionary
                self.reward_calculator_ref # Pass the reward calculator instance
            )
            return chosen_move_dict, chosen_action_norm_features

        # Exploitation: Use the DQN model
        board_state_tf = tf.expand_dims(tf.convert_to_tensor(state_dict_s['board'], dtype=tf.float32), 0)
        hand_state_tf = tf.expand_dims(tf.convert_to_tensor(state_dict_s['hand'], dtype=tf.float32), 0)
        unseen_state_tf = tf.expand_dims(tf.convert_to_tensor(state_dict_s['unseen'], dtype=tf.float32), 0)
        game_features_tf = tf.expand_dims(tf.convert_to_tensor(state_dict_s['game_features'], dtype=tf.float32), 0)

        # Get the 7 output weights from the model for the current state s
        # These weights are W_0(s) to W_5(s) for features, and W_6(s) for bias.
        state_dependent_weights = self.model(
            [board_state_tf, hand_state_tf, unseen_state_tf, game_features_tf], training=False
        )[0] # Get the first (and only) batch item, result is a Tensor of shape (7,)

        best_move_dict = None
        best_q_value = -float('inf')
        best_action_norm_features = None

        for move_s_a in valid_moves_s:
            # Calculate the 6 normalized features f_k(a) for the current action a in state s
            action_norm_features_np = self._get_normalized_action_features(
                move_s_a,
                current_game_state_obj_s, # Game state before this move
                player_id_s,
                state_dict_s['unseen'],   # Unseen tiles from player's perspective in state s
                self.reward_calculator_ref
            )
            action_norm_features_tf = tf.convert_to_tensor(action_norm_features_np, dtype=tf.float32) # Shape (6,)

            # Calculate Q(s,a) = sum_{k=0 to 5} ( W_k(s) * f_k(a) ) + Bias(s)
            # state_dependent_weights[:6] are W_0(s) to W_5(s)
            # state_dependent_weights[6] is Bias(s)
            q_value = tf.reduce_sum(state_dependent_weights[:6] * action_norm_features_tf) + state_dependent_weights[6]

            # Logging (using your existing structure, ensure original values are correctly sourced if needed)
            if episode_num >= self.log_eval_start_episode: # Ensure self.log_eval_start_episode is defined
                # For detailed logging, re-calculate original feature values if _get_normalized_action_features only returns normalized.
                # Or modify _get_normalized_action_features to return both.
                # Here, we'll log the normalized features and weights.
                log_entry = {
                    'move_number': episode_num,
                    'player_id': player_id_s,
                    'move_word': move_s_a.get('word', None),
                    'start_space': move_s_a.get('start_space', None),
                    'direction': move_s_a.get('direction', None),
                    # Placeholder for original values - you'd need to re-calculate or pass them through
                    'original_score': move_s_a.get('score', 0.0), 
                    # 'original_tiles_used': ...,
                    # 'original_bingo_potential': ...,
                    # ... etc.
                    'norm_score': action_norm_features_np[0],
                    'norm_tiles': action_norm_features_np[1],
                    'norm_bingo_potential': action_norm_features_np[2],
                    'norm_setup': action_norm_features_np[3],
                    'norm_defensive': action_norm_features_np[4],
                    'norm_endgame': action_norm_features_np[5],
                    'calculated_move_value': q_value.numpy(),
                    'weight_score': state_dependent_weights[0].numpy(),
                    'weight_tiles_used': state_dependent_weights[1].numpy(),
                    'weight_bingo_potential': state_dependent_weights[2].numpy(),
                    'weight_setup_value': state_dependent_weights[3].numpy(),
                    'weight_defensive_value': state_dependent_weights[4].numpy(),
                    'weight_endgame': state_dependent_weights[5].numpy(),
                    'weight_bias': state_dependent_weights[6].numpy()
                }
                self.move_evaluation_log.append(log_entry)

            if q_value > best_q_value:
                best_q_value = q_value
                best_move_dict = move_s_a
                best_action_norm_features = action_norm_features_np # Store the numpy array

        return best_move_dict, best_action_norm_features

    def _calculate_q_value_for_action_batch(self, model_to_use, state_inputs_batch_tf_list, action_features_batch_tf):
        """
        Calculates Q(s,a) for a batch of states and corresponding action features.
        Args:
            model_to_use: The Keras model (self.model or self.target_model).
            state_inputs_batch_tf_list: A list of batched state component Tensors:
                                     [board_b_tf, hand_b_tf, unseen_b_tf, game_feat_b_tf].
            action_features_batch_tf: A batch Tensor of normalized action features (shape: batch_size, num_action_features=6).
        Returns:
            A Tensor of Q-values (shape: batch_size,).
        """
        # Get state-dependent weights W(s) for the batch of states
        state_weights_batch_tf = model_to_use(state_inputs_batch_tf_list, training=False) # Shape: (batch_size, 7)
        
        # Q(s,a) = sum_{k=0 to 5} ( W_k(s) * f_k(a) ) + Bias(s)
        # action_features_batch_tf has shape (batch_size, 6)
        # state_weights_batch_tf[:, :6] has shape (batch_size, 6) for feature weights
        # state_weights_batch_tf[:, 6] has shape (batch_size,) for bias (W_6(s))
        
        q_values_tf = tf.reduce_sum(state_weights_batch_tf[:, :6] * action_features_batch_tf, axis=1) + state_weights_batch_tf[:, 6]
        return q_values_tf

    @tf.function
    def _train_step_v2(self, current_states_inputs_b_tuple_tf, chosen_actions_features_b_tf, target_q_values_b_tf):
        """
        Performs a single training step using GradientTape.
        Args:
            current_states_inputs_b_tuple_tf: A tuple of batched current state Tensors (board, hand, unseen, game_features).
            chosen_actions_features_b_tf: Batch Tensor of features for the actions taken (batch_size, 6).
            target_q_values_b_tf: Batch Tensor of precomputed target Q-values (batch_size,).
        Returns:
            The loss value for this training step (Tensor).
        """
        with tf.GradientTape() as tape:
            # Calculate Q_current(s, a_chosen) using the main model
            # Ensure current_states_inputs_b_tuple_tf is passed as a list to the model
            predicted_q_values_for_chosen_actions_tf = self._calculate_q_value_for_action_batch(
                self.model, list(current_states_inputs_b_tuple_tf), chosen_actions_features_b_tf
            )
            
            # Calculate loss: MSE(target_Q, predicted_Q)
            loss = self.loss_fn(target_q_values_b_tf, predicted_q_values_for_chosen_actions_tf)
        
        # Compute gradients and apply them
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        return loss

    def replay(self, batch_size):
        """
        Train model using experience replay with custom training step and correct target Q-value calculation.
        """
        if len(self.memory) < batch_size:
            return 0.0 # Return 0 loss if not enough samples

        minibatch = random.sample(self.memory, batch_size)

        # Unpack experiences from the minibatch
        states_s_dicts_b = [t[0] for t in minibatch]
        chosen_actions_s_features_b_np = np.array([t[1] for t in minibatch]) # (batch_size, 6)
        rewards_b_np = np.array([t[2] for t in minibatch], dtype=np.float32) # (batch_size,)
        next_states_s_prime_dicts_b = [t[3] for t in minibatch]
        dones_b_np = np.array([t[4] for t in minibatch]) # (batch_size,)
        all_next_s_prime_actions_features_lists_b = [t[5] for t in minibatch] # List of lists of np arrays
        # next_s_prime_player_ids_b = [t[6] for t in minibatch] # Not directly used here as features are precomputed

        # --- Calculate target Q-values: y_i = r_i + gamma * max_a' Q_target(s'_i, a') ---

        # 1. Prepare batch of next_state (s') inputs for the target_model
        next_s_prime_board_b_np = np.array([ns_dict['board'] for ns_dict in next_states_s_prime_dicts_b])
        next_s_prime_hand_b_np = np.array([ns_dict['hand'] for ns_dict in next_states_s_prime_dicts_b])
        next_s_prime_unseen_b_np = np.array([ns_dict['unseen'] for ns_dict in next_states_s_prime_dicts_b])
        next_s_prime_game_features_b_np = np.array([ns_dict['game_features'] for ns_dict in next_states_s_prime_dicts_b])
        
        next_s_prime_target_model_inputs_tf_list = [
            tf.convert_to_tensor(next_s_prime_board_b_np, dtype=tf.float32),
            tf.convert_to_tensor(next_s_prime_hand_b_np, dtype=tf.float32),
            tf.convert_to_tensor(next_s_prime_unseen_b_np, dtype=tf.float32),
            tf.convert_to_tensor(next_s_prime_game_features_b_np, dtype=tf.float32)
        ]
        # 2. Get W_target(s') (weights) for all next states s' in the batch
        target_next_s_prime_weights_batch_tf = self.target_model(
            next_s_prime_target_model_inputs_tf_list, training=False
        ) # Shape: (batch_size, 7)

        # 3. Calculate max_a' Q_target(s'_i, a') for each sample i in the batch
        max_next_q_values_np = np.zeros(batch_size, dtype=np.float32)
        for i in range(batch_size):
            if dones_b_np[i] or not all_next_s_prime_actions_features_lists_b[i]:
                # If s'_i is terminal or no valid actions from s'_i, future value is 0
                max_next_q_values_np[i] = 0.0
            else:
                # W_target(s'_i) for the i-th sample
                s_prime_i_target_weights_tf = target_next_s_prime_weights_batch_tf[i] # Shape: (7,)
                
                # Features f_k(a') for all valid actions a' in s'_i
                # This is a list of np arrays; convert to a 2D TF tensor
                features_for_s_prime_i_actions_tf = tf.convert_to_tensor(
                    np.array(all_next_s_prime_actions_features_lists_b[i]), dtype=tf.float32
                ) # Shape: (num_valid_moves_in_s_prime_i, 6)
                
                # Q_target(s'_i, a') = sum_{k=0 to 5} ( W_target_k(s'_i) * f_k(a') ) + Bias_target(s'_i)
                q_values_for_s_prime_i_actions_tf = tf.reduce_sum(
                    s_prime_i_target_weights_tf[:6] * features_for_s_prime_i_actions_tf, axis=1
                ) + s_prime_i_target_weights_tf[6]
                
                max_next_q_values_np[i] = tf.reduce_max(q_values_for_s_prime_i_actions_tf).numpy()

        # 4. Compute the final target Q-values (y_i)
        target_q_values_b_np = rewards_b_np + self.gamma * max_next_q_values_np * (1.0 - dones_b_np.astype(np.float32))
        
        # --- Perform training step ---
        # Prepare batch of current_state (s) inputs for the main model
        current_s_board_b_tf = tf.convert_to_tensor(np.array([s_dict['board'] for s_dict in states_s_dicts_b]), dtype=tf.float32)
        current_s_hand_b_tf = tf.convert_to_tensor(np.array([s_dict['hand'] for s_dict in states_s_dicts_b]), dtype=tf.float32)
        current_s_unseen_b_tf = tf.convert_to_tensor(np.array([s_dict['unseen'] for s_dict in states_s_dicts_b]), dtype=tf.float32)
        current_s_game_features_b_tf = tf.convert_to_tensor(np.array([s_dict['game_features'] for s_dict in states_s_dicts_b]), dtype=tf.float32)
        
        current_s_inputs_b_tuple_tf = (
            current_s_board_b_tf, current_s_hand_b_tf, 
            current_s_unseen_b_tf, current_s_game_features_b_tf
        )
        # Convert chosen action features and target Q-values to Tensors
        chosen_actions_s_features_b_tf = tf.convert_to_tensor(chosen_actions_s_features_b_np, dtype=tf.float32)
        target_q_values_b_tf = tf.convert_to_tensor(target_q_values_b_np, dtype=tf.float32)
        
        # Call the @tf.function decorated training step
        loss_value_tf = self._train_step_v2(
            current_s_inputs_b_tuple_tf,
            chosen_actions_s_features_b_tf,
            target_q_values_b_tf
        )

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return loss_value_tf.numpy()
    
    def load(self, name):
        """
        Args:
            name: Path to weights file
        """
        self.model.load_weights(name)
        self.update_target_model()
    
    def save(self, name):
        """
        Args:
            name: Path to save weights
        """
        self.model.save_weights(name)
    
    def normalize_strategic_value(value):
        """
        Normalizes any strategic value to range [0,1] using a sigmoid function.
        
        Args:
            value: The raw strategic value
        
        Returns:
            float: Normalized value between 0 and 1
        """
        return 1.0 / (1.0 + np.exp(-value))

    def clear_move_evaluation_log(self):
        self.move_evaluation_log = []
    
    def _get_normalized_action_features(self, move, game_state_obj_for_eval, player_id_for_eval, state_unseen_vector_for_eval, reward_calculator_ref_for_eval):
        """
        Helper to calculate normalized features for a single move.
        Args:
            move: The move dictionary. (Should contain 'score', 'positions', 'word', and optionally 'blank_assignments')
            game_state_obj_for_eval: The Scrabble_Game object representing the state for which to evaluate this move.
            player_id_for_eval: The ID of the player making the move in game_state_obj_for_eval.
            state_unseen_vector_for_eval: The 'unseen' tile vector for player_id_for_eval in game_state_obj_for_eval.
            reward_calculator_ref_for_eval: Reference to the reward calculator instance.
        """
        score_value = move.get('score', 0.0)
        if score_value is None:  # Handle cases where score might be None
            score_value = 0.0

        current_hand = game_state_obj_for_eval.player_1_hand if player_id_for_eval == 1 else game_state_obj_for_eval.player_2_hand
        
        tiles_used_from_hand = 0
        # Create a mutable copy of the hand to simulate tile removal for bingo potential calculation
        hand_after_move_simulated = current_hand.copy() 
        # For calculating defensive value, we need the positions where new tiles are placed
        newly_placed_tile_positions = [] 
        
        move_blank_assignments = move.get('blank_assignments', {})

        if 'positions' in move and move['positions'] and 'word' in move:
            word_played = move['word']
            for i, pos_str in enumerate(move['positions']):
                # Determine if this position involves placing a new tile from hand
                is_new_tile_placement = False
                if pos_str in game_state_obj_for_eval.is_tile_present:
                    # Square exists, check if it's empty
                    if not game_state_obj_for_eval.is_tile_present[pos_str][0]:
                        is_new_tile_placement = True
                # This elif handles cases where pos_str might not be in is_tile_present,
                # which implies it's an empty square not yet explicitly tracked (less likely with full init).
                # Or, if is_tile_present was sparse and only held occupied tiles.
                # Given full init, this is a fallback.
                elif pos_str not in game_state_obj_for_eval.is_tile_present:
                     is_new_tile_placement = True 

                if is_new_tile_placement:
                    tiles_used_from_hand += 1
                    newly_placed_tile_positions.append(pos_str)

                    # Simulate removing the tile from hand_after_move_simulated
                    is_blank_used_at_this_pos = pos_str in move_blank_assignments
                    
                    if is_blank_used_at_this_pos:
                        if '_' in hand_after_move_simulated:
                            hand_after_move_simulated.remove('_')
                        # else: Error - blank indicated but not in hand. Validation should prevent.
                    else:
                        # If not a blank, it's a specific letter tile from hand.
                        # The letter played at this segment of the word is word_played[i]
                        if i < len(word_played): # Ensure index is valid
                            letter_on_tile = word_played[i].lower()
                            if letter_on_tile in hand_after_move_simulated:
                                hand_after_move_simulated.remove(letter_on_tile)
                            # else: Error - specific letter needed but not in hand. Validation should prevent.
                        # else: Error - move['positions'] and move['word'] lengths mismatch. Validation should prevent.
        else:
            # This case implies a move type that doesn't use 'positions' or 'word' (e.g., pass/exchange if featurized)
            # or an invalid play move structure. For standard play moves, this block shouldn't be hit.
            pass # tiles_used_from_hand remains 0

        # Calculate features based on the move and simulated hand state
        tiles_in_bag = sum(game_state_obj_for_eval.letter_counts.values())
        
        # This uses tiles_used_from_hand, which is the count of tiles actually played from the rack
        hand_size_after_play_before_draw = len(current_hand) - tiles_used_from_hand
        tiles_needed_to_refill_rack = max(0, 7 - hand_size_after_play_before_draw)

        bingo_potential_of_leave = 0.0
        # Only calculate leave potential if it's not a bingo and tiles can be drawn to refill
        # Using score_value < 50 as a heuristic for non-bingo (bingos usually score >= 50)
        if score_value < 56 and (tiles_needed_to_refill_rack == 0 or tiles_in_bag >= tiles_needed_to_refill_rack):
            bingo_potential_of_leave = reward_calculator_ref_for_eval._evaluate_bingo_potential(
                hand_after_move_simulated, # Use the simulated hand *after* tiles are played
                state_unseen_vector_for_eval
            )
        elif tiles_used_from_hand == 7: # Actual bingo played
            # For a bingo, the "leave" is effectively 0 tiles before draw, so potential is for the new hand.
            # We can assign a fixed "good leave" value or a more neutral one, as the bingo score itself is dominant.
            # Or, calculate based on an empty hand_after_move_simulated and full draw if bag allows.
            # For simplicity, assign a representative value indicating a good outcome.
            bingo_potential_of_leave = 6.0 # Tunable: Represents a good state after a bingo.

        setup_value = reward_calculator_ref_for_eval.calculate_setup_value(
            game_state_obj_for_eval, move, player_id_for_eval
        )
        # Defensive value uses the positions where new tiles were actually placed
        defensive_value = reward_calculator_ref_for_eval.calculate_defensive_value(
            game_state_obj_for_eval, move, newly_placed_tile_positions, state_unseen_vector_for_eval
        )
        
        endgame_value = 0.0
        # The state_dict for endgame_value should correspond to game_state_obj_for_eval
        # It primarily uses 'unseen' from the encoded_state argument.
        temp_state_dict_for_endgame_calc = {'unseen': state_unseen_vector_for_eval} 
        if tiles_in_bag <= 7: # Condition for endgame considerations
             endgame_value = reward_calculator_ref_for_eval.calculate_endgame_strategy_reward(
                 game_state_obj_for_eval, player_id_for_eval, move, 
                 None, # next_game_state is not available during 'act' phase for this evaluation
                 temp_state_dict_for_endgame_calc
             )

        # --- NORMALIZATION (Adjust these denominators based on observed ranges and desired emphasis) ---
        norm_score = score_value / 25.0
        norm_tiles = tiles_used_from_hand / 7.0 # Based on tiles played from hand
        norm_bingo_potential = bingo_potential_of_leave / 18.0
        norm_setup = setup_value / 8.0
        norm_defensive = defensive_value / 10.0
        norm_endgame = endgame_value / 20.0
        # --- END NORMALIZATION ---

        # Clip normalized values to a reasonable range (e.g., max of 3 or 4) to prevent extreme outliers
        # from dominating if a chosen denominator is occasionally too small for an exceptional raw value.
        clip_max = 4.0
        action_features_normalized = np.array([
            min(norm_score, clip_max),
            min(norm_tiles, clip_max), # tiles_used_from_hand / 7.0 should naturally be <= 1.0
            min(norm_bingo_potential, clip_max),
            min(norm_setup, clip_max),
            min(norm_defensive, clip_max),
            min(norm_endgame, clip_max)
        ], dtype=np.float32)
        
        return action_features_normalized