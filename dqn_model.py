import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import numpy as np
import random
from collections import deque

import pandas as pd

class ScrabbleDQN:
    def __init__(self, state_size=(15, 15, 27), hand_size=27, unseen_size=27, 
                 game_features_size=5, learning_rate=0.001, gamma=0.95, epsilon=1.0,
                 epsilon_min=0.01, epsilon_decay=0.995, memory_size=10000):
        self.state_size = state_size
        self.hand_size = hand_size
        self.unseen_size = unseen_size
        self.game_features_size = game_features_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        
        self.memory = deque(maxlen=memory_size)
        
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

        # # debug
        # self.move_evaluation_log = []
        # self.log_eval_start_episode = 170
        
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
    
    def remember(self, state, action, reward, next_state, done):
        """
        Store experience in memory
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode is finished
        """
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state, valid_moves, reward_calculator, game_state, player_id, episode_num=0):
        """
        Choose action on epsilon-greedy policy
        
        Args:
            state: Current state
            valid_moves: List of valid moves
            reward_calculator: Reward calculator instance
            game_state: Current game state
            player_id: ID of the player making the move
            
        Returns:
            Selected move
        """
        if not valid_moves:
            return None
        
        if np.random.rand() <= self.epsilon:
            scored_moves = reward_calculator.estimate_move_quality(game_state, valid_moves)
            # # debug
            # print(f"Move qualities:\n{scored_moves}")

            top_moves = scored_moves[:max(1, len(scored_moves)//5)]
            return random.choice(top_moves)['move']
        
        board_state = np.expand_dims(state['board'], axis=0)
        hand_state = np.expand_dims(state['hand'], axis=0)
        unseen_state = np.expand_dims(state['unseen'], axis=0)
        game_features = np.expand_dims(state['game_features'], axis=0)

        
        move_features = self.model.predict(
            [board_state, hand_state, unseen_state, game_features],
            verbose=0
        )[0]
        
        best_move = None
        best_value = float('-inf')

        current_hand = game_state.player_1_hand if player_id == 1 else game_state.player_2_hand
        
        for move in valid_moves:
            score_value = move['score']
            
            tiles_used = 0
            hand_after_move = current_hand.copy()
            played_positions = []

            if 'positions' in move:
                for pos in move['positions']:
                    if pos in game_state.is_tile_present:
                        if not game_state.is_tile_present[pos][0]:
                            for i, placed_pos in enumerate(move['positions']):
                                if placed_pos == pos:
                                    letter = move['word'][i].lower()
                                    tiles_used += 1
                                    played_positions.append(pos)
                                    if letter in hand_after_move:
                                        hand_after_move.remove(letter)
                                    elif '_' in hand_after_move:
                                        hand_after_move.remove('_')
                                    break
                    else:
                        for i, placed_pos in enumerate(move['positions']):
                            if placed_pos == pos:
                                letter = move['word'][i].lower()
                                tiles_used += 1
                                played_positions.append(pos)
                                if letter in hand_after_move:
                                    hand_after_move.remove(letter)
                                elif '_' in hand_after_move:
                                    hand_after_move.remove('_')
                                break
            else:
                tiles_used = len(move['word'])
            
            bingo_potential = reward_calculator._evaluate_bingo_potential(hand_after_move, state['unseen'])
            setup_value = reward_calculator.calculate_setup_value(game_state, move, player_id)
            defensive_value = reward_calculator.calculate_defensive_value(game_state, move, played_positions, state['unseen'])
            endgame_value = reward_calculator.calculate_endgame_strategy_reward(game_state, move, player_id, None, state)
            
            norm_score = score_value / 20.0
            norm_tiles = tiles_used / 7.0
            norm_bingo_potential = bingo_potential / 6.0
            norm_setup = setup_value / 4.0
            norm_defensive = defensive_value / 8.0
            norm_endgame = endgame_value / 8.0

            
            move_value = (
                move_features[0] * norm_score +
                move_features[1] * norm_tiles +
                move_features[2] * norm_bingo_potential +
                move_features[3] * norm_setup + 
                move_features[4] * norm_defensive +
                move_features[5] * norm_endgame +
                move_features[6]
            )

            # # debug
            # print(f"norm_score: {norm_score} | original: {score_value}\nnorm_tiles: {norm_tiles} | original: {tiles_used}\nnorm_bingo_potential: {norm_bingo_potential} | original: {bingo_potential}\nnorm_setup: {norm_setup} | original: {setup_value}\nnorm_defensive: {norm_defensive} | original: {defensive_value}\nmove_value: {move_value}\nmove_features[0]: {move_features[0]}\nmove_features[1]: {move_features[1]}\nmove_features[2]: {move_features[2]}\nmove_features[3]: {move_features[3]}\nmove_features[4]: {move_features[4]}\nmove_features[5]: {move_features[5]}\nmove_features[6]: {move_features[6]}\n")
            
            # # debug
            # if episode_num >= self.log_eval_start_episode:
            #     log_entry = {
            #         'episode': episode_num,
            #         'player_id': player_id,
            #         'move_word': move.get('word', None),
            #         'original_score': score_value,
            #         'original_tiles_used': tiles_used,
            #         'original_bingo_potential': bingo_potential,
            #         'original_setup_value': setup_value,
            #         'original_defensive_value': defensive_value,
            #         'original_endgame_value': endgame_value,
            #         'calculated_move_value': move_value,
            #         'weight_score': move_features[0],
            #         'weight_tiles_used': move_features[1],
            #         'weight_bingo_potential': move_features[2],
            #         'weight_setup_value': move_features[3],
            #         'weight_defensive_value': move_features[4],
            #         'weight_endgame': move_features[5],
            #         'weight_bias': move_features[6]
            #     }
            #     self.move_evaluation_log.append(log_entry)

            if move_value > best_value:
                best_value = move_value
                best_move = move
        
        return best_move


    def replay(self, batch_size):
        """
        Train model using experience replay with efficient batching
        
        Args:
            batch_size: Number of experiences to sample
        """
        if len(self.memory) < batch_size:
            return
        
        minibatch = random.sample(self.memory, batch_size)
        
        board_states = np.array([state['board'] for state, _, _, _, _ in minibatch])
        hand_states = np.array([state['hand'] for state, _, _, _, _ in minibatch])
        unseen_states = np.array([state['unseen'] for state, _, _, _, _ in minibatch])
        game_features = np.array([state['game_features'] for state, _, _, _, _ in minibatch])
        
        rewards = np.array([reward for _, _, reward, _, _ in minibatch])
        dones = np.array([done for _, _, _, _, done in minibatch])
        
        current_q_values = self.model.predict(
            [board_states, hand_states, unseen_states, game_features],
            verbose=0
        )
        
        target_f = current_q_values.copy()
        
        for i, idx in enumerate(np.where(dones)[0]):
            target_f[idx] = rewards[idx]
        
        if not np.all(dones):
            next_indices = np.where(~dones)[0]
            next_board = np.array([minibatch[i][3]['board'] for i in next_indices])
            next_hand = np.array([minibatch[i][3]['hand'] for i in next_indices])
            next_unseen = np.array([minibatch[i][3]['unseen'] for i in next_indices])
            next_game_features = np.array([minibatch[i][3]['game_features'] for i in next_indices])
            
            next_q_values = self.target_model.predict(
                [next_board, next_hand, next_unseen, next_game_features],
                verbose=0
            )
            
            for idx, next_idx in enumerate(next_indices):
                target_f[next_idx] = rewards[next_idx] + self.gamma * next_q_values[idx]
        
        self.model.fit(
            [board_states, hand_states, unseen_states, game_features],
            target_f,
            epochs=1, 
            verbose=0,
            batch_size=batch_size
        )
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
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