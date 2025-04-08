import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import numpy as np
import random
from collections import deque

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
        
        outputs = layers.Dense(4, activation='linear')(x)
        
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
    
    def act(self, state, valid_moves, reward_calculator, game_state, player_id):
        """
        Choose action based on epsilon-greedy policy
        
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
            scored_moves = reward_calculator.estimate_move_quality(game_state, player_id, valid_moves)
            top_moves = scored_moves[:max(1, len(scored_moves)//4)]
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
        
        for move in valid_moves:
            score_value = move['score']
            
            tiles_used = 0
            if 'positions' in move:
                for pos in move['positions']:
                    if pos in game_state.is_tile_present:
                        if not game_state.is_tile_present[pos][0]:
                            tiles_used += 1
                    else:
                        tiles_used += 1
            else:
                tiles_used = len(move['word'])
            
            uses_blank = 'blank_assignments' in move and move['blank_assignments']
            blank_value = 0
            if uses_blank:
                for _, letter in move['blank_assignments'].items():
                    letter_value = reward_calculator.letter_point_values.get(letter.lower(), 0)
                    blank_value += letter_value / reward_calculator.max_letter_value
            
            norm_score = score_value / 50.0
            norm_tiles = tiles_used / 7.0
            norm_blank = blank_value / 1.0
            
            move_value = (
                move_features[0] * norm_score +
                move_features[1] * (1.0 - (len(valid_moves) / 100.0)) +
                move_features[2] * norm_tiles +
                move_features[3] * norm_blank
            )
            
            if move_value > best_value:
                best_value = move_value
                best_move = move
        
        return best_move
    
    def replay(self, batch_size):
        """
        Train model using experience replay
        
        Args:
            batch_size: Number of experiences to sample
        """
        if len(self.memory) < batch_size:
            return
        
        minibatch = random.sample(self.memory, batch_size)

        for state, action_idx, reward, next_state, done in minibatch:
            board_state = np.expand_dims(state['board'], axis=0)
            hand_state = np.expand_dims(state['hand'], axis=0)
            unseen_state = np.expand_dims(state['unseen'], axis=0)
            game_features = np.expand_dims(state['game_features'], axis=0)

            target_q = np.zeros_like(self.model.predict([board_state, hand_state, unseen_state, game_features], verbose=0)[0])

            if done:
                target_q[:] = reward
            else:
                next_board = np.expand_dims(next_state['board'], axis=0)
                next_hand = np.expand_dims(next_state['hand'], axis=0)
                next_unseen = np.expand_dims(next_state['unseen'], axis=0)
                next_game_features = np.expand_dims(next_state['game_features'], axis=0)

                next_q_values = self.target_model.predict(
                    [next_board, next_hand, next_unseen, next_game_features],
                    verbose=0
                )[0]
                target_q = reward + self.gamma * next_q_values

            current_q_values = self.model.predict(
                [board_state, hand_state, unseen_state, game_features],
                verbose=0
            )

            target_f = current_q_values[0]
            target_f[:] = target_q

            self.model.fit(
                [board_state, hand_state, unseen_state, game_features],
                np.expand_dims(target_f, axis=0),
                epochs=1, verbose=0
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