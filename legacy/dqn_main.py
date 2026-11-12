import os
import argparse
import tensorflow as tf
import numpy as np
import random
import copy # Keep copy if used elsewhere, not directly in this version of main
from game_api import Scrabble_Game
from dqn_state import ScrabbleDQNState
from dqn_reward_functions import ScrabbleDQNReward
from dqn_model import ScrabbleDQN
from self_play import ScrabbleSelfPlay

def set_seeds(seed=85):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def train_model(args):
    print("Starting training...")
    
    set_seeds(args.seed)
    
    state_encoder = ScrabbleDQNState()
    reward_calculator = ScrabbleDQNReward(
        win_reward=args.win_reward, 
        loss_penalty=args.loss_penalty
    )
    game_api_class_for_dqn = Scrabble_Game # Pass the class itself

    agent1 = ScrabbleDQN(
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        epsilon=args.epsilon,
        epsilon_min=args.epsilon_min,
        epsilon_decay=args.epsilon_decay,
        memory_size=args.memory_size,
        game_api_class_ref=game_api_class_for_dqn, # ADDED
        state_encoder_ref=state_encoder,           # ADDED
        reward_calculator_ref=reward_calculator    # ADDED
    )
    
    if args.separate_agents:
        agent2 = ScrabbleDQN(
            learning_rate=args.learning_rate,
            gamma=args.gamma,
            epsilon=args.epsilon,
            epsilon_min=args.epsilon_min,
            epsilon_decay=args.epsilon_decay,
            memory_size=args.memory_size,
            game_api_class_ref=game_api_class_for_dqn, # ADDED
            state_encoder_ref=state_encoder,           # ADDED
            reward_calculator_ref=reward_calculator    # ADDED
        )
    else:
        agent2 = agent1
    
    trainer = ScrabbleSelfPlay(
        game_api_class=Scrabble_Game, # Pass the class for game instantiation
        state_encoder=state_encoder,
        reward_calculator=reward_calculator,
        model_dir=args.model_dir,
        log_dir=args.log_dir
    )
    
    stats = trainer.train(
        agent1=agent1,
        agent2=agent2,
        num_episodes=args.num_episodes,
        target_update_freq=args.target_update_freq,
        batch_size=args.batch_size,
        checkpoint_freq=args.checkpoint_freq,
        verbose=True
    )
    
    print("Training completed!")
    # Save the logs from agent1 (or combine if separate and desired)
    if agent1.move_evaluation_log:
        try:
            log_df = pd.DataFrame(agent1.move_evaluation_log)
            log_file_path = os.path.join(args.log_dir, "move_evaluation_log_final.csv")
            log_df.to_csv(log_file_path, index=False)
            print(f"Move evaluation log saved to {log_file_path}")
        except Exception as e:
            print(f"Error saving move evaluation log: {e}")

    return agent1, stats # agent1 now contains the trained model (and agent2 if separate)


def main():
    # print("Main: Loading dictionary for move generator...") # If you have separate dict loading
    parser = argparse.ArgumentParser(description='Scrabble DQN')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    train_parser = subparsers.add_parser('train', help='Train the DQN model')
    train_parser.add_argument('--num-episodes', type=int, default=1000, help='Number of episodes to train for')
    train_parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
    train_parser.add_argument('--gamma', type=float, default=0.95, help='Discount factor')
    train_parser.add_argument('--epsilon', type=float, default=1.0, help='Initial exploration rate')
    train_parser.add_argument('--epsilon-min', type=float, default=0.01, help='Minimum exploration rate')
    train_parser.add_argument('--epsilon-decay', type=float, default=0.995, help='Exploration rate decay') # Consider slower decay like 0.999 or 0.9995
    train_parser.add_argument('--memory-size', type=int, default=20000, help='Replay memory size') # Increased
    train_parser.add_argument('--batch-size', type=int, default=64, help='Batch size for training')
    train_parser.add_argument('--target-update-freq', type=int, default=10, help='Target network update frequency (in episodes)') # Slightly increased
    train_parser.add_argument('--checkpoint-freq', type=int, default=100, help='Checkpoint frequency')
    train_parser.add_argument('--win-reward', type=float, default=100.0, help='Reward for winning')
    train_parser.add_argument('--loss-penalty', type=float, default=-100.0, help='Penalty for losing')
    train_parser.add_argument('--separate-agents', action='store_true', help='Use separate agents for players')
    train_parser.add_argument('--model-dir', type=str, default='./models', help='Directory to save models')
    train_parser.add_argument('--log-dir', type=str, default='./logs', help='Directory to save logs')
    train_parser.add_argument('--seed', type=int, default=31, help='Random seed')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        # Make sure pandas is imported if saving logs from main
        global pd 
        import pandas as pd
        train_model(args)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()