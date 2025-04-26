import os
import argparse
import tensorflow as tf
import numpy as np
import random
import copy
from game_api import Scrabble_Game
# from valid_moves_script import load_dictionary as load_move_gen_dictionary
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
    
    agent1 = ScrabbleDQN(
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        epsilon=args.epsilon,
        epsilon_min=args.epsilon_min,
        epsilon_decay=args.epsilon_decay,
        memory_size=args.memory_size
    )
    
    if args.separate_agents:
        agent2 = ScrabbleDQN(
            learning_rate=args.learning_rate,
            gamma=args.gamma,
            epsilon=args.epsilon,
            epsilon_min=args.epsilon_min,
            epsilon_decay=args.epsilon_decay,
            memory_size=args.memory_size
        )
    else:
        agent2 = agent1
    
    trainer = ScrabbleSelfPlay(
        game_api_class=Scrabble_Game,
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
    return agent1, stats


def main():
    print("Main: Loading dictionary for move generator...")
    # load_move_gen_dictionary()
    parser = argparse.ArgumentParser(description='Scrabble DQN')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    train_parser = subparsers.add_parser('train', help='Train the DQN model')
    train_parser.add_argument('--num-episodes', type=int, default=1000, help='Number of episodes to train for')
    train_parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
    train_parser.add_argument('--gamma', type=float, default=0.95, help='Discount factor')
    train_parser.add_argument('--epsilon', type=float, default=1.0, help='Initial exploration rate')
    train_parser.add_argument('--epsilon-min', type=float, default=0.01, help='Minimum exploration rate')
    train_parser.add_argument('--epsilon-decay', type=float, default=0.995, help='Exploration rate decay')
    train_parser.add_argument('--memory-size', type=int, default=10000, help='Replay memory size')
    train_parser.add_argument('--batch-size', type=int, default=64, help='Batch size for training')
    train_parser.add_argument('--target-update-freq', type=int, default=5, help='Target network update frequency')
    train_parser.add_argument('--checkpoint-freq', type=int, default=100, help='Checkpoint frequency')
    train_parser.add_argument('--win-reward', type=float, default=100.0, help='Reward for winning')
    train_parser.add_argument('--loss-penalty', type=float, default=-100.0, help='Penalty for losing')
    train_parser.add_argument('--separate-agents', action='store_true', help='Use separate agents for players')
    train_parser.add_argument('--model-dir', type=str, default='./models', help='Directory to save models')
    train_parser.add_argument('--log-dir', type=str, default='./logs', help='Directory to save logs')
    train_parser.add_argument('--seed', type=int, default=31, help='Random seed')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        train_model(args)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()