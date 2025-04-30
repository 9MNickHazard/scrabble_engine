# scrabble_engine

## NOTE (4/30/25): Using "human vs ai" or "ai vs ai" will not currently work until the new 7 output model is finished training on AWS and is uploaded to this repo. 

This is a project I have been working on for a while which trains and plays against a Deep Q-Network (DQN) Scrabble Engine. Started off as a fun exercise of re-creating scrabble in python and evolved into a fully fledged project on training a Deep Q-Learning Model to be able to predict the next best scrabble move. The new model now included in this repo was trained on an AWS EC2 instance and took approximately 34 hours to train for 200 episodes, a drastic increase in efficiency, with the last model taking around 50 hours for 100 episodes. I made some crucial tweaks (all included in the files currently in this repo) to the reward functions, that has made a massive difference in the capabilities of the resulting model.


## Prerequisites

- Python 3.6+
- Rust

Pip Installs:
- maturin (if building rust library on a system other than windows)
- tensorflow
- numpy
- pandas
- pygame
- pillow
- matplotlib
- /path/to/wheel/file/scrabble_valid_moves_rust-0.1.0-cp310-cp310-win_amd64.whl


## Installation

- Clone the repository and install the required dependencies.
- Either use maturin to build a linux/mac compatible Python library for the Rust script, or pip install the existing .whl file for windows use
- If using maturin for non-windows build, use command `maturin build --release` in main project directory, then pip install the resulting .whl file


## Training the Model

To train the DQN model, use the following command with any parameters you would like to change:
`python dqn_main.py train`


### Training Parameters

Customize the training with these parameters:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--num-episodes` | Number of episodes to train for | 1000 |
| `--learning-rate` | Learning rate for the optimizer | 0.001 |
| `--gamma` | Discount factor for future rewards | 0.95 |
| `--epsilon` | Initial exploration rate | 1.0 |
| `--epsilon-min` | Minimum exploration rate | 0.01 |
| `--epsilon-decay` | Exploration rate decay factor | 0.995 |
| `--memory-size` | Size of the replay memory buffer | 10000 |
| `--batch-size` | Batch size for training | 64 |
| `--target-update-freq` | Target network update frequency | 5 |
| `--checkpoint-freq` | Frequency to save model checkpoints | 100 |
| `--win-reward` | Reward value for winning a game | 100 |
| `--loss-penalty` | Penalty value for losing a game | -100 |
| `--separate-agents` | Use separate agents for players | False |
| `--model-dir` | Directory to save trained models | ./models |
| `--log-dir` | Directory to save training logs | ./logs |
| `--seed` | Random seed for reproducibility | 31 |


### Example

Train for 2000 episodes with a custom learning rate and exploration parameters:
`python dqn_main.py train --num-episodes 2000 --learning-rate 0.0005 --epsilon 0.9 --epsilon-decay 0.99`


## To play against the trained model/have two models play against each other:

For human vs ai (choose model once gui pops up):
`python human_vs_ai_gui.py`

Fro ai vs ai (choose models once gui pops up):
`python ai_vs_ai_gui.py`


## Project Structure

- `dqn_main.py`: Main entry point for training
- `game_api.py`: Scrabble game implementation
- `dqn_state.py`: State encoding for the DQN
- `dqn_reward_functions.py`: Reward calculation
- `dqn_model.py`: DQN model implementation
- `self_play.py`: Self-play training framework
- `ai_vs_ai_gui.py`: GUI for finished models to play each other
- `human_vs_ai_gui.py`: GUI for you to play against a finished model
- `scrabble_valid_moves_rust-0.1.0-cp310-cp310-win_amd64.whl`: Windows packaged rust library for python for valid moves script
- `/src/lib.rs`: Main rust script for valid moves algorithm
- `/src/main.rs`: File for testing valid moves script on example board state and player hand
- `agent1_final_4-27_200_eps.weights.h5`: Trained model, ready to play against

## Output

Training results and model checkpoints will be saved to these default directories:
- Models: `./models/`
- Logs: `./logs/`
