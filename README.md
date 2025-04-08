# scrabble_engine

This is a project I have been working on for a while which trains and plays against a Deep Q-Network (DQN) Scrabble Engine. Started off as a fun exercise of re-creating scrabble in python and evolved into a fully fledged project on training a Deep Q-Learning Model to be able to predict the next best scrabble move. The trained model included in this repo was trained on an AWS EC2 instance and took approximately 48 hours to train. The model is not perfect by any means at the moment and was only trained for 100 episodes, as there is a large bottleneck in training with regards to the valid moves script, which takes anywhere from 3-10 seconds to run per call. I am still actively working on this project and plan to refine the training process, optimize the valid moves script, and research and implement further ways to improve and build on the foundation I already have.


## Prerequisites

- Python 3.6+

Pip Installs:
- tensorflow
- numpy
- pandas
- pygame
- pillow


## Installation

Clone the repository and install the required dependencies.


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
- `valid_moves_script.py`: Move validation and generation
- `dqn_state.py`: State encoding for the DQN
- `dqn_reward_functions.py`: Reward calculation
- `dqn_model.py`: DQN model implementation
- `self_play.py`: Self-play training framework
- `ai_vs_ai_gui.py`: GUI for finished models to play each other
- `human_vs_ai_gui.py`: GUI for you to play against a finished model

## Output

Training results and model checkpoints will be saved to these default directories:
- Models: `./models/`
- Logs: `./logs/`
