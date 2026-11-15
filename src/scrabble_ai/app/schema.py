"""What the "new run" form and the tuning guide need to know about every training setting.

Defaults and types come from the dataclasses in `scrabble_ai.config`, so they cannot drift.
This file adds what a person needs on top: a label, what the setting does, what turning it down
or up does, the range worth trying, sensible limits for the form, which settings are everyday
ones and which belong behind "all settings", and which cannot change on resume. A test checks
every config field is described here.
"""

from __future__ import annotations

import dataclasses

from scrabble_ai.config import OFFLINE_DEFAULTS, Config


def _f(label: str, what: str, lower: str, higher: str, typical: str = "", **extra) -> dict:
    """One setting. `extra` may hold min, max, step, options, advanced, locked, multiple_of.
    `step` is only how far the arrow keys move; `multiple_of` is a rule."""
    return {"label": label, "help": what, "lower": lower, "higher": higher, "typical": typical, **extra}


_FIELDS: dict[str, dict] = {
    # ---- run ---------------------------------------------------------------------------------
    "run.hours": _f(
        "Time limit", "Stop after this much wall-clock time. 0 means no time limit.",
        "A shorter session: fewer games, an earlier answer.",
        "More games. Worth it only once a short run shows the network is learning.",
        "0.5 for a test, 4 to 12 for a real run", min=0, max=1000, step=0.5, advanced=False),
    "run.max_games": _f(
        "Game limit", "Stop after this many self-play games. 0 means no limit.",
        "Stops sooner.", "Runs longer. Whichever of the two limits is reached first ends the run.",
        "0 (use the time limit), or 200,000 to 2,000,000", min=0, max=1_000_000_000, step=50_000, advanced=False),
    "run.seed": _f(
        "Random seed", "Fixes the self-play game seeds and all sampling.",
        "Any other number gives the same experiment with different games.",
        "Run two seeds of one setting to see how much of a difference is luck.",
        "any whole number", min=0, max=1_000_000, step=1),
    "run.lexicon": _f(
        "Word list", "A name from configs/lexicons.toml. Empty uses that file's default.",
        "Empty: the default list.", "Another list is a different game: networks and leave values are tied to their list.",
        "leave empty", locked=True),
    "run.leaves": _f(
        "Leave values file", "Empty uses the fitted file for the word list. \"zero\" trains without leave values.",
        "Empty: the fitted values, so an untrained network already plays like greedy + leave.",
        "\"zero\": the network must learn what tiles are worth from nothing. Much harder; only to measure what leave values buy.",
        "leave empty", locked=True),
    "run.device": _f(
        "Device", "Train on the GPU or, very slowly, the CPU.",
        "cuda: the GPU.", "cpu: far slower; only for checking that things run.",
        "cuda", options=["cuda", "cpu"]),
    "run.vram_fraction": _f(
        "GPU memory cap", "Share of GPU memory PyTorch may use. With the cap an oversized setting fails at once "
        "instead of silently spilling into system RAM.",
        "Leaves room for other programs; a large network or batch fails sooner.",
        "Allows larger networks and batches; less room for anything else on the GPU.",
        "0.8 to 0.95", min=0.1, max=1.0, step=0.05),
    # ---- model -------------------------------------------------------------------------------
    "model.blocks": _f(
        "Residual blocks", "Depth of the network.",
        "Faster games, so more of them per hour; less capacity.",
        "Slower (cost grows with depth), more capacity, memorises more readily and needs more games to pay off.",
        "4 to 10", min=1, max=40, step=1, advanced=False, locked=True),
    "model.channels": _f(
        "Channels", "Width of the network (a multiple of 8).",
        "Much faster: cost grows with the square of the width.",
        "Much slower and more memory; more capacity. 10x128 ran at a third of the speed of 6x64.",
        "64 to 128", min=8, max=512, step=8, multiple_of=8, advanced=False, locked=True),
    "model.norm": _f(
        "Normalization", "\"none\": no normalization, each block starts as the identity. \"group\": GroupNorm.",
        "none: fastest, and what every speed figure was measured with.",
        "group: about half the speed and 2.7 times the training memory. Try it only if training is unstable at any learning rate.",
        "none", options=["none", "group"], locked=True),
    "model.vector_hidden": _f(
        "Rack features width", "Width of the layers that read the leave, the unseen tiles and the scalars.",
        "A little less capacity for rack and bag knowledge.", "A little more; hardly changes the speed.",
        "64 to 256", min=8, max=1024, step=8, locked=True),
    "model.head_hidden": _f(
        "Head width", "Width of the last hidden layer.",
        "A smaller head: less able to memorise particular boards.",
        "A larger head: more capacity, most of it in the layer that reads the whole board at once.",
        "128 to 512", min=8, max=2048, step=8, locked=True),
    "model.value_channels": _f(
        "Per-square channels", "Channels kept per square for the head, so it can see where things are.",
        "The head sees a coarser picture of the board; fewer weights, less memorising.",
        "More positional detail, and many more weights in the head (225 x this x head width).",
        "2 to 16", min=1, max=64, step=1, locked=True),
    "model.cross_checks": _f(
        "Cross-check planes", "Show the network which letters fit on each empty square. This is how it sees the dictionary.",
        "Off: the network is blind to the dictionary. Only to measure what the planes are worth.",
        "On: the default.", "on", locked=True),
    "model.leave_baseline": _f(
        "Leave baseline", "The network predicts a correction to the static leave value.",
        "Off: it must learn leave values itself and starts far weaker than greedy + leave.",
        "On: an untrained network already plays like greedy + leave, and training can only add to that.",
        "on", locked=True),
    "model.extra_scalars": _f(
        "Extra facts", "Thirteen numbers the network would otherwise have to work out: the points the move scored, "
        "how long since the opponent last played all seven tiles, how much of the unseen pool is blanks, S, J, Q, X "
        "and Z, and the vowel balance of the pool and of the kept tiles.",
        "Off: only to measure what they are worth.", "On: the default. They cost nothing measurable.",
        "on", locked=True),
    "model.last_move_plane": _f(
        "Opponent's last move", "Marks the squares where the opponent's previous move put its tiles.",
        "Off: the network knows how many tiles they placed and for how many points, but not where.",
        "On: the default. It costs nothing measurable.", "on", locked=True),
    "model.openness": _f(
        "Board openness", "What the next player could do with the board a move leaves: which empty squares a "
        "placement can reach and with how few tiles, where seven tiles fit, and which triple- and double-word "
        "squares are in reach.",
        "Off: the network must work reach out from the tiles and the cross-check planes, a few squares at a time.",
        "On: the default. The engine works it out for every candidate: 7 to 11 microseconds a board on one "
        "core, against 3 to 8 for the cross-check masks.", "on", locked=True),
    "model.hooks": _f(
        "Hook squares", "Marks the squares where one letter extends a word already on the board, in front of it or "
        "after it, with how many letters fit there and where an S does.",
        "Off: the same facts are in the cross-check planes, less directly.",
        "On: the default. Worked out on the GPU from the cross-check masks.", "on", locked=True),
    "model.belief_samples": _f(
        "Opponent-rack inference", "After each placement the engine samples this many racks the mover might have "
        "had and weighs each by how good the move played would have been with it. The expected tiles on the "
        "opponent's rack are then an input. 0 leaves the input out.",
        "Cheaper and a rougher estimate. 0: no inference, and the network has no such input.",
        "A steadier estimate. Each sample is one full move generation per move played, so self-play slows down; "
        "when training from games it sets the pace: 11,500 positions a second were read with 4 racks, against "
        "187,000 with none.",
        "0, or 4 to 12", min=0, max=64, step=1, locked=True),
    "model.kind": _f(
        "Kind of network", "The body of the network. cnn: residual blocks of 3 x 3 convolutions. cnn-global: the same "
        "with board-wide pooling in every third block, so each square is told about the whole board (from KataGo). "
        "transformer: one token per square and layers of attention, in which every square can look at every other; "
        "Residual blocks is then the number of layers and Channels the width of a token.",
        "cnn and cnn-global: fast. In the one published comparison on Scrabble, convolutional networks of 64 to 128 "
        "channels all stopped at about 52.5% of games against the static evaluator.",
        "transformer: the kind behind the strongest published Scrabble value networks (57.65%, at 8 layers of 192), "
        "which also had the helper outputs. Several times slower; not timed here yet.",
        "cnn first; then transformer at 8 blocks x 192 channels", options=["cnn", "cnn-global", "transformer"], locked=True),
    "model.head": _f(
        "Head", "How the outputs read the board. flat: through a layer with a weight for every square, which is where "
        "the first networks learned particular boards by heart. pooled: through each channel's average and maximum "
        "over the board only.",
        "flat: what every network before 15 November 2026 has.",
        "pooled: far fewer weights (about 610,000 in a 6x64 network against 1.07 million) and nothing that can "
        "memorise a board square by square.", "pooled for training from games", options=["flat", "pooled"], locked=True),
    "model.win_head": _f(
        "Win output", "binary: one number, the chance of winning. wdl: three, for win, draw and loss.",
        "binary: what every earlier network has.", "wdl: the output training from games is built around.",
        "wdl for training from games", options=["binary", "wdl"], locked=True),
    "model.aux_heads": _f(
        "Helper outputs", "Extra outputs used only as training signal: the opponent's next score, whether they place "
        "all seven tiles, and where the next two moves land. They are never used to choose a move.",
        "Off: the network learns from the result of the game alone.",
        "On: in the published Scrabble results these were the largest single gain, about four points of win rate, "
        "measured on a transformer. Only training from games teaches them.", "on for training from games", locked=True),
    # ---- selfplay ----------------------------------------------------------------------------
    "selfplay.games": _f(
        "Games in lockstep", "Self-play games advanced together.",
        "Smaller batches for the GPU: fewer games a second.",
        "Speed is flat from 256 to 512; beyond that only more memory.",
        "256 to 512", min=1, max=2048, step=32),
    "selfplay.max_candidates": _f(
        "Candidates per move", "Moves per decision the network evaluates: the best by score + leave value.",
        "Faster (32 is about 30% faster a game), but the network can never choose a move the static ranking puts below the cut.",
        "Slower; lets the network find moves the static ranking rates low.",
        "32 to 128", min=2, max=512, step=8),
    "selfplay.max_exchanges": _f(
        "Exchange candidates", "Of those candidates, at most this many are exchanges.",
        "Fewer exchanges considered; a little faster.", "More kinds of exchange considered.",
        "2 to 8", min=0, max=32, step=1),
    "selfplay.temperature_start": _f(
        "Exploration at the start", "Moves are sampled from softmax(value / temperature), in points. At 4, a move 4 points "
        "worse is chosen about a third as often as the best.",
        "Early games stay close to the best move: stronger, less varied play. 0 always takes the top move.",
        "More second-best moves early: more variety to learn from, weaker and noisier games.",
        "2 to 6", min=0, max=50, step=0.5),
    "selfplay.temperature_end": _f(
        "Exploration at the end", "Temperature once the decay is complete.",
        "Late games are nearly the network's best play, which is the play its values should describe.",
        "Keeps exploring: more variety, but the values describe a sloppier player.",
        "0.3 to 1", min=0, max=50, step=0.1),
    "selfplay.temperature_decay_games": _f(
        "Exploration decay", "Self-play games over which the temperature falls from start to end.",
        "Exploration ends sooner.", "Explores for longer.",
        "a fifth to a half of the games you expect to play", min=1, max=100_000_000, step=50_000),
    "selfplay.win_weight": _f(
        "Play for wins", "0 chooses moves by expected spread, 1 by win probability, between blends the two.",
        "0: plays for points. The right choice until the value network is clearly good.",
        "Leans on the win estimate, which matters near the end of close games. Needs a well-trained win head.",
        "0, later 0.1 to 0.3", min=0, max=1, step=0.05),
    "selfplay.win_scale": _f(
        "Win value", "Points a certain win is worth in that blend.",
        "Win probability counts for less.", "Win probability counts for more.",
        "50 to 150", min=1, max=1000, step=10),
    # ---- warmstart ---------------------------------------------------------------------------
    "warmstart.games": _f(
        "Warm-start games", "Games played by the static evaluator before the network plays. 0 skips the warm start.",
        "A quicker start with less data to pretrain on. 0: self-play starts from an untrained network (which, with "
        "the leave baseline, plays like greedy + leave).",
        "More pretraining data, at about two and a half minutes per 100,000 games. 100,000 games nearly fill a "
        "3-million-position replay buffer.",
        "50,000 to 200,000", min=0, max=10_000_000, step=10_000),
    "warmstart.temperature": _f(
        "Warm-start exploration", "Exploration in those games, so the network also sees poor moves.",
        "Games close to greedy + leave play.", "More poor moves in the data.",
        "3 to 8", min=0, max=50, step=0.5),
    "warmstart.max_candidates": _f(
        "Warm-start candidates", "Candidates per move during the warm start.",
        "Only the top static moves are ever played.", "A wider spread of moves; slower.",
        "8 to 32", min=2, max=512, step=2),
    "warmstart.steps": _f(
        "Pretraining steps", "Training steps on the warm-start games before self-play begins.",
        "Less pretraining.", "More. Past one pass over the data (warm-start positions / batch size, about 2,300 steps "
        "by default) positions repeat and the network starts memorising them.",
        "1,000 to 4,000", min=0, max=1_000_000, step=500),
    # ---- train -------------------------------------------------------------------------------
    "train.batch_size": _f(
        "Batch size", "Positions per training step.",
        "More, noisier updates for the same games.", "Fewer, smoother updates; more GPU memory.",
        "512 to 2,048", min=16, max=16384, step=256),
    "train.learning_rate": _f(
        "Learning rate", "Step size of each update.",
        "Slower, steadier learning that chases less noise. The first thing to lower (to 0.0001) if a run keeps getting worse.",
        "Faster, and more easily knocked about by noisy targets. 0.001 was too high with the first runs' settings.",
        "0.0001 to 0.001", min=1e-6, max=1.0, step=0.0001),
    "train.warmup_steps": _f(
        "Warm-up steps", "Steps over which the learning rate ramps up from zero.",
        "Full speed sooner; a rougher start.", "A gentler start.",
        "200 to 2,000", min=0, max=100_000, step=100),
    "train.weight_decay": _f(
        "Weight decay", "Pulls the weights towards zero at every step.",
        "Less restraint on the network.", "More restraint: less memorising, slower fitting; too much and it cannot fit anything.",
        "0.00001 to 0.001", min=0, max=1, step=0.0001),
    "train.grad_clip": _f(
        "Gradient clip", "Maximum gradient norm. 0 turns clipping off.",
        "Caps large updates harder: steadier, slower.", "Lets large updates through; 0 removes the cap.",
        "0.5 to 5", min=0, max=100, step=0.5),
    "train.value_scale": _f(
        "Value scale", "Points per unit of the value output.",
        "The network works in larger numbers.", "The network works in smaller numbers.",
        "leave at 50", min=1, max=500, step=5, locked=True),
    "train.huber_delta": _f(
        "Huber delta", "The value loss is quadratic within this many units (2 = 100 points) and linear beyond.",
        "Big misses count for less: more robust to lucky and unlucky games.",
        "Closer to squared error: outlying games pull harder.",
        "1 to 3", min=0.01, max=100, step=0.5),
    "train.win_loss_weight": _f(
        "Win loss weight", "Weight of the win-probability loss relative to the value loss.",
        "The win estimate gets less attention; 0 leaves it untrained.", "More of the network's effort goes to predicting wins.",
        "0.25 to 1", min=0, max=10, step=0.1),
    "train.target": _f(
        "Learning rule", "What each position learns from. td: what followed in its own game, as TD lambda sets out. "
        "q: deep Q-learning: the best move the network believes the opponent has next, with the self-play copy of "
        "the network as the target network.",
        "td: learns from moves that were played. Steadier.",
        "q: learns from the move rated best, whether or not it was played. It takes a maximum over many uncertain "
        "ratings, which overstates them: watch the edge over leave values for it.",
        "td, or q to compare", options=["td", "q"]),
    "train.td_lambda": _f(
        "TD lambda", "How far ahead the training targets look. At 1 the target is the spread actually gained by the "
        "end of the game; below 1 it leans on the estimate of the next position.",
        "Targets lean on the next position's estimate: little noise, but mistakes in the estimates feed back into the targets.",
        "Targets move towards actual final results: unbiased but noisy (about 75 points at 1.0), which invites "
        "memorising them. 1.0 was what the first two runs used.",
        "0.5 to 0.8", min=0, max=1, step=0.05),
    "train.replay_capacity": _f(
        "Replay capacity", "Positions kept in RAM, about 370 bytes each: 3 million is 1.1 GB.",
        "Only recent games are kept: less memory, less variety.", "More variety and more memory; old games stay longer.",
        "2,000,000 to 10,000,000", min=1_000, max=100_000_000, step=500_000),
    "train.min_replay": _f(
        "Replay needed to start", "Positions required before training begins.",
        "Training starts on very little data.", "A longer wait before the first update. A warm start already provides this.",
        "50,000 to 500,000", min=1, max=100_000_000, step=10_000),
    "train.replay_ratio": _f(
        "Replay ratio", "Training samples drawn per new self-play position: about how many times each position is trained on.",
        "Each position is used once or less: nothing to memorise, fewer updates per game, more of the GPU left for self-play.",
        "More updates per game, but the same games come round again and their outcomes get memorised. At 4 both "
        "of the first runs got much worse.",
        "0.5 to 1.5", min=0.1, max=64, step=0.5),
    "train.actor_sync_steps": _f(
        "Self-play sync", "Training steps between copies of the weights into the self-play network.",
        "Self-play uses fresher weights.", "A steadier but staler self-play player.",
        "50 to 500", min=1, max=100_000, step=50),
    "train.guard": _f(
        "Guard", "When a network is significantly behind the best one for several evaluations in a row, go back to "
        "the best network's weights and halve the learning rate.",
        "Off: nothing stops a run drifting far below its best (best.pt is still kept).",
        "On: a run can waste time but cannot end far worse than it started.",
        "on"),
    "train.guard_patience": _f(
        "Guard patience", "Evaluations in a row significantly behind the best before the guard acts.",
        "1 reacts at once, and to the occasional unlucky evaluation too.", "Waits longer before going back.",
        "2 to 3", min=1, max=20, step=1),
    "train.augment_transpose": _f(
        "Transpose augmentation", "Randomly swap rows and columns: the one exact symmetry of the board.",
        "Off: half the variety in the training data.", "On: the default.", "on"),
    "train.amp": _f(
        "Mixed precision", "bf16 autocast.",
        "Off: full precision; slower and more memory (how much has not been measured here).",
        "On: what every speed figure was measured with.", "on"),
    "train.checkpoint_minutes": _f(
        "Checkpoint every", "How often the full training state, replay buffer included, is saved.",
        "Less lost if the run dies; each save writes about a gigabyte.", "Fewer saves, more to lose.",
        "10 to 60", min=0.5, max=600, step=1),
    "train.eval_minutes": _f(
        "Evaluate every", "How often the network plays the baselines and is tested against the best so far.",
        "More readings and a quicker guard; each evaluation stops training for about 25 seconds (6x64) to a "
        "minute (10x128).",
        "Fewer readings; more time for training.",
        "10 to 30", min=0.5, max=600, step=1),
    "train.eval_pairs": _f(
        "Evaluation pairs", "Seat-swapped pairs per baseline in each evaluation.",
        "Quicker and vaguer: 250 pairs give about +/- 8 points.", "Slower and sharper: 1,000 pairs give about +/- 4.",
        "250 to 1,000", min=2, max=100_000, step=50),
    "train.gate_pairs": _f(
        "Gate pairs", "Pairs when testing a network against the current best.",
        "Promotions and the guard act on noisier results.", "Surer decisions, slower: 500 pairs give about +/- 5.5 points.",
        "300 to 1,000", min=2, max=100_000, step=50),
    "train.keep_checkpoints": _f(
        "Checkpoints kept", "Full checkpoints kept on disk. The best network is always kept.",
        "Less disk used.", "More history to go back to, at 20 to 75 MB each.",
        "3 to 10", min=1, max=1000, step=1),
    # ---- offline: training from saved games ----------------------------------------------------
    "offline.games": _f(
        "Game set", "The saved games to learn from: a set made on this page.",
        "A smaller set: in the published results 5.4 million games were too few (50.98% against the static "
        "evaluator).", "A larger set: about 20 million games reached 53.1%, and doubling that added nothing without "
        "the helper outputs.", "a set of 20 million games or more", advanced=False, locked=True),
    "offline.passes": _f(
        "Passes", "Times through the game set. Each pass takes one position from every game, a different one each time.",
        "Sooner done; fewer positions seen.", "More positions from the same games. The published recipe used 6.",
        "4 to 8", min=0.01, max=100, step=1, advanced=False),
    "offline.batch_size": _f(
        "Batch size", "Positions per training step.",
        "More, noisier updates per pass.", "Fewer, smoother updates; more GPU memory.",
        "1,024 to 4,096", min=16, max=16384, step=256, locked=True),
    "offline.learning_rate": _f(
        "Learning rate", "Step size at its peak. It ramps up over the warm-up, then falls along a cosine to a "
        "twentieth of this by the last step.",
        "Slower, steadier.", "Faster; too high and the loss on held-out games stops falling.",
        "0.0001 to 0.001", min=1e-6, max=1.0, step=0.0001),
    "offline.warmup_steps": _f(
        "Warm-up steps", "Steps over which the learning rate ramps up from zero.",
        "Full speed sooner.", "A gentler start.", "200 to 2,000", min=0, max=100_000, step=100),
    "offline.weight_decay": _f(
        "Weight decay", "Pulls the weights towards zero at every step.",
        "Less restraint.", "More restraint; too much and the network cannot fit.", "0.00001 to 0.001", min=0, max=1, step=0.0001),
    "offline.grad_clip": _f(
        "Gradient clip", "Maximum gradient norm. 0 turns clipping off.",
        "Caps large updates harder.", "Lets large updates through.", "0.5 to 5", min=0, max=100, step=0.5),
    "offline.validation_games": _f(
        "Held-out games", "Games kept out of training, to measure the loss on games never learned from. At most a "
        "tenth of the set.",
        "Almost every game is trained on; the measurement is rougher.", "A steadier measurement; fewer games to train on.",
        "10,000 to 50,000", min=0, max=10_000_000, step=5_000, locked=True),
    "offline.validate_steps": _f(
        "Check every", "Training steps between measurements on the held-out games.",
        "A finer curve; a little time each.", "A coarser curve.", "200 to 2,000", min=1, max=1_000_000, step=100),
    "offline.aux_share": _f(
        "Helper pull", "Each helper output's pull on the network, as a share of the main output's. The weights are "
        "re-measured as training goes. 0 uses the two fixed weights below instead.",
        "Helpers matter less. In the published results fixed weights let one helper out-pull the main output and "
        "cost 1.4 points of win rate.", "Helpers matter more, at the main output's expense.",
        "0.1 to 0.25", min=0, max=1, step=0.05),
    "offline.spread_weight": _f(
        "Points weight", "Fixed weight of the points output, used only when Helper pull is 0.",
        "Points matter less.", "Points matter more.", "0.1 to 0.5", min=0, max=10, step=0.05),
    "offline.aux_weight": _f(
        "Helper weight", "Fixed weight of each other helper output, used only when Helper pull is 0.",
        "Helpers matter less.", "Helpers matter more.", "0.05 to 0.3", min=0, max=10, step=0.05),
    "offline.eval_minutes": _f(
        "Play the static bot every", "How often the network plays the static evaluator. The best network is the one "
        "with the highest win rate in these matches.",
        "More readings; each takes the GPU from training.", "Fewer readings.", "20 to 60", min=0.5, max=600, step=5),
    "offline.eval_pairs": _f(
        "Pairs per match", "Seat-swapped pairs in each of those matches.",
        "Quicker and vaguer.", "Slower and sharper: 2,000 pairs give about +/- 1.3 points of win rate, 20,000 about "
        "+/- 0.4.", "2,000 to 5,000", min=2, max=1_000_000, step=500),
    "offline.final_pairs": _f(
        "Pairs in the final match", "Pairs in the match played when every pass is done.",
        "A rougher final figure.", "A sharper one: 10,000 pairs give about +/- 0.6 points of win rate.",
        "10,000 to 50,000", min=2, max=1_000_000, step=1_000),
    "offline.checkpoint_minutes": _f(
        "Checkpoint every", "How often the training state is saved.",
        "Less lost if the run dies.", "Fewer saves.", "10 to 30", min=0.5, max=600, step=1),
    "offline.keep_checkpoints": _f(
        "Checkpoints kept", "Step files kept on disk. The best network is always kept.",
        "Less disk used.", "More history.", "2 to 5", min=1, max=1000, step=1),
    "offline.augment_transpose": _f(
        "Transpose augmentation", "Randomly swap rows and columns: the one exact symmetry of the board.",
        "Off: half the variety.", "On: the default.", "on"),
    "offline.amp": _f(
        "Mixed precision", "bf16 autocast.", "Off: full precision; slower and more memory.", "On: the default.", "on"),
}

# What a number counts, shown beside its input.
_UNITS = {
    "run.hours": "hours",
    "run.max_games": "games",
    "selfplay.games": "games",
    "selfplay.temperature_start": "points",
    "selfplay.temperature_end": "points",
    "selfplay.temperature_decay_games": "games",
    "selfplay.win_scale": "points",
    "warmstart.games": "games",
    "warmstart.temperature": "points",
    "warmstart.steps": "steps",
    "train.batch_size": "positions",
    "train.warmup_steps": "steps",
    "train.value_scale": "points",
    "train.replay_capacity": "positions",
    "train.min_replay": "positions",
    "train.actor_sync_steps": "steps",
    "train.checkpoint_minutes": "minutes",
    "train.eval_minutes": "minutes",
    "train.eval_pairs": "pairs",
    "train.gate_pairs": "pairs",
    "train.guard_patience": "evaluations",
    "model.belief_samples": "racks",
    "offline.batch_size": "positions",
    "offline.warmup_steps": "steps",
    "offline.validation_games": "games",
    "offline.validate_steps": "steps",
    "offline.eval_minutes": "minutes",
    "offline.eval_pairs": "pairs",
    "offline.final_pairs": "pairs",
    "offline.checkpoint_minutes": "minutes",
}

# The form asks for the name itself, and the app decides where runs live.
HIDDEN = {"run.name", "run.out_dir"}

_SECTION_TITLES = {"run": "Run", "model": "Network", "selfplay": "Self-play", "warmstart": "Warm start", "train": "Training",
                   "offline": "Training from games"}

# Which way of training a setting belongs to: self-play, training from saved games, or both. The
# two forms each show only their own.
_GAMES_TOO = {"selfplay.max_candidates", "selfplay.max_exchanges", "selfplay.win_weight", "selfplay.win_scale",
              "train.value_scale", "train.huber_delta"}


def _modes(key: str) -> list[str]:
    section = key.partition(".")[0]
    if section == "offline":
        return ["games"]
    if key == "run.max_games":
        return ["selfplay"]  # training from games ends when its passes are done
    if section in ("run", "model") or key in _GAMES_TOO:
        return ["selfplay", "games"]
    return ["selfplay"]


PRESETS = [
    {"name": "Default 6x64", "description": "The size to get working first. About 57 games a second with the earlier inputs; not timed with the new ones yet.", "overrides": {}},
    {"name": "Larger 10x128", "description": "Try it once the default has clearly learned something. About 18 games a second with the earlier inputs.",
     "overrides": {"model.blocks": 10, "model.channels": 128}},
    {"name": "Wide 12x192", "description": "A weekend run: an estimated 8.5 games a second. Fewer candidates and a shorter exploration decay to suit.",
     "overrides": {"model.blocks": 12, "model.channels": 192, "selfplay.max_candidates": 32,
                   "selfplay.temperature_decay_games": 100_000}},
    {"name": "Deep Q-learning 6x64", "description": "The default network, learning by Q-learning: each position learns "
     "from the move the network rates best for the opponent, not from the move that was played.",
     "overrides": {"train.target": "q"}},
    {"name": "Quick test (30 minutes)", "description": "A short run to check everything works. Too short to learn much.",
     "overrides": {"run.hours": 0.5, "warmstart.games": 20_000, "warmstart.steps": 400,
                   "selfplay.temperature_decay_games": 50_000, "train.eval_minutes": 10.0}},
]

# Starting points for training from saved games. Each is applied on top of OFFLINE_DEFAULTS.
GAMES_PRESETS = [
    {"name": "CNN 6x64", "description": "The fastest to train and to play. Convolutional networks stopped near 52.5% "
     "against the static evaluator in the published results, without the helper outputs this one has.", "overrides": {}},
    {"name": "CNN with board-wide pooling 6x64", "description": "The same with KataGo's global pooling, so every "
     "square is told about the whole board. Hardly slower. Not tried on Scrabble by anyone yet.",
     "overrides": {"model.kind": "cnn-global"}},
    {"name": "Transformer 8x192", "description": "The shape of the strongest published Scrabble value networks "
     "(57.65% against the static evaluator). Several times slower to train and to play; not timed here yet.",
     "overrides": {"model.kind": "transformer", "model.blocks": 8, "model.channels": 192}},
    {"name": "Quick test", "description": "A tenth of a pass with short matches, to check everything works.",
     "overrides": {"offline.passes": 0.1, "offline.eval_pairs": 200, "offline.final_pairs": 500, "offline.validate_steps": 100}},
]

# Speeds on the RTX 5070 Ti (docs/BENCHMARKS.md). 6x64 and 10x128 are whole-run averages from the
# first two real runs (2026-11-13: 450,005 games in 2.19 h and 441,805 games in 6.67 h, with a
# replay ratio of 4; the current default of 1 leaves more of the GPU for self-play, so expect
# somewhat more). The larger sizes are worked out from separately measured speeds.
#
# Every figure was measured with the inputs of before 2026-11-14. The inputs added then make
# each decision cost the engine more (openness for every candidate, and the opponent-rack
# inference at `model.belief_samples` move generations per move played), which has not been
# timed on the GPU yet, so no size is marked as measured until it has. The parameter counts are
# those of networks with the current default inputs.
SPEED = [
    {"network": "6x64", "parameters": 1_071_760, "games_per_second": 57, "measured": False, "vram_gb": 1.4},
    {"network": "10x128", "parameters": 3_679_508, "games_per_second": 18, "measured": False, "vram_gb": 4.1},
    {"network": "12x192", "parameters": 8_794_262, "games_per_second": 8.5, "measured": False, "vram_gb": 7.1},
    {"network": "16x192", "parameters": 11_450_010, "games_per_second": 6, "measured": False, "vram_gb": 9.1},
    {"network": "20x256", "parameters": 24_529_438, "games_per_second": 3.4, "measured": False, "vram_gb": 7.6},
]

_TYPE_NAMES = {"int": "int", "float": "float", "bool": "bool", "str": "str"}


def schema() -> dict:
    """The JSON of GET /api/config/schema."""
    defaults = Config()
    sections = []
    for section_field in dataclasses.fields(defaults):
        section = getattr(defaults, section_field.name)
        fields = []
        for field in dataclasses.fields(section):
            key = f"{section_field.name}.{field.name}"
            if key in HIDDEN:
                continue
            described = _FIELDS[key]
            type_name = _TYPE_NAMES[field.type if isinstance(field.type, str) else field.type.__name__]
            options = described.get("options")
            fields.append({
                "key": key,
                "label": described["label"],
                "type": "choice" if options else type_name,
                "default": getattr(section, field.name),
                "min": described.get("min"),
                "max": described.get("max"),
                "step": described.get("step"),
                "multiple_of": described.get("multiple_of"),
                "unit": _UNITS.get(key),
                "options": options,
                "advanced": described.get("advanced", True),
                "locked_on_resume": described.get("locked", False),
                "help": described["help"],
                "effects": {"lower": described["lower"], "higher": described["higher"], "typical": described["typical"]},
                "modes": _modes(key),
            })
        sections.append({"name": section_field.name, "title": _SECTION_TITLES[section_field.name], "fields": fields})
    return {"sections": sections, "presets": PRESETS, "speed": SPEED, "games_presets": GAMES_PRESETS,
            "games_defaults": dict(OFFLINE_DEFAULTS)}


def described_keys() -> set[str]:
    return set(_FIELDS)


def _shown(default) -> str:
    if default is True:
        return "on"
    if default is False:
        return "off"
    if default == "":
        return "(empty)"
    return f"{default:,}" if isinstance(default, int) else str(default)


def guide_markdown() -> str:
    """The tuning guide as one Markdown table per section. docs/TUNING.md holds its output, and a
    test keeps the two the same."""
    lines = []
    for section in schema()["sections"]:
        lines += [f"### {section['title']} (`[{section['name']}]`)", "",
                  "| Setting | Default | What it does | Lower | Higher | Worth trying |", "|---|---|---|---|---|---|"]
        for field in section["fields"]:
            cells = [f"**{field['label']}**<br>`{field['key']}`", _shown(field["default"]), field["help"],
                     field["effects"]["lower"], field["effects"]["higher"], field["effects"]["typical"]]
            lines.append("| " + " | ".join(cell.replace("|", "/") for cell in cells) + " |")
        lines.append("")
    return "\n".join(lines)


GUIDE_MARKER = "<!-- The tables below are generated: python -m scrabble_ai.app.schema --write -->"


def write_guide(path) -> None:
    """Replace everything after GUIDE_MARKER in docs/TUNING.md with the current tables."""
    text = path.read_text(encoding="utf-8")
    head, marker, _ = text.partition(GUIDE_MARKER)
    if not marker:
        raise SystemExit(f"{path} has no line reading {GUIDE_MARKER!r}")
    path.write_text(f"{head}{marker}\n\n{guide_markdown()}", encoding="utf-8", newline="\n")


if __name__ == "__main__":  # prints the tables; --write puts them into docs/TUNING.md
    import sys

    if "--write" in sys.argv[1:]:
        from scrabble_ai.lexicon import REPO_ROOT

        write_guide(REPO_ROOT / "docs" / "TUNING.md")
    else:
        print(guide_markdown())
