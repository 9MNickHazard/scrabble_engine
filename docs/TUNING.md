# Tuning guide

What each training setting does, what turning it down or up does, and the range worth trying.
The app shows the same text under **Training → Tuning guide** and beside each setting in the new
run form.

The tables at the end are generated from `src/scrabble_ai/app/schema.py`
(`python -m scrabble_ai.app.schema --write` rewrites them here), and a test fails if this file
and the app differ.

## Where the defaults come from

The `[train]` defaults changed on 2026-11-13, after the first two real runs both got worse as they
trained ([PLAN.md](PLAN.md) has the numbers). Those runs trained on every position about four
times (`replay_ratio = 4`) against the actual final result of its game (`td_lambda = 1`), at a
learning rate of 0.001. The networks memorised the outcomes of the games in the replay buffer:
their error on those games fell while their error on new games rose above that of leave values
alone, and their play fell 45 to 80 points a game behind greedy + leave.

The current defaults train on each position about once (`replay_ratio = 1`), against targets that
lean on the estimate of the next position (`td_lambda = 0.7`), at a learning rate of 0.0003, with
a guard that goes back to the best network when a run falls behind it. They follow from that
diagnosis. **No full run has used them yet**, so treat them as the place to start, not as proven.

## How to tune

1. Get the default 6x64 network to beat greedy-leave before trying anything larger. A larger
   network plays fewer games an hour and memorises more readily; it only pays off once the
   smaller one has stopped improving.
2. Change one setting at a time, and give the two runs the same time limit.
3. A difference between two runs counts only if their evaluation intervals do not overlap. For a
   closer call, play the two best networks against each other in the Arena with 1,000 pairs or
   more, or repeat the run with another `run.seed`.
4. Use the "Quick test" preset to check a setting runs at all. It is too short to say whether the
   setting is better.

## What to watch

| Chart on the run page | Healthy | Trouble |
|---|---|---|
| Evaluation against greedy-leave | Rises, and its band clears zero | Falls away from its first reading |
| Edge (leave-only error minus the network's, on new games) | Above zero and rising | Below zero: the network predicts new games worse than no network |
| Error on seen and unseen positions | The two lines stay close | Seen falls while unseen rises: memorising |
| Gate against the best network | Promotions now and then | Every gate lost; the guard going back repeatedly |
| Loss | Falls slowly, then levels | Spikes, or a steady rise |

## If you see this

| What you see | Likely cause | Try, in this order |
|---|---|---|
| Evaluation falls and the edge goes below zero | Memorising the replay buffer | `train.replay_ratio` 0.5; `train.td_lambda` 0.5; `train.learning_rate` 0.0001; a larger `train.replay_capacity` |
| The guard goes back to the best network again and again | Each stretch of training makes the network worse | The same as above; a lower learning rate first |
| Evaluation flat at zero, edge near zero | Learning too slowly, or nothing left to learn at this size | `train.learning_rate` 0.001; `train.replay_ratio` 1.5; more hours; then a larger network |
| Loss spikes | Updates too large | `train.learning_rate` halved; `train.grad_clip` 0.5 |
| Good against greedy-leave but its own games look odd | Too little exploration late in the run | `selfplay.temperature_end` 1.0 |
| Self-play slower than the speed table | The GPU is shared, or training takes too much of it | Close other GPU programs; `train.replay_ratio` lower; `selfplay.max_candidates` 32 |
| "CUDA out of memory" at the start | Network or batch too large for the cap | `train.batch_size` halved; `selfplay.games` 256; a smaller network |

## Every setting

Settings marked as locked in the app (the network's shape, the word list, the leave values,
`train.value_scale`) cannot change when a run is resumed.

<!-- The tables below are generated: python -m scrabble_ai.app.schema --write -->

### Run (`[run]`)

| Setting | Default | What it does | Lower | Higher | Worth trying |
|---|---|---|---|---|---|
| **Word list**<br>`run.lexicon` | (empty) | A name from configs/lexicons.toml. Empty uses that file's default. | Empty: the default list. | Another list is a different game: networks and leave values are tied to their list. | leave empty |
| **Leave values file**<br>`run.leaves` | (empty) | Empty uses the fitted file for the word list. "zero" trains without leave values. | Empty: the fitted values, so an untrained network already plays like greedy + leave. | "zero": the network must learn what tiles are worth from nothing. Much harder; only to measure what leave values buy. | leave empty |
| **Random seed**<br>`run.seed` | 1 | Fixes the self-play game seeds and all sampling. | Any other number gives the same experiment with different games. | Run two seeds of one setting to see how much of a difference is luck. | any whole number |
| **Time limit**<br>`run.hours` | 0.0 | Stop after this much wall-clock time. 0 means no time limit. | A shorter session: fewer games, an earlier answer. | More games. Worth it only once a short run shows the network is learning. | 0.5 for a test, 4 to 12 for a real run |
| **Game limit**<br>`run.max_games` | 0 | Stop after this many self-play games. 0 means no limit. | Stops sooner. | Runs longer. Whichever of the two limits is reached first ends the run. | 0 (use the time limit), or 200,000 to 2,000,000 |
| **Device**<br>`run.device` | cuda | Train on the GPU or, very slowly, the CPU. | cuda: the GPU. | cpu: far slower; only for checking that things run. | cuda |
| **GPU memory cap**<br>`run.vram_fraction` | 0.9 | Share of GPU memory PyTorch may use. With the cap an oversized setting fails at once instead of silently spilling into system RAM. | Leaves room for other programs; a large network or batch fails sooner. | Allows larger networks and batches; less room for anything else on the GPU. | 0.8 to 0.95 |

### Network (`[model]`)

| Setting | Default | What it does | Lower | Higher | Worth trying |
|---|---|---|---|---|---|
| **Residual blocks**<br>`model.blocks` | 6 | Depth of the network. | Faster games, so more of them per hour; less capacity. | Slower (cost grows with depth), more capacity, memorises more readily and needs more games to pay off. | 4 to 10 |
| **Channels**<br>`model.channels` | 64 | Width of the network (a multiple of 8). | Much faster: cost grows with the square of the width. | Much slower and more memory; more capacity. 10x128 ran at a third of the speed of 6x64. | 64 to 128 |
| **Normalization**<br>`model.norm` | none | "none": no normalization, each block starts as the identity. "group": GroupNorm. | none: fastest, and what every speed figure was measured with. | group: about half the speed and 2.7 times the training memory. Try it only if training is unstable at any learning rate. | none |
| **Rack features width**<br>`model.vector_hidden` | 128 | Width of the layers that read the leave, the unseen tiles and the scalars. | A little less capacity for rack and bag knowledge. | A little more; hardly changes the speed. | 64 to 256 |
| **Head width**<br>`model.head_hidden` | 256 | Width of the last hidden layer. | A smaller head: less able to memorise particular boards. | A larger head: more capacity, most of it in the layer that reads the whole board at once. | 128 to 512 |
| **Per-square channels**<br>`model.value_channels` | 8 | Channels kept per square for the head, so it can see where things are. | The head sees a coarser picture of the board; fewer weights, less memorising. | More positional detail, and many more weights in the head (225 x this x head width). | 2 to 16 |
| **Cross-check planes**<br>`model.cross_checks` | on | Show the network which letters fit on each empty square. This is how it sees the dictionary. | Off: the network is blind to the dictionary. Only to measure what the planes are worth. | On: the default. | on |
| **Leave baseline**<br>`model.leave_baseline` | on | The network predicts a correction to the static leave value. | Off: it must learn leave values itself and starts far weaker than greedy + leave. | On: an untrained network already plays like greedy + leave, and training can only add to that. | on |
| **Extra facts**<br>`model.extra_scalars` | on | Thirteen numbers the network would otherwise have to work out: the points the move scored, how long since the opponent last played all seven tiles, how much of the unseen pool is blanks, S, J, Q, X and Z, and the vowel balance of the pool and of the kept tiles. | Off: only to measure what they are worth. | On: the default. They cost nothing measurable. | on |
| **Opponent's last move**<br>`model.last_move_plane` | on | Marks the squares where the opponent's previous move put its tiles. | Off: the network knows how many tiles they placed and for how many points, but not where. | On: the default. It costs nothing measurable. | on |
| **Board openness**<br>`model.openness` | on | What the next player could do with the board a move leaves: which empty squares a placement can reach and with how few tiles, where seven tiles fit, and which triple- and double-word squares are in reach. | Off: the network must work reach out from the tiles and the cross-check planes, a few squares at a time. | On: the default. The engine works it out for every candidate: 7 to 11 microseconds a board on one core, against 3 to 8 for the cross-check masks. | on |
| **Hook squares**<br>`model.hooks` | on | Marks the squares where one letter extends a word already on the board, in front of it or after it, with how many letters fit there and where an S does. | Off: the same facts are in the cross-check planes, less directly. | On: the default. Worked out on the GPU from the cross-check masks. | on |
| **Opponent-rack inference**<br>`model.belief_samples` | 4 | After each placement the engine samples this many racks the mover might have had and weighs each by how good the move played would have been with it. The expected tiles on the opponent's rack are then an input. 0 leaves the input out. | Cheaper and a rougher estimate. 0: no inference, and the network has no such input. | A steadier estimate. Each sample is one full move generation per move played, so self-play slows down. | 0, or 4 to 12 |

### Self-play (`[selfplay]`)

| Setting | Default | What it does | Lower | Higher | Worth trying |
|---|---|---|---|---|---|
| **Games in lockstep**<br>`selfplay.games` | 256 | Self-play games advanced together. | Smaller batches for the GPU: fewer games a second. | Speed is flat from 256 to 512; beyond that only more memory. | 256 to 512 |
| **Candidates per move**<br>`selfplay.max_candidates` | 64 | Moves per decision the network evaluates: the best by score + leave value. | Faster (32 is about 30% faster a game), but the network can never choose a move the static ranking puts below the cut. | Slower; lets the network find moves the static ranking rates low. | 32 to 128 |
| **Exchange candidates**<br>`selfplay.max_exchanges` | 4 | Of those candidates, at most this many are exchanges. | Fewer exchanges considered; a little faster. | More kinds of exchange considered. | 2 to 8 |
| **Exploration at the start**<br>`selfplay.temperature_start` | 4.0 | Moves are sampled from softmax(value / temperature), in points. At 4, a move 4 points worse is chosen about a third as often as the best. | Early games stay close to the best move: stronger, less varied play. 0 always takes the top move. | More second-best moves early: more variety to learn from, weaker and noisier games. | 2 to 6 |
| **Exploration at the end**<br>`selfplay.temperature_end` | 0.5 | Temperature once the decay is complete. | Late games are nearly the network's best play, which is the play its values should describe. | Keeps exploring: more variety, but the values describe a sloppier player. | 0.3 to 1 |
| **Exploration decay**<br>`selfplay.temperature_decay_games` | 300,000 | Self-play games over which the temperature falls from start to end. | Exploration ends sooner. | Explores for longer. | a fifth to a half of the games you expect to play |
| **Play for wins**<br>`selfplay.win_weight` | 0.0 | 0 chooses moves by expected spread, 1 by win probability, between blends the two. | 0: plays for points. The right choice until the value network is clearly good. | Leans on the win estimate, which matters near the end of close games. Needs a well-trained win head. | 0, later 0.1 to 0.3 |
| **Win value**<br>`selfplay.win_scale` | 100.0 | Points a certain win is worth in that blend. | Win probability counts for less. | Win probability counts for more. | 50 to 150 |

### Warm start (`[warmstart]`)

| Setting | Default | What it does | Lower | Higher | Worth trying |
|---|---|---|---|---|---|
| **Warm-start games**<br>`warmstart.games` | 100,000 | Games played by the static evaluator before the network plays. 0 skips the warm start. | A quicker start with less data to pretrain on. 0: self-play starts from an untrained network (which, with the leave baseline, plays like greedy + leave). | More pretraining data, at about two and a half minutes per 100,000 games. 100,000 games nearly fill a 3-million-position replay buffer. | 50,000 to 200,000 |
| **Warm-start exploration**<br>`warmstart.temperature` | 6.0 | Exploration in those games, so the network also sees poor moves. | Games close to greedy + leave play. | More poor moves in the data. | 3 to 8 |
| **Warm-start candidates**<br>`warmstart.max_candidates` | 16 | Candidates per move during the warm start. | Only the top static moves are ever played. | A wider spread of moves; slower. | 8 to 32 |
| **Pretraining steps**<br>`warmstart.steps` | 2,000 | Training steps on the warm-start games before self-play begins. | Less pretraining. | More. Past one pass over the data (warm-start positions / batch size, about 2,300 steps by default) positions repeat and the network starts memorising them. | 1,000 to 4,000 |

### Training (`[train]`)

| Setting | Default | What it does | Lower | Higher | Worth trying |
|---|---|---|---|---|---|
| **Batch size**<br>`train.batch_size` | 1,024 | Positions per training step. | More, noisier updates for the same games. | Fewer, smoother updates; more GPU memory. | 512 to 2,048 |
| **Learning rate**<br>`train.learning_rate` | 0.0003 | Step size of each update. | Slower, steadier learning that chases less noise. The first thing to lower (to 0.0001) if a run keeps getting worse. | Faster, and more easily knocked about by noisy targets. 0.001 was too high with the first runs' settings. | 0.0001 to 0.001 |
| **Warm-up steps**<br>`train.warmup_steps` | 500 | Steps over which the learning rate ramps up from zero. | Full speed sooner; a rougher start. | A gentler start. | 200 to 2,000 |
| **Weight decay**<br>`train.weight_decay` | 0.0001 | Pulls the weights towards zero at every step. | Less restraint on the network. | More restraint: less memorising, slower fitting; too much and it cannot fit anything. | 0.00001 to 0.001 |
| **Gradient clip**<br>`train.grad_clip` | 1.0 | Maximum gradient norm. 0 turns clipping off. | Caps large updates harder: steadier, slower. | Lets large updates through; 0 removes the cap. | 0.5 to 5 |
| **Value scale**<br>`train.value_scale` | 50.0 | Points per unit of the value output. | The network works in larger numbers. | The network works in smaller numbers. | leave at 50 |
| **Huber delta**<br>`train.huber_delta` | 2.0 | The value loss is quadratic within this many units (2 = 100 points) and linear beyond. | Big misses count for less: more robust to lucky and unlucky games. | Closer to squared error: outlying games pull harder. | 1 to 3 |
| **Win loss weight**<br>`train.win_loss_weight` | 0.5 | Weight of the win-probability loss relative to the value loss. | The win estimate gets less attention; 0 leaves it untrained. | More of the network's effort goes to predicting wins. | 0.25 to 1 |
| **TD lambda**<br>`train.td_lambda` | 0.7 | How far ahead the training targets look. At 1 the target is the spread actually gained by the end of the game; below 1 it leans on the estimate of the next position. | Targets lean on the next position's estimate: little noise, but mistakes in the estimates feed back into the targets. | Targets move towards actual final results: unbiased but noisy (about 75 points at 1.0), which invites memorising them. 1.0 was what the first two runs used. | 0.5 to 0.8 |
| **Replay capacity**<br>`train.replay_capacity` | 3,000,000 | Positions kept in RAM, about 370 bytes each: 3 million is 1.1 GB. | Only recent games are kept: less memory, less variety. | More variety and more memory; old games stay longer. | 2,000,000 to 10,000,000 |
| **Replay needed to start**<br>`train.min_replay` | 100,000 | Positions required before training begins. | Training starts on very little data. | A longer wait before the first update. A warm start already provides this. | 50,000 to 500,000 |
| **Replay ratio**<br>`train.replay_ratio` | 1.0 | Training samples drawn per new self-play position: about how many times each position is trained on. | Each position is used once or less: nothing to memorise, fewer updates per game, more of the GPU left for self-play. | More updates per game, but the same games come round again and their outcomes get memorised. At 4 both of the first runs got much worse. | 0.5 to 1.5 |
| **Self-play sync**<br>`train.actor_sync_steps` | 100 | Training steps between copies of the weights into the self-play network. | Self-play uses fresher weights. | A steadier but staler self-play player. | 50 to 500 |
| **Guard**<br>`train.guard` | on | When a network is significantly behind the best one for several evaluations in a row, go back to the best network's weights and halve the learning rate. | Off: nothing stops a run drifting far below its best (best.pt is still kept). | On: a run can waste time but cannot end far worse than it started. | on |
| **Guard patience**<br>`train.guard_patience` | 2 | Evaluations in a row significantly behind the best before the guard acts. | 1 reacts at once, and to the occasional unlucky evaluation too. | Waits longer before going back. | 2 to 3 |
| **Transpose augmentation**<br>`train.augment_transpose` | on | Randomly swap rows and columns: the one exact symmetry of the board. | Off: half the variety in the training data. | On: the default. | on |
| **Mixed precision**<br>`train.amp` | on | bf16 autocast. | Off: full precision; slower and more memory (how much has not been measured here). | On: what every speed figure was measured with. | on |
| **Checkpoint every**<br>`train.checkpoint_minutes` | 10.0 | How often the full training state, replay buffer included, is saved. | Less lost if the run dies; each save writes about a gigabyte. | Fewer saves, more to lose. | 10 to 60 |
| **Evaluate every**<br>`train.eval_minutes` | 20.0 | How often the network plays the baselines and is tested against the best so far. | More readings and a quicker guard; each evaluation stops training for about 25 seconds (6x64) to a minute (10x128). | Fewer readings; more time for training. | 10 to 30 |
| **Evaluation pairs**<br>`train.eval_pairs` | 250 | Seat-swapped pairs per baseline in each evaluation. | Quicker and vaguer: 250 pairs give about +/- 8 points. | Slower and sharper: 1,000 pairs give about +/- 4. | 250 to 1,000 |
| **Gate pairs**<br>`train.gate_pairs` | 500 | Pairs when testing a network against the current best. | Promotions and the guard act on noisier results. | Surer decisions, slower: 500 pairs give about +/- 5.5 points. | 300 to 1,000 |
| **Checkpoints kept**<br>`train.keep_checkpoints` | 5 | Full checkpoints kept on disk. The best network is always kept. | Less disk used. | More history to go back to, at 20 to 75 MB each. | 3 to 10 |
