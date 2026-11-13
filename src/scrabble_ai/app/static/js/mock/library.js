/* Mock API for the Training and Models pages. Loaded only with ?mock=1.
 *
 * Answers GET /api/config/schema, GET /api/models, POST /api/models/<id>/meta, GET /api/leaves
 * and POST /api/runs. It also registers fallbacks for GET /api/status, GET /api/runs, the
 * per-run metrics and stop / resume / delete, but only when js/mock/training.js (loaded first)
 * has not already registered them.
 *
 * Mock-only extras:
 *   index.html?mock=1&theme=light       start in the light theme (or theme=dark)
 *   GET /api/models?demo=empty|many     no models, or forty of them
 */
(function () {
  "use strict";

  const M = App.mock;
  const search = new URLSearchParams(location.search);
  const theme = search.get("theme");
  if (theme === "light" || theme === "dark") document.documentElement.dataset.theme = theme;

  const NOW = Date.now() / 1000;
  const HOUR = 3600;
  const fail = (message) => ({ error: message });

  function random(seed) {
    let s = seed >>> 0;
    return () => {
      s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
      return s / 4294967296;
    };
  }

  // ---- Settings schema, built from configs/train.toml and config.py ---------------------

  function field(key, label, type, def, extra) {
    return Object.assign({
      key, label, type, default: def, min: null, max: null, step: null, options: null,
      unit: null, advanced: true, locked_on_resume: false, help: "",
    }, extra);
  }
  const INT = (min, max, extra) => Object.assign({ min, max, step: 1 }, extra);
  const FLOAT = (min, max, step, extra) => Object.assign({ min, max, step }, extra);
  const LOCKED = { locked_on_resume: true };

  const SCHEMA = {
    sections: [
      {
        name: "run", title: "Run",
        fields: [
          field("run.name", "Run name", "str", "run", { advanced: false,
            help: "Outputs go to runs/<name>/. Letters, digits, - and _ only." }),
          field("run.out_dir", "Output folder", "str", "runs", {
            help: "Where runs are written, relative to the repo root unless absolute. The app only lists runs in runs/." }),
          field("run.lexicon", "Word list", "str", "", Object.assign({
            help: "A name from configs/lexicons.toml. Empty uses that file's default." }, LOCKED)),
          field("run.leaves", "Leave values file", "str", "", Object.assign({
            help: "Empty uses data/leaves/<word list id>.leaves; \"zero\" uses no leave values at all." }, LOCKED)),
          field("run.seed", "Seed", "int", 1, INT(0, 2147483647, {
            help: "Fixes the self-play game seeds and all sampling, so a run can be repeated." })),
          field("run.hours", "Time limit", "float", 0, FLOAT(0, 168, 0.5, { advanced: false, unit: "hours",
            help: "Stop after this much wall-clock time. 0 means no time limit." })),
          field("run.max_games", "Game limit", "int", 0, INT(0, 1000000000, { advanced: false, unit: "games",
            help: "Stop after this many self-play games. 0 means no limit." })),
          field("run.device", "Device", "choice", "cuda", { options: ["cuda", "cpu"],
            help: "Where the network runs. Training on the CPU is very much slower." }),
          field("run.vram_fraction", "GPU memory cap", "float", 0.9, FLOAT(0.05, 1, 0.05, {
            help: "Cap on PyTorch's share of GPU memory. Windows does not raise out-of-memory when the GPU fills: it spills into system RAM and slows to a crawl. With the cap, an oversized setting fails straight away instead." })),
        ],
      },
      {
        name: "model", title: "Network",
        fields: [
          field("model.blocks", "Residual blocks", "int", 6, INT(1, 40, Object.assign({ advanced: false,
            help: "Depth of the network. More blocks is slower and needs more games to train. 6x64 scores about 109,000 positions a second; 10x128 about 33,000." }, LOCKED))),
          field("model.channels", "Channels", "int", 64, Object.assign({ min: 8, max: 512, step: 8, multiple_of: 8, advanced: false,
            help: "Width of each block, a multiple of 8. Wider is slower." }, LOCKED)),
          field("model.norm", "Normalization", "choice", "none", Object.assign({ options: ["none", "group"],
            help: "\"none\": no normalization, each block starts as the identity (fastest). \"group\": GroupNorm, about half the speed and 2.7 times the training memory." }, LOCKED)),
          field("model.vector_hidden", "Vector MLP width", "int", 128, INT(8, 2048, Object.assign({
            help: "Width of the MLP over the inputs that are not on the board: the leave, the unseen tiles and the scalars." }, LOCKED))),
          field("model.head_hidden", "Head width", "int", 256, INT(8, 4096, Object.assign({
            help: "Width of the last hidden layer, before the value and win outputs." }, LOCKED))),
          field("model.value_channels", "Value channels", "int", 8, INT(1, 128, Object.assign({
            help: "Per-square channels kept for the head, so it can see where things are on the board." }, LOCKED))),
          field("model.cross_checks", "Cross-check planes", "bool", true, Object.assign({
            help: "Show the network which letters fit on every empty square, given the tiles next to it. This is the only way it sees the dictionary. Turn off only to measure what it is worth." }, LOCKED)),
          field("model.leave_baseline", "Leave baseline", "bool", true, Object.assign({
            help: "The network outputs a correction to the static leave value, so an untrained network already plays like the greedy + leave bot." }, LOCKED)),
        ],
      },
      {
        name: "selfplay", title: "Self-play",
        fields: [
          field("selfplay.games", "Games in lockstep", "int", 256, INT(1, 4096, {
            help: "Games played at the same time, batched on the GPU. Speed is flat from 256 to 512." })),
          field("selfplay.max_candidates", "Candidate moves", "int", 64, INT(2, 512, {
            help: "Moves per decision the network evaluates: the best by score plus leave value. 32 is about 30% faster per game; with fewer, the network can never pick a move the static evaluator ranks low." })),
          field("selfplay.max_exchanges", "Exchange candidates", "int", 4, INT(0, 511, {
            help: "Of those candidates, at most this many are exchanges. Must be below the number of candidate moves." })),
          field("selfplay.temperature_start", "Start temperature", "float", 4.0, FLOAT(0, 50, 0.1, { unit: "points",
            help: "Exploration: moves are sampled from softmax(value / temperature). At 4.0 a move 4 points worse is chosen about 1/e as often as the best." })),
          field("selfplay.temperature_end", "End temperature", "float", 0.5, FLOAT(0, 50, 0.1, { unit: "points",
            help: "The temperature once the decay is over. Raise it if training stalls from too little variety; lower it if late-run games look random." })),
          field("selfplay.temperature_decay_games", "Temperature decay", "int", 300000, INT(1, 100000000, { unit: "games",
            help: "Self-play games over which the temperature falls from start to end." })),
          field("selfplay.win_weight", "Win weight", "float", 0.0, FLOAT(0, 1, 0.05, {
            help: "0 chooses moves by expected spread, 1 by win probability, and values between blend the two. Leave at 0 while training." })),
          field("selfplay.win_scale", "Win scale", "float", 100.0, FLOAT(1, 1000, 1, { unit: "points",
            help: "Points that certain victory is worth in that blend." })),
        ],
      },
      {
        name: "warmstart", title: "Warm start",
        fields: [
          field("warmstart.games", "Warm-start games", "int", 100000, INT(0, 10000000, { unit: "games",
            help: "Games played by the static evaluator before the network plays, to fill the replay buffer. 0 skips the warm start." })),
          field("warmstart.temperature", "Temperature", "float", 6.0, FLOAT(0, 50, 0.5, { unit: "points",
            help: "Exploration in those games, so the network also sees poor moves." })),
          field("warmstart.max_candidates", "Candidate moves", "int", 16, INT(2, 512, {
            help: "Moves per decision the static evaluator samples from during the warm start." })),
          field("warmstart.steps", "Pretraining steps", "int", 2000, INT(0, 1000000, { unit: "steps",
            help: "Training steps on the warm-start games before self-play begins." })),
        ],
      },
      {
        name: "train", title: "Training",
        fields: [
          field("train.batch_size", "Batch size", "int", 1024, INT(1, 65536, { unit: "positions",
            help: "Positions per training step." })),
          field("train.learning_rate", "Learning rate", "float", 0.0003, FLOAT(0.00001, 0.1, 0.00001, {
            help: "Step size. Lower it (0.0003) if the loss jumps around or goes up; this is the first thing to try when a run is unstable." })),
          field("train.warmup_steps", "Warm-up steps", "int", 500, INT(0, 100000, { unit: "steps",
            help: "The learning rate rises linearly from zero over this many steps." })),
          field("train.weight_decay", "Weight decay", "float", 0.0001, FLOAT(0, 0.1, 0.00001, {
            help: "How strongly the weights are pulled towards zero at each step." })),
          field("train.grad_clip", "Gradient clip", "float", 1.0, FLOAT(0, 100, 0.1, {
            help: "Maximum gradient norm. 0 turns clipping off." })),
          field("train.value_scale", "Value scale", "float", 50.0, FLOAT(1, 1000, 1, { unit: "points",
            help: "Points per unit of the value output." })),
          field("train.huber_delta", "Huber delta", "float", 2.0, FLOAT(0.1, 100, 0.1, {
            help: "The value loss is quadratic within this many units (100 points at the default scale) and linear beyond." })),
          field("train.win_loss_weight", "Win loss weight", "float", 0.5, FLOAT(0, 10, 0.1, {
            help: "Weight of the win-probability loss relative to the value loss." })),
          field("train.td_lambda", "TD lambda", "float", 0.7, FLOAT(0, 1, 0.05, {
            help: "1 trains on the spread actually gained from each position on. Below 1, targets lean on the network's own estimate of the next position: less noise, more bias. Try 0.8 to 0.9 once a first run works." })),
          field("train.replay_capacity", "Replay capacity", "int", 3000000, INT(10000, 100000000, { unit: "positions",
            help: "Positions kept in RAM, about 310 bytes each: 3,000,000 is 0.9 GB and 10,000,000 about 3 GB. Raise it to train on a longer history." })),
          field("train.min_replay", "Minimum replay", "int", 100000, INT(1, 100000000, { unit: "positions",
            help: "Positions needed before training starts. At least the batch size and at most the replay capacity." })),
          field("train.replay_ratio", "Replay ratio", "float", 1.0, FLOAT(0.1, 64, 0.5, {
            help: "Training samples drawn per new self-play position. Higher learns more from each game, slows self-play and risks overfitting old data." })),
          field("train.actor_sync_steps", "Weight sync interval", "int", 100, INT(1, 100000, { unit: "steps",
            help: "Training steps between copies of the weights into the self-play network." })),
          field("train.augment_transpose", "Transpose augmentation", "bool", true, {
            help: "Randomly swap rows and columns: the one exact symmetry of the board." }),
          field("train.amp", "Mixed precision", "bool", true, {
            help: "Train with bf16 mixed precision on the GPU." }),
          field("train.checkpoint_minutes", "Checkpoint interval", "float", 10.0, FLOAT(1, 600, 1, { unit: "minutes",
            help: "How often to save a checkpoint. Killing the run loses at most this much work." })),
          field("train.eval_minutes", "Evaluation interval", "float", 20.0, FLOAT(1, 600, 1, { unit: "minutes",
            help: "How often to evaluate against the baseline bots and gate against the best network so far." })),
          field("train.eval_pairs", "Evaluation pairs", "int", 250, INT(2, 100000, { unit: "pairs",
            help: "Seat-swapped pairs per baseline in each evaluation. More pairs narrow the interval and cost a few seconds each time." })),
          field("train.gate_pairs", "Gate pairs", "int", 500, INT(2, 100000, { unit: "pairs",
            help: "Pairs when testing a checkpoint against the best network so far. best.pt is replaced only when the new network wins with an interval that excludes zero." })),
          field("train.keep_checkpoints", "Checkpoints kept", "int", 5, INT(1, 100, {
            help: "step_*.pt files kept, about 20 MB each. best.pt is always kept." })),
          field("train.guard", "Guard", "bool", true, {
            help: "When a network is significantly behind the best one for several evaluations in a row, go back to the best network's weights and halve the learning rate." }),
          field("train.guard_patience", "Guard patience", "int", 2, INT(1, 20, { unit: "evaluations",
            help: "Evaluations in a row significantly behind the best before the guard acts." })),
        ],
      },
    ],
    presets: [
      { name: "Default 6x64", description: "About 55 games a second. A good first run.", overrides: {} },
      { name: "Larger 10x128", description: "About 18 games a second. Try it once the small network has learned something.",
        overrides: { "model.blocks": 10, "model.channels": 128 } },
      { name: "Quick test (30 minutes)", description: "Checks that everything works: a short warm start and an evaluation every 10 minutes.",
        overrides: { "run.hours": 0.5, "warmstart.games": 20000, "warmstart.steps": 500,
          "train.eval_minutes": 10, "train.checkpoint_minutes": 5 } },
      { name: "Wide 12x192", description: "About 8.5 games a second and 7.1 GB of GPU memory. For long runs.",
        overrides: { "model.blocks": 12, "model.channels": 192 } },
    ],
    speed: [
      { network: "6x64", parameters: 1055056, games_per_second: 55, measured: true, vram_gb: 1.4 },
      { network: "10x128", parameters: 3655316, games_per_second: 18, measured: false, vram_gb: 4.1 },
      { network: "12x192", parameters: 8763412, games_per_second: 8.5, measured: false, vram_gb: 7.1 },
      { network: "16x192", parameters: 11417876, games_per_second: 6, measured: false, vram_gb: 9.1 },
      { network: "20x256", parameters: 24486868, games_per_second: 3.4, measured: false, vram_gb: 7.6 },
    ],
  };

  // What turning each setting down or up does, and the range worth trying (`effects` in
  // docs/APP_API.md). For an on/off setting `lower` is off and `higher` on; for a choice, `lower`
  // describes the first option and `higher` the second. The real text comes from the server.
  const EFFECTS = {
    "run.lexicon": ["", "", "Leave empty for the default word list. It cannot change once the run has started."],
    "run.leaves": ["", "", "Leave empty for the fitted leave values of the word list."],
    "run.seed": ["No effect on strength: a different seed gives different games, not better ones.",
      "No effect on strength either. Change it to repeat an experiment on fresh games.",
      "Any whole number. Keep it fixed when comparing runs that differ in one setting."],
    "run.hours": ["A shorter run: fewer games, and the network may stop before it has learned anything.",
      "More games and usually a stronger network, until it levels off. The GPU is busy for longer.",
      "0.5 hours for a quick check, 6 to 10 overnight."],
    "run.max_games": ["Stops sooner, after fewer self-play games.",
      "Lets the run play more games before it stops. 0 means no game limit.",
      "0 (let the time limit decide), or 400,000 to 2,000,000 games."],
    "run.device": ["Trains on the GPU, about fifty times faster.", "Runs without a GPU: far too slow for real training.", "cuda."],
    "run.vram_fraction": ["Leaves more GPU memory to other programs; an oversized setting fails sooner with an out-of-memory error.",
      "Lets the run use more of the GPU. Near 1, Windows may spill into system RAM and crawl instead of failing.",
      "0.8 to 0.9."],
    "model.blocks": ["A shallower, faster network: more games an hour, less it can learn.",
      "A deeper network that can learn more, but slower, and it needs many more games before it pays off.",
      "6 for a first run; 10 to 12 for long runs."],
    "model.channels": ["A narrower, faster network with less capacity.",
      "A wider network: more capacity, slower self-play, more GPU memory.",
      "64 to 128, a multiple of 8."],
    "model.norm": ["No normalization: the fastest, each block starting as the identity.",
      "GroupNorm: steadier for deep networks, about half the speed and 2.7 times the memory.",
      "none."],
    "model.vector_hidden": ["Less capacity for the leave, the unseen tiles and the scores; slightly faster.",
      "More capacity for those inputs, at little cost in speed.", "64 to 256."],
    "model.head_hidden": ["A smaller last layer: slightly faster, less room to combine what the network sees.",
      "A larger last layer: more capacity, and more room to memorise games.", "128 to 512."],
    "model.value_channels": ["The head sees less of where things are on the board.",
      "The head keeps more detail per square; a little slower.", "4 to 16."],
    "model.cross_checks": ["The network cannot see which letters fit where, and plays much worse.",
      "The network sees the dictionary through the cross-checks. Keep it on.", "On."],
    "model.leave_baseline": ["The network learns the whole value from scratch and starts far below the greedy + leave bot.",
      "The network learns a correction to the leave values, so it starts level with greedy + leave.", "On."],
    "selfplay.games": ["Fewer games at a time: smaller GPU batches, fewer games a second.",
      "More games at a time: more games a second up to about 512, more memory.", "256 to 512."],
    "selfplay.max_candidates": ["Faster games, but the network can only choose among the moves the static evaluator likes.",
      "The network can find moves the static evaluator ranks low, at the cost of slower games.", "32 to 64."],
    "selfplay.max_exchanges": ["Fewer exchanges considered: the network rarely learns when exchanging pays.",
      "More exchange options per decision, taking candidate places from plays.", "2 to 8."],
    "selfplay.temperature_start": ["Less exploration early: games stay close to the network's own choices and show it fewer alternatives.",
      "More random early games: more variety, noisier targets.", "2 to 6 points."],
    "selfplay.temperature_end": ["Late games follow the network's choices closely: sharper targets, less variety, more risk of memorising.",
      "Keeps variety late in the run; games look more random.", "0.3 to 1 point."],
    "selfplay.temperature_decay_games": ["The temperature reaches its end value sooner.", "Exploration lasts further into the run.",
      "200,000 to 1,000,000 games."],
    "selfplay.win_weight": ["Moves are chosen by expected spread, the right target while training.",
      "Leans towards win probability: safer when ahead, riskier when behind.", "0 while training."],
    "selfplay.win_scale": ["A certain win is worth fewer points in the blend, so spread dominates.",
      "A certain win is worth more, so the blend leans further towards winning.",
      "50 to 200 points; it matters only when the win weight is above 0."],
    "warmstart.games": ["A shorter warm start: self-play begins sooner with less in the replay buffer.",
      "A fuller replay buffer before the network plays, and a longer wait.", "50,000 to 200,000 games."],
    "warmstart.temperature": ["Warm-start games closer to the static evaluator's best play.",
      "More varied warm-start games, so the network also sees poor moves.", "4 to 8 points."],
    "warmstart.max_candidates": ["The static evaluator samples from fewer moves: less variety.",
      "More variety in warm-start play; slower games.", "8 to 32."],
    "warmstart.steps": ["Self-play starts with a less trained network.",
      "More pretraining on the warm-start games; too much and the network fits them too closely.", "1,000 to 5,000 steps."],
    "train.batch_size": ["Noisier, cheaper steps, and more of them per game.",
      "Smoother steps and more GPU memory; may want a higher learning rate.", "1,024 to 4,096 positions."],
    "train.learning_rate": ["Steadier, slower learning. The first thing to try when the loss jumps or evaluations fall.",
      "Faster learning, with more risk of instability and of the network getting worse.", "0.0003 to 0.001."],
    "train.warmup_steps": ["Reaches the full learning rate sooner; a fresh network can lurch at the start.",
      "A gentler start, a little slower to get going.", "200 to 2,000 steps."],
    "train.weight_decay": ["Weights are free to grow: more risk of memorising the replay buffer.",
      "A stronger pull towards small weights: less memorising, but too much limits what it can learn.", "0.00005 to 0.0005."],
    "train.grad_clip": ["Clips more steps: steadier, slower learning. 0 turns clipping off entirely.",
      "Clips only extreme steps, so large updates get through.", "0.5 to 2."],
    "train.value_scale": ["Each unit of the value output is fewer points, so the loss sees spreads as larger.",
      "Each unit is more points, so small spreads look smaller to the loss.", "50 points. Fixed for the life of a run."],
    "train.huber_delta": ["Large errors count for less: robust to blowouts, slower to fit big spreads.",
      "Closer to squared error: fits big spreads harder, more sensitive to freak games.", "1 to 4."],
    "train.win_loss_weight": ["The win output trains less and the value output gets more of the network.",
      "The win output matters more relative to the spread.", "0.25 to 1."],
    "train.td_lambda": ["Targets lean on the network's own estimates: less noise, more bias.",
      "Targets follow what really happened in the game: unbiased but noisy. 1 is the pure result.", "0.8 to 1."],
    "train.replay_capacity": ["Trains on recent games only: adapts quickly, memorises sooner.",
      "Trains on a longer history: less memorising, more RAM (about 310 bytes a position).", "2,000,000 to 10,000,000 positions."],
    "train.min_replay": ["Training starts sooner, on fewer positions.", "Waits for more positions before the first step.",
      "50,000 to 500,000 positions."],
    "train.replay_ratio": ["Each position is trained on fewer times: less memorising, slower learning per game.",
      "Learns more from each game, but slows self-play and risks memorising the buffer. Lower it if the error on unseen games rises.",
      "2 to 8."],
    "train.actor_sync_steps": ["Self-play uses the newest weights more often.", "Self-play lags further behind training; slightly faster.",
      "100 to 1,000 steps."],
    "train.augment_transpose": ["Half as much variety in the positions trained on.",
      "Uses the board's one exact symmetry for free extra data. Keep it on.", "On."],
    "train.amp": ["Full precision: about twice as slow.", "bf16 mixed precision: faster, with no loss of strength.", "On."],
    "train.checkpoint_minutes": ["Saves more often, so a crash loses less.", "Saves less often; a crash or a force stop loses more training.",
      "5 to 30 minutes."],
    "train.eval_minutes": ["Evaluations more often: a finer strength curve, a little less training.",
      "Fewer evaluations: more time training, a coarser picture.", "10 to 30 minutes."],
    "train.eval_pairs": ["Faster evaluations with wider intervals.", "Narrower intervals, so small gains can be told from noise.",
      "250 to 1,000 pairs."],
    "train.gate_pairs": ["Faster gates, but a new best network is kept on weaker evidence.",
      "A new best network has to win a longer match to be kept.", "250 to 1,000 pairs."],
    "train.keep_checkpoints": ["Fewer step files on disk.", "More older networks to compare against, about 20 MB each.", "3 to 10."],
    // On/off and choice sentences may carry their own label, as the server's do; the page shows it once.
    "train.guard": ["Off: nothing stops a run drifting far below its best (best.pt is still kept).",
      "On: a run can waste time but cannot end far worse than it started.", "On."],
    "train.guard_patience": ["1 reacts at once, and to the occasional unlucky evaluation too.", "Waits longer before going back.", "2 to 3."],
  };
  if (!search.has("old")) {
    for (const section of SCHEMA.sections) {
      for (const f of section.fields) {
        const [lower, higher, typical] = EFFECTS[f.key] || ["", "", ""];
        f.effects = { lower, higher, typical };
      }
    }
  }

  const FIELDS = {};
  for (const section of SCHEMA.sections) for (const f of section.fields) FIELDS[f.key] = f;
  const PARAMS = Object.fromEntries(SCHEMA.speed.map((row) => [row.network, row.parameters]));

  M.on("GET", /^\/api\/config\/schema$/, () => SCHEMA);

  // ---- Runs (a fallback, used only if js/mock/training.js does not provide runs) ---------

  const ownRuns = !M.has("GET", "/api/runs");

  /** Evaluation records for a run, rising towards `final` with noise. */
  function evalRecords(spec) {
    const rand = random(spec.seed);
    const records = [];
    const count = Math.max(1, Math.round(spec.hours * 3)); // every 20 minutes
    for (let i = 1; i <= count; i += 1) {
      const t = i / count;
      const games = Math.round(spec.games * t);
      const hours = spec.hours * t;
      const step = Math.round(spec.step * t);
      const base = spec.start + (spec.final - spec.start) * (1 - Math.exp(-3.2 * t)) / (1 - Math.exp(-3.2));
      const noise = (rand() - 0.5) * 5;
      const ci = 7.4 + rand() * 0.6;
      const spread = i === count ? spec.final : base + noise;
      const common = { kind: "eval", time: spec.created + hours * HOUR, step, games, positions: games * 57, hours, pairs: 250 };
      records.push(Object.assign({}, common, { opponent: "greedy-leave", spread, spread_ci: ci,
        win_rate: 0.5 + spread / 220, win_ci: 0.044, score_for: 420 + spread / 2, score_against: 420 - spread / 2 }));
      const g = spec.greedy + (rand() - 0.5) * 4 - (1 - t) * 6;
      const gs = i === count ? spec.greedy : g;
      records.push(Object.assign({}, common, { opponent: "greedy", spread: gs, spread_ci: 8.6 + rand() * 0.8,
        win_rate: 0.5 + gs / 220, win_ci: 0.042, score_for: 430 + gs / 2, score_against: 430 - gs / 2 }));
    }
    return records;
  }

  function makeRun(spec) {
    const records = spec.games > 0 ? evalRecords(spec) : [];
    const last = (opponent) => {
      const list = records.filter((r) => r.opponent === opponent);
      const r = list[list.length - 1];
      return r ? { spread: r.spread, ci: r.spread_ci, win_rate: r.win_rate } : null;
    };
    const latest = records.length ? {
      games: records[records.length - 1].games, "greedy-leave": last("greedy-leave"), greedy: last("greedy"),
    } : null;
    const limit = { hours: spec.limitHours || 0, max_games: spec.limitGames || 0 };
    const progress = limit.hours ? Math.min(1, spec.hours / limit.hours) : limit.max_games ? Math.min(1, spec.games / limit.max_games) : null;
    return {
      summary: {
        name: spec.name, status: spec.status, phase: spec.phase || (spec.games > 0 ? "selfplay" : null),
        network: spec.network, parameters: PARAMS[spec.network] || null,
        games: spec.games, step: spec.step, positions: spec.games * 57, hours: spec.hours,
        games_per_second: spec.games && spec.hours ? spec.rate || spec.games / (spec.hours * HOUR) : null,
        created: spec.created, updated: spec.updated, limit, progress,
        latest_eval: latest, best_step: spec.games > 0 ? Math.round(spec.step * 0.9) : null,
        has_best: spec.games > 0 && spec.status !== "failed", checkpoints: spec.checkpoints ?? (spec.games > 0 ? 5 : 0),
        changed: spec.changed || {}, error: spec.error || null,
      },
      records,
    };
  }

  const RUNS = [
    { name: "lr-0003", status: "running", network: "6x64", games: 412300, step: 27400, hours: 2.07, rate: 55.3,
      created: NOW - 2.1 * HOUR, updated: NOW - 12, limitHours: 8, start: -3, final: 3.1, greedy: 40.2, seed: 11,
      changed: { "run.hours": 8, "train.learning_rate": 0.0003 } },
    { name: "lr-0003-wide", status: "queued", network: "10x128", games: 0, step: 0, hours: 0, created: NOW - 2.1 * HOUR,
      updated: NOW - 2.1 * HOUR, limitHours: 8, checkpoints: 0,
      changed: { "run.hours": 8, "train.learning_rate": 0.0003, "model.blocks": 10, "model.channels": 128 } },
    { name: "night2-10x128", status: "finished", network: "10x128", games: 512400, step: 33800, hours: 8.0,
      created: NOW - 30 * HOUR, updated: NOW - 22 * HOUR, limitHours: 8, start: -4, final: 9.6, greedy: 47.1, seed: 5,
      changed: { "run.hours": 8, "model.blocks": 10, "model.channels": 128 } },
    { name: "td-08", status: "stopped", network: "6x64", games: 221800, step: 14900, hours: 1.12,
      created: NOW - 50 * HOUR, updated: NOW - 48.8 * HOUR, limitHours: 8, start: -3, final: 1.2, greedy: 36.0, seed: 8,
      changed: { "run.hours": 8, "train.td_lambda": 0.8 } },
    { name: "night1", status: "finished", network: "6x64", games: 1612000, step: 71000, hours: 8.0,
      created: NOW - 76 * HOUR, updated: NOW - 68 * HOUR, limitHours: 8, start: -3, final: 8.1, greedy: 44.0, seed: 3,
      changed: { "run.hours": 8 } },
    { name: "wide-12x192", status: "failed", network: "12x192", games: 2900, step: 0, hours: 0.07, phase: "warmstart",
      created: NOW - 80 * HOUR, updated: NOW - 79.9 * HOUR, limitHours: 8, checkpoints: 0,
      changed: { "run.hours": 8, "model.blocks": 12, "model.channels": 192, "selfplay.games": 1024 },
      error: "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 1.17 GiB. GPU 0 has a total capacity of 15.92 GiB, of which the 90% cap (run.vram_fraction) was reached." },
    { name: "csw-test", status: "stopped", network: "6x64", games: 96000, step: 6100, hours: 0.5,
      created: NOW - 140 * HOUR, updated: NOW - 139.5 * HOUR, limitHours: 0.5, start: -3, final: 2.2, greedy: 31.0, seed: 21,
      changed: { "run.hours": 0.5, "run.lexicon": "CSW21" } },
    { name: "smoke-1105", status: "finished", network: "6x64", games: 98000, step: 6000, hours: 0.5,
      created: NOW - 170 * HOUR, updated: NOW - 169.5 * HOUR, limitHours: 0.5, start: -14, final: -9.8, greedy: 25.4, seed: 2,
      changed: { "run.hours": 0.5, "warmstart.games": 20000, "warmstart.steps": 500, "train.eval_minutes": 10, "train.checkpoint_minutes": 5 } },
  ].map(makeRun);

  const findRun = (name) => RUNS.find((r) => r.summary.name === name);
  const takenNames = new Set(RUNS.map((r) => r.summary.name.toLowerCase()));

  if (ownRuns) {
    M.on("GET", /^\/api\/runs$/, () => {
      for (const run of RUNS) {
        const s = run.summary;
        if (s.status === "running") {
          s.updated = Date.now() / 1000 - 3;
          s.games += Math.round(55 * 5);
        }
        if (s.status === "stopping") s.status = "stopped";
      }
      return { runs: RUNS.map((r) => r.summary) };
    });
  }
  if (!M.has("GET", "/api/status")) {
    M.on("GET", /^\/api\/status$/, () => ({
      version: "0.1.0",
      lexicon: { id: "NSWL2023-91a67230e1f6", words: 195747 },
      leaves: { ok: true, file: "NSWL2023-91a67230e1f6.leaves" },
      gpu: { name: "NVIDIA GeForce RTX 5070 Ti", utilization: 87, memory_used_mb: 2140, memory_total_mb: 16303, temperature_c: 61 },
      active_runs: RUNS.filter((r) => r.summary.status === "running").map((r) => r.summary.name),
      queued_runs: RUNS.filter((r) => r.summary.status === "queued").map((r) => r.summary.name),
      active_matches: [],
    }));
  }
  if (!M.has("GET", "/api/runs/x/metrics")) {
    M.on("GET", /^\/api\/runs\/([^/]+)\/metrics$/, (match) => {
      const run = findRun(decodeURIComponent(match[1]));
      if (!run) return fail("no run named " + decodeURIComponent(match[1]));
      return { records: run.records, next: run.records.length };
    });
  }
  if (!M.has("POST", "/api/runs/x/stop")) {
    M.on("POST", /^\/api\/runs\/([^/]+)\/stop$/, (match) => {
      const run = findRun(decodeURIComponent(match[1]));
      if (run) run.summary.status = run.summary.status === "queued" ? "stopped" : "stopping";
      return { ok: true };
    });
  }
  if (!M.has("POST", "/api/runs/x/resume")) {
    M.on("POST", /^\/api\/runs\/([^/]+)\/resume$/, (match, body) => {
      const run = findRun(decodeURIComponent(match[1]));
      if (!run) return fail("no run named " + decodeURIComponent(match[1]));
      const s = run.summary;
      if (s.status === "running" || s.status === "queued") return fail(`${s.name} is already ${s.status}`);
      if (!s.checkpoints) return fail(`${s.name} has no checkpoint to resume from`);
      for (const key of Object.keys((body && body.overrides) || {})) {
        if (!FIELDS[key]) return fail(`unknown setting ${key}`);
        if (FIELDS[key].locked_on_resume) return fail(`${key} cannot be changed when resuming`);
      }
      s.status = "running";
      s.updated = Date.now() / 1000;
      s.limit = { hours: (body && body.hours) || 0, max_games: (body && body.max_games) || 0 };
      s.hours = 0;
      Object.assign(s.changed, (body && body.overrides) || {});
      return s;
    });
  }
  if (!M.has("DELETE", "/api/runs/x")) {
    M.on("DELETE", /^\/api\/runs\/([^/]+)$/, (match) => {
      const name = decodeURIComponent(match[1]);
      const index = RUNS.findIndex((r) => r.summary.name === name);
      if (index >= 0 && ["running", "stopping"].includes(RUNS[index].summary.status)) return fail(`${name} is running; stop it first`);
      if (index >= 0) RUNS.splice(index, 1);
      takenNames.delete(name.toLowerCase());
      return { ok: true };
    });
  }

  if (!M.has("POST", "/api/runs/x/rename")) {
    M.on("POST", /^\/api\/runs\/([^/]+)\/rename$/, (match, body) => {
      const name = decodeURIComponent(match[1]);
      const run = findRun(name);
      if (!run) return fail("no run named " + name);
      const to = String((body && body.name) || "").trim();
      if (["running", "stopping", "queued"].includes(run.summary.status)) return fail(`${name} is ${run.summary.status}; it cannot be renamed now`);
      if (!/^[A-Za-z0-9_-]+$/.test(to)) return fail(`invalid run name '${to}': use letters, digits, - and _`);
      if (to === name) return fail(`${name} already has that name`);
      if (takenNames.has(to.toLowerCase()) && to.toLowerCase() !== name.toLowerCase()) return fail(`a run named '${to}' already exists`);
      takenNames.delete(name.toLowerCase());
      takenNames.add(to.toLowerCase());
      run.summary.name = to;
      App.emit("mock:rename", { from: name, to });
      return run.summary;
    });
  }

  /** The checks of Config.validate() that a form can trip, in the server's words. */
  function validateConfig(values) {
    const v = (key) => values[key];
    const checks = [
      [v("model.channels") % 8 === 0, "model.channels must be a multiple of 8"],
      [v("selfplay.max_exchanges") < v("selfplay.max_candidates"), "selfplay.max_exchanges must be below max_candidates"],
      [v("train.replay_capacity") >= v("train.min_replay") && v("train.min_replay") >= v("train.batch_size"),
        "need replay_capacity >= min_replay >= batch_size"],
      [v("selfplay.win_weight") >= 0 && v("selfplay.win_weight") <= 1, "selfplay.win_weight must be between 0 and 1"],
      [v("train.td_lambda") >= 0 && v("train.td_lambda") <= 1, "train.td_lambda must be between 0 and 1"],
      [v("run.vram_fraction") > 0 && v("run.vram_fraction") <= 1, "run.vram_fraction must be in (0, 1]"],
    ];
    const failed = checks.find(([ok]) => !ok);
    return failed ? failed[1] : null;
  }

  if (!M.has("POST", "/api/runs")) {
    M.on("POST", /^\/api\/runs$/, (match, body) => {
      const list = (body && body.runs) || [];
      if (!list.length) return fail("no runs given");
      const seen = new Set();
      for (const item of list) {
        const name = String(item.name || "");
        if (!/^[A-Za-z0-9_-]+$/.test(name)) return fail(`invalid run name '${name}': use letters, digits, - and _`);
        if (takenNames.has(name.toLowerCase()) || seen.has(name.toLowerCase())) return fail(`a run named '${name}' already exists`);
        seen.add(name.toLowerCase());
        const values = {};
        for (const key of Object.keys(FIELDS)) values[key] = FIELDS[key].default;
        for (const [key, value] of Object.entries(item.overrides || {})) {
          if (!FIELDS[key]) {
            const section = key.split(".")[0];
            const known = Object.keys(FIELDS).filter((k) => k.startsWith(section + ".")).map((k) => k.split(".")[1]);
            return fail(`unknown setting ${key}; [${section}] has: ${known.join(", ")}`);
          }
          values[key] = value;
        }
        const problem = validateConfig(values);
        if (problem) return fail(`${name}: ${problem}`);
      }
      const anyActive = RUNS.some((r) => ["running", "queued", "stopping"].includes(r.summary.status));
      const started = [];
      const queued = [];
      list.forEach((item, i) => {
        const startNow = body.mode === "parallel" || (i === 0 && !anyActive);
        (startNow ? started : queued).push(item.name);
        takenNames.add(item.name.toLowerCase());
        if (ownRuns) {
          const o = item.overrides || {};
          const network = `${o["model.blocks"] || FIELDS["model.blocks"].default}x${o["model.channels"] || FIELDS["model.channels"].default}`;
          const run = makeRun({ name: item.name, status: startNow ? "running" : "queued", network, games: 0, step: 0,
            hours: 0, created: Date.now() / 1000, updated: Date.now() / 1000, limitHours: o["run.hours"] || 0,
            limitGames: o["run.max_games"] || 0, checkpoints: 0, changed: o, phase: startNow ? "warmstart" : null });
          run.summary.progress = run.summary.limit.hours || run.summary.limit.max_games ? 0 : null;
          RUNS.unshift(run);
        }
      });
      return { started, queued };
    });
  }

  // ---- Model library ---------------------------------------------------------------------

  const BOTS = [
    { spec: "bot:greedy-leave", name: "Greedy + leave", description: "Highest score plus leave value." },
    { spec: "bot:greedy", name: "Greedy", description: "Highest score." },
    { spec: "bot:random", name: "Random", description: "A random legal placement." },
  ];

  function model(run, file, network, step, games, ageHours, gl, g, extra) {
    const kind = file === "best.pt" ? "best" : "checkpoint";
    const params = PARAMS[network] || null;
    return Object.assign({
      id: `${run}/${file}`, spec: `model:${run}/${file}`,
      name: kind === "best" ? `${run} (best)` : `${run} (step ${step.toLocaleString("en-US")})`,
      run, file, kind, network, parameters: params, step, games,
      modified: NOW - ageHours * HOUR, size_bytes: kind === "best" ? Math.round(params * 4.08) : Math.round(params * 12.3),
      compatible: true,
      eval: gl ? { "greedy-leave": { spread: gl[0], ci: gl[1], win_rate: 0.5 + gl[0] / 220 },
        greedy: { spread: g[0], ci: g[1], win_rate: 0.5 + g[0] / 220 } } : null,
      starred: false, note: "",
    }, extra);
  }

  // Run names match js/mock/training.js (night2 finished, night3 running, dryrun an old test).
  const MODELS = [
    model("night2", "best.pt", "6x64", 61000, 1402000, 9.2, [8.1, 4.1], [44.0, 4.4],
      { starred: true, note: "First to beat greedy-leave" }),
    model("night2", "step_000071000.pt", "6x64", 71000, 1612000, 9, [5.2, 4.2], [42.3, 4.5]),
    model("night2", "step_000069500.pt", "6x64", 69500, 1585000, 9.1, [6.0, 4.1], [43.1, 4.5]),
    model("night2", "step_000068000.pt", "6x64", 68000, 1551000, 9.3, [4.4, 4.2], [41.7, 4.4]),
    model("night3", "best.pt", "6x64", 24800, 560000, 0.4, [5.3, 7.8], [40.9, 9.0], { starred: true }),
    model("night3", "step_000027400.pt", "6x64", 27400, 612000, 0.1, [3.1, 7.6], [40.2, 9.1]),
    model("night3", "step_000025900.pt", "6x64", 25900, 583000, 0.3, [4.0, 7.7], [39.8, 9.0]),
    model("dryrun", "best.pt", "6x64", 1100, 21000, 50, [-1.8, 7.9], [28.5, 9.4],
      { incompatible_reason: "Trained with leave values that have since been refitted, so its values no longer mean the same thing." }),
    // wide2 got worse as it trained (js/mock/training.js): its best is the first network evaluated.
    model("wide2", "best.pt", "10x128", 4100, 23300, 3.1, [-7.8, 7.0], [33.1, 8.2]),
    model("wide2", "step_000024100.pt", "10x128", 24100, 460800, 3, [-71.3, 8.4], [-30.2, 8.6]),
  ].map((m) => (m.incompatible_reason ? Object.assign(m, { compatible: false }) : m));

  // A run renamed in js/mock/training.js renames its models (they are named after their run).
  App.on("mock:rename", ({ from, to }) => {
    for (const m of MODELS) {
      if (m.run !== from) continue;
      m.run = to;
      m.id = `${to}/${m.file}`;
      m.spec = `model:${m.id}`;
      m.name = m.kind === "best" ? `${to} (best)` : `${to} (step ${m.step.toLocaleString("en-US")})`;
    }
  });

  function manyModels() {
    const rand = random(99);
    const nets = ["6x64", "10x128", "12x192", "6x64", "16x192", "6x64", "10x128", "20x256"];
    const out = [];
    for (let r = 0; r < 8; r += 1) {
      const run = ["sweep", "lr", "td", "wide", "deep", "temp", "night", "big"][r] + "-" + (r + 3);
      for (let k = 0; k < 5; k += 1) {
        const best = k === 0;
        const step = 20000 + Math.round(rand() * 50000);
        const gl = -4 + rand() * 16;
        out.push(model(run, best ? "best.pt" : `step_0000${step}.pt`, nets[r], step, step * 22, 10 + r * 20 + k,
          [gl, 7.3 + rand()], [30 + gl * 2 + rand() * 6, 8.5 + rand()], { starred: best && rand() > 0.7 }));
      }
    }
    return out.concat(MODELS);
  }

  M.on("GET", /^\/api\/models$/, (match, body, query) => {
    if (query.demo === "empty") return { bots: BOTS, models: [] };
    return { bots: BOTS, models: query.demo === "many" ? manyModels() : MODELS };
  });

  M.on("POST", /^\/api\/models\/(.+)\/meta$/, (match, body) => {
    const id = match[1].split("/").map(decodeURIComponent).join("/");
    const found = MODELS.find((m) => m.id === id);
    if (!found) return fail("no model " + id);
    if (body && "starred" in body) found.starred = !!body.starred;
    if (body && "note" in body) found.note = String(body.note || "");
    return found;
  });

  // ---- Leave values ----------------------------------------------------------------------

  const TILE_LEAVE = { "?": 24.5, S: 8.1, Z: 3.2, X: 3.6, R: 1.6, E: 2.4, A: 0.6, I: -0.9, O: -1.4, U: -3.3, N: 0.7,
    T: 0.4, L: 0.1, D: 0.2, H: 1.1, C: 0.6, M: 0.5, P: -0.3, B: -2.1, F: -2.0, G: -2.6, K: -2.4, W: -3.6,
    V: -5.6, Y: -0.4, J: -1.4, Q: -6.9 };

  M.on("GET", /^\/api\/leaves$/, (match, body, query) => {
    const rack = String(query.rack || "").toUpperCase();
    if (!/^[A-Z?]{0,7}$/.test(rack)) return fail("a leave is up to 7 letters or ?");
    const bag = query.bag === undefined ? 60 : Number(query.bag);
    const scale = 0.45 + 0.55 * Math.min(1, Math.max(0, bag) / 60);
    let raw = 0;
    const counts = {};
    for (const ch of rack) {
      raw += TILE_LEAVE[ch] || 0;
      counts[ch] = (counts[ch] || 0) + 1;
    }
    for (const [ch, n] of Object.entries(counts)) if (n > 1 && ch !== "?") raw -= (n - 1) * (/[AEIOU]/.test(ch) ? 5.5 : 4);
    if (counts.Q && !counts.U) raw -= 4.5;
    const vowels = [...rack].filter((c) => /[AEIOU]/.test(c)).length;
    const consonants = [...rack].filter((c) => /[B-DF-HJ-NP-TV-Z]/.test(c)).length;
    if (rack.length >= 3) raw -= Math.abs(vowels - consonants * 0.75) * 1.6;
    const empty = -24.0 * scale;
    const relative = rack.length ? raw * scale : 0;
    return { rack, bag, value: Math.round((empty + relative) * 10) / 10, relative: Math.round(relative * 10) / 10 };
  });
})();
