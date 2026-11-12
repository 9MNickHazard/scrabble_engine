//! Game state and the rules for a turn (docs/RULES.md).
//!
//! `Game` is plain data, about 400 bytes, so cloning it is cheap. It holds no lexicon: move
//! generation and validation take the GADDAG as an argument.

use crate::board::{square_tile, Board, EMPTY};
use crate::gaddag::Gaddag;
use crate::movegen::{generate_exchanges, generate_plays};
use crate::moves::{Move, Play};
use crate::rng::Rng;
use crate::tiles::{Rack, RACK_SIZE, TILE_COUNTS, TILE_TYPES, TOTAL_TILES};

/// Tiles needed in the bag before an exchange is allowed.
pub const EXCHANGE_MINIMUM: usize = 7;
/// Consecutive passes and exchanges that end the game.
pub const SCORELESS_LIMIT: u8 = 6;

/// The bag. Its order is fixed when the game is created; draws take from the end of `tiles`.
#[derive(Clone, PartialEq, Eq)]
pub struct Bag {
    tiles: [u8; TOTAL_TILES],
    len: u8,
}

impl Bag {
    /// A bag holding `tiles`, drawn from the last element backwards.
    pub fn from_tiles(tiles: &[u8]) -> Bag {
        assert!(tiles.len() <= TOTAL_TILES);
        let mut bag = Bag { tiles: [0; TOTAL_TILES], len: tiles.len() as u8 };
        bag.tiles[..tiles.len()].copy_from_slice(tiles);
        bag
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.len as usize
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[inline]
    fn draw(&mut self) -> Option<u8> {
        (self.len > 0).then(|| {
            self.len -= 1;
            self.tiles[self.len as usize]
        })
    }

    /// Put a tile back at a uniformly random position.
    fn insert(&mut self, tile: u8, rng: &mut Rng) {
        let position = rng.below(self.len as u32 + 1) as usize;
        self.tiles[self.len as usize] = self.tiles[position];
        self.tiles[position] = tile;
        self.len += 1;
    }

    /// The tiles still in the bag, next draw last.
    pub fn tiles(&self) -> &[u8] {
        &self.tiles[..self.len as usize]
    }

    pub fn counts(&self) -> [u8; TILE_TYPES] {
        let mut counts = [0; TILE_TYPES];
        for &tile in self.tiles() {
            counts[tile as usize] += 1;
        }
        counts
    }
}

/// The public facts about a player's most recent turn. The opponent's last turn hints at what
/// they are holding (an exchange of five tiles says something a 70-point bingo does not), so it
/// is part of what the network is shown.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct LastTurn {
    /// 0 no turn yet, 1 placement, 2 exchange, 3 pass.
    pub kind: u8,
    /// Tiles placed or exchanged.
    pub tiles: u8,
    pub score: i16,
}

/// What a turn did to the scores.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct Applied {
    /// Points the move itself scored.
    pub score: i32,
    /// End-of-game changes to each player's score (zero unless this turn ended the game).
    pub adjustment: [i32; 2],
    pub game_over: bool,
}

#[derive(Clone)]
pub struct Game {
    pub board: Board,
    pub racks: [Rack; 2],
    pub bag: Bag,
    pub scores: [i32; 2],
    /// Player to move, 0 or 1.
    pub to_move: usize,
    /// Passes and exchanges in a row, counting both players.
    pub scoreless_turns: u8,
    /// Turns taken so far.
    pub turn: u16,
    pub over: bool,
    /// Each player's most recent turn.
    pub last_turn: [LastTurn; 2],
    rng: Rng,
}

impl Game {
    /// A new game. The seed fixes the order of the bag and every later random choice.
    pub fn new(seed: u64) -> Game {
        let mut rng = Rng::new(seed);
        let mut tiles = [0u8; TOTAL_TILES];
        let mut next = 0;
        for (tile, &count) in TILE_COUNTS.iter().enumerate() {
            for _ in 0..count {
                tiles[next] = tile as u8;
                next += 1;
            }
        }
        rng.shuffle(&mut tiles);
        let mut game = Game {
            board: Board::new(),
            racks: [Rack::new(), Rack::new()],
            bag: Bag { tiles, len: TOTAL_TILES as u8 },
            scores: [0, 0],
            to_move: 0,
            scoreless_turns: 0,
            turn: 0,
            over: false,
            last_turn: [LastTurn::default(); 2],
            rng,
        };
        game.refill(0);
        game.refill(1);
        game
    }

    /// A game set up in an arbitrary position, for tests and analysis. `bag` is drawn from its
    /// last element backwards. Nothing is checked: see [`Game::tile_census`].
    pub fn from_position(board: Board, racks: [Rack; 2], bag: &[u8], scores: [i32; 2], to_move: usize, seed: u64) -> Game {
        Game {
            board,
            racks,
            bag: Bag::from_tiles(bag),
            scores,
            to_move,
            scoreless_turns: 0,
            turn: 0,
            over: false,
            last_turn: [LastTurn::default(); 2],
            rng: Rng::new(seed),
        }
    }

    fn refill(&mut self, player: usize) {
        while self.racks[player].len() < RACK_SIZE {
            match self.bag.draw() {
                Some(tile) => self.racks[player].add(tile),
                None => break,
            }
        }
    }

    #[inline]
    pub fn can_exchange(&self) -> bool {
        self.bag.len() >= EXCHANGE_MINIMUM
    }

    /// `player`'s score minus the opponent's.
    #[inline]
    pub fn spread(&self, player: usize) -> i32 {
        self.scores[player] - self.scores[1 - player]
    }

    /// Tiles `player` cannot see: the bag plus the opponent's rack. Computed the way a player
    /// would, from the full distribution minus the board minus their own rack.
    pub fn unseen(&self, player: usize) -> [u8; TILE_TYPES] {
        let mut counts = TILE_COUNTS;
        for &square in self.board.cells.iter() {
            if square != EMPTY {
                counts[square_tile(square) as usize] -= 1;
            }
        }
        for (count, held) in counts.iter_mut().zip(self.racks[player].counts) {
            *count -= held;
        }
        counts
    }

    /// Every tile in the game by type: board + both racks + bag. Always equals `TILE_COUNTS`
    /// in a legal game.
    pub fn tile_census(&self) -> [u8; TILE_TYPES] {
        let mut counts = self.bag.counts();
        for &square in self.board.cells.iter() {
            if square != EMPTY {
                counts[square_tile(square) as usize] += 1;
            }
        }
        for rack in &self.racks {
            for (count, held) in counts.iter_mut().zip(rack.counts) {
                *count += held;
            }
        }
        counts
    }

    /// Append every legal tile placement for the player to move.
    pub fn plays(&self, gaddag: &Gaddag, out: &mut Vec<Play>) {
        if !self.over {
            generate_plays(gaddag, &self.board, &self.racks[self.to_move], out);
        }
    }

    /// Every legal move for the player to move: placements, exchanges (if the bag allows) and
    /// pass. Empty once the game is over.
    pub fn legal_moves(&self, gaddag: &Gaddag) -> Vec<Move> {
        let mut moves = Vec::new();
        if self.over {
            return moves;
        }
        let mut plays = Vec::new();
        self.plays(gaddag, &mut plays);
        moves.extend(plays.into_iter().map(Move::Play));
        if self.can_exchange() {
            let mut exchanges = Vec::new();
            generate_exchanges(&self.racks[self.to_move], &mut exchanges);
            moves.extend(exchanges.into_iter().map(Move::Exchange));
        }
        moves.push(Move::Pass);
        moves
    }

    /// Play a move for the player to move. The move is trusted: it must come from the generator
    /// or have passed [`crate::validate::check_move`].
    pub fn apply(&mut self, mv: &Move) -> Applied {
        assert!(!self.over, "the game is over");
        let (me, opponent) = (self.to_move, 1 - self.to_move);
        let mut applied = Applied { score: mv.score(), ..Applied::default() };
        self.last_turn[me] = match mv {
            Move::Play(play) => LastTurn { kind: 1, tiles: play.tiles_played, score: play.score },
            Move::Exchange(tiles) => LastTurn { kind: 2, tiles: tiles.len() as u8, score: 0 },
            Move::Pass => LastTurn { kind: 3, tiles: 0, score: 0 },
        };
        match mv {
            Move::Play(play) => {
                for tile in play.placed() {
                    self.board.set(tile.index, tile.square());
                    self.racks[me].remove(tile.tile());
                }
                self.scores[me] += applied.score;
                self.scoreless_turns = 0;
                self.refill(me);
                if self.racks[me].is_empty() {
                    // Out: the rack could not be refilled, so the bag is empty too.
                    applied.adjustment[me] = 2 * self.racks[opponent].value();
                    applied.game_over = true;
                }
            }
            Move::Exchange(tiles) => {
                assert!(self.can_exchange(), "exchange with fewer than {EXCHANGE_MINIMUM} tiles in the bag");
                self.racks[me].remove_all(tiles);
                self.refill(me);
                for tile in tiles.tiles() {
                    self.bag.insert(tile, &mut self.rng);
                }
                self.scoreless_turns += 1;
            }
            Move::Pass => self.scoreless_turns += 1,
        }
        if self.scoreless_turns >= SCORELESS_LIMIT {
            applied.adjustment = [-self.racks[0].value(), -self.racks[1].value()];
            applied.game_over = true;
        }
        self.scores[0] += applied.adjustment[0];
        self.scores[1] += applied.adjustment[1];
        self.over = applied.game_over;
        self.turn += 1;
        self.to_move = opponent;
        applied
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tiles::BLANK;

    fn gaddag() -> Gaddag {
        Gaddag::build(&["AT", "TA", "CAT", "CATS", "ACT", "AS", "QI", "ZA", "AA", "EE", "TEE", "TEA", "EAT", "ATE"])
    }

    #[test]
    fn a_new_game_deals_seven_each_and_is_reproducible() {
        let game = Game::new(1);
        assert_eq!((game.racks[0].len(), game.racks[1].len(), game.bag.len()), (7, 7, 86));
        assert_eq!(game.tile_census(), TILE_COUNTS);
        let again = Game::new(1);
        assert_eq!(game.racks, again.racks);
        assert_eq!(game.bag.tiles(), again.bag.tiles());
        assert_ne!(Game::new(2).bag.tiles(), game.bag.tiles());
    }

    #[test]
    fn unseen_is_bag_plus_opponent_rack_with_board_blanks_as_blanks() {
        let gaddag = gaddag();
        let mut game = Game::new(3);
        // Hand player 0 a known rack by swapping with the bag through from_position.
        let racks = [Rack::parse("CAT?EEE").unwrap(), game.racks[1]];
        let mut pool = TILE_COUNTS;
        for rack in &racks {
            for (count, held) in pool.iter_mut().zip(rack.counts) {
                *count -= held;
            }
        }
        let bag: Vec<u8> = (0..TILE_TYPES as u8).flat_map(|t| std::iter::repeat(t).take(pool[t as usize] as usize)).collect();
        game = Game::from_position(Board::new(), racks, &bag, [0, 0], 0, 9);
        assert_eq!(game.tile_census(), TILE_COUNTS);

        let mut plays = Vec::new();
        game.plays(&gaddag, &mut plays);
        let with_blank = plays.iter().find(|p| p.to_string().starts_with("8H CAt ")).copied().expect("CAt");
        game.apply(&Move::Play(with_blank));
        assert_eq!(game.tile_census(), TILE_COUNTS);

        for player in 0..2 {
            let mut expected = game.bag.counts();
            for (count, held) in expected.iter_mut().zip(game.racks[1 - player].counts) {
                *count += held;
            }
            assert_eq!(game.unseen(player), expected, "player {player}");
        }
        // The blank on the board is no longer unseen; the T it stands for still is.
        assert_eq!(game.unseen(1)[BLANK as usize] + game.racks[1].count(BLANK), 1);
    }

    #[test]
    fn going_out_ends_the_game_and_doubles_the_opponents_rack() {
        let gaddag = gaddag();
        let racks = [Rack::parse("AT").unwrap(), Rack::parse("QZ").unwrap()];
        let mut game = Game::from_position(Board::new(), racks, &[], [10, 30], 0, 0);
        let mut plays = Vec::new();
        game.plays(&gaddag, &mut plays);
        let applied = game.apply(&Move::Play(plays[0]));
        assert!(applied.game_over && game.over);
        assert_eq!(applied.score, 4);
        assert_eq!(applied.adjustment, [2 * 20, 0]);
        assert_eq!(game.scores, [10 + 4 + 40, 30]);
        assert!(game.legal_moves(&gaddag).is_empty());
    }

    #[test]
    fn playing_the_last_tiles_with_tiles_in_the_bag_does_not_end_the_game() {
        let gaddag = gaddag();
        let racks = [Rack::parse("AT").unwrap(), Rack::parse("QZ").unwrap()];
        let mut game = Game::from_position(Board::new(), racks, &[4, 4, 4], [0, 0], 0, 0);
        let mut plays = Vec::new();
        game.plays(&gaddag, &mut plays);
        let applied = game.apply(&Move::Play(plays[0]));
        assert!(!applied.game_over);
        assert_eq!(game.racks[0].to_string(), "EEE");
        assert!(game.bag.is_empty());
    }

    #[test]
    fn six_scoreless_turns_end_the_game_and_each_side_loses_its_rack() {
        let gaddag = gaddag();
        let mut game = Game::new(5);
        let values = [game.racks[0].value(), game.racks[1].value()];
        for turn in 0..6 {
            assert!(!game.over, "ended after {turn} passes");
            game.apply(&Move::Pass);
        }
        assert!(game.over);
        assert_eq!(game.scores, [-values[0], -values[1]]);

        // A placement in between resets the count.
        let mut game = Game::new(5);
        for _ in 0..5 {
            game.apply(&Move::Pass);
        }
        let mut plays = Vec::new();
        game.plays(&gaddag, &mut plays);
        if let Some(play) = plays.first() {
            game.apply(&Move::Play(*play));
            assert_eq!(game.scoreless_turns, 0);
            for _ in 0..5 {
                game.apply(&Move::Pass);
            }
            assert!(!game.over);
        }
    }

    #[test]
    fn exchange_draws_first_keeps_the_count_and_needs_seven_in_the_bag() {
        let mut game = Game::new(11);
        let before = game.racks[0];
        let thrown = Rack::from_counts({
            let mut counts = [0; TILE_TYPES];
            let first = before.tiles().next().unwrap();
            counts[first as usize] = 1;
            counts
        });
        let next_draw = *game.bag.tiles().last().unwrap();
        let applied = game.apply(&Move::Exchange(thrown));
        assert_eq!(applied.score, 0);
        assert_eq!(game.scoreless_turns, 1);
        assert_eq!((game.racks[0].len(), game.bag.len()), (7, 86));
        assert_eq!(game.tile_census(), TILE_COUNTS);
        // The replacement is the tile that was next in the bag, not the one thrown back.
        let mut expected = before;
        expected.remove_all(&thrown);
        expected.add(next_draw);
        assert_eq!(game.racks[0], expected);

        let short = Game::from_position(Board::new(), [before, game.racks[1]], &[0; 6], [0, 0], 0, 0);
        assert!(!short.can_exchange());
        assert!(short.legal_moves(&gaddag()).iter().all(|m| !matches!(m, Move::Exchange(_))));
        let enough = Game::from_position(Board::new(), [before, game.racks[1]], &[0; 7], [0, 0], 0, 0);
        assert!(enough.legal_moves(&gaddag()).iter().any(|m| matches!(m, Move::Exchange(_))));
    }
}
