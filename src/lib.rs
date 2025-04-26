use once_cell::sync::Lazy;
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::error::Error;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;
use itertools::Itertools;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyString};
use serde::{Serialize, Deserialize};


const BOARD_SIZE: usize = 15;
const RACK_SIZE: usize = 7;
const ALPHABET: &str = "abcdefghijklmnopqrstuvwxyz";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum SpecialSpace {
    TripleWord,
    DoubleWord,
    TripleLetter,
    DoubleLetter,
}

static SPECIAL_SPACES: phf::Map<&'static str, SpecialSpace> = phf::phf_map! {
    "a1" => SpecialSpace::TripleWord, "a4" => SpecialSpace::DoubleLetter, "a8" => SpecialSpace::TripleWord, "a12" => SpecialSpace::DoubleLetter, "a15" => SpecialSpace::TripleWord,
    "b2" => SpecialSpace::DoubleWord, "b6" => SpecialSpace::TripleLetter, "b10" => SpecialSpace::TripleLetter, "b14" => SpecialSpace::DoubleWord,
    "c3" => SpecialSpace::DoubleWord, "c7" => SpecialSpace::DoubleLetter, "c9" => SpecialSpace::DoubleLetter, "c13" => SpecialSpace::DoubleWord,
    "d1" => SpecialSpace::DoubleLetter, "d4" => SpecialSpace::DoubleWord, "d8" => SpecialSpace::DoubleLetter, "d12" => SpecialSpace::DoubleWord, "d15" => SpecialSpace::DoubleLetter,
    "e5" => SpecialSpace::DoubleWord, "e11" => SpecialSpace::DoubleWord,
    "f2" => SpecialSpace::TripleLetter, "f6" => SpecialSpace::TripleLetter, "f10" => SpecialSpace::TripleLetter, "f14" => SpecialSpace::TripleLetter,
    "g3" => SpecialSpace::DoubleLetter, "g7" => SpecialSpace::DoubleLetter, "g9" => SpecialSpace::DoubleLetter, "g13" => SpecialSpace::DoubleLetter,
    "h1" => SpecialSpace::TripleWord, "h4" => SpecialSpace::DoubleLetter, "h8" => SpecialSpace::DoubleWord, "h12" => SpecialSpace::DoubleLetter, "h15" => SpecialSpace::TripleWord,
    "i3" => SpecialSpace::DoubleLetter, "i7" => SpecialSpace::DoubleLetter, "i9" => SpecialSpace::DoubleLetter, "i13" => SpecialSpace::DoubleLetter,
    "j2" => SpecialSpace::TripleLetter, "j6" => SpecialSpace::TripleLetter, "j10" => SpecialSpace::TripleLetter, "j14" => SpecialSpace::TripleLetter,
    "k5" => SpecialSpace::DoubleWord, "k11" => SpecialSpace::DoubleWord,
    "l1" => SpecialSpace::DoubleLetter, "l4" => SpecialSpace::DoubleWord, "l8" => SpecialSpace::DoubleLetter, "l12" => SpecialSpace::DoubleWord, "l15" => SpecialSpace::DoubleLetter,
    "m3" => SpecialSpace::DoubleWord, "m7" => SpecialSpace::DoubleLetter, "m9" => SpecialSpace::DoubleLetter, "m13" => SpecialSpace::DoubleWord,
    "n2" => SpecialSpace::DoubleWord, "n6" => SpecialSpace::TripleLetter, "n10" => SpecialSpace::TripleLetter, "n14" => SpecialSpace::DoubleWord,
    "o1" => SpecialSpace::TripleWord, "o4" => SpecialSpace::DoubleLetter, "o8" => SpecialSpace::TripleWord, "o12" => SpecialSpace::DoubleLetter, "o15" => SpecialSpace::TripleWord,
};

static LETTER_POINT_VALUES: phf::Map<char, u32> = phf::phf_map! {
    'a' => 1, 'b' => 3, 'c' => 3, 'd' => 2, 'e' => 1, 'f' => 4, 'g' => 2, 'h' => 4, 'i' => 1,
    'j' => 8, 'k' => 5, 'l' => 1, 'm' => 3, 'n' => 1, 'o' => 1, 'p' => 3, 'q' => 10, 'r' => 1,
    's' => 1, 't' => 1, 'u' => 1, 'v' => 4, 'w' => 4, 'x' => 8, 'y' => 4, 'z' => 10,
    '_' => 0,
};

struct Dictionary {
    words: HashSet<String>,
    prefixes: HashSet<String>,
}

static DICTIONARY: Lazy<Result<Arc<Dictionary>, Box<dyn Error + Send + Sync>>> =
    Lazy::new(|| {
        load_dictionary().map(Arc::new)
    });

fn load_dictionary() -> Result<Dictionary, Box<dyn Error + Send + Sync>> {
    let dict_path = Path::new("word_dictionary.csv");
    if !dict_path.exists() {
         let src_dict_path = Path::new("src/word_dictionary.csv");
         if src_dict_path.exists() {
             return load_from_path(src_dict_path);
         } else {
            let msg = format!(
                "Dictionary file not found at {:?} or {:?}",
                dict_path.canonicalize().unwrap_or_else(|_| dict_path.to_path_buf()),
                src_dict_path.canonicalize().unwrap_or_else(|_| src_dict_path.to_path_buf())
            );
             eprintln!("ERROR: {}", msg);
             return Err(msg.into());
         }
    }
    load_from_path(dict_path)
}

fn load_from_path(path: &Path) -> Result<Dictionary, Box<dyn Error + Send + Sync>> {
    println!("Loading dictionary from: {:?}", path);
    let file = File::open(path)?;
    let mut rdr = csv::Reader::from_reader(file);
    let mut words = HashSet::new();
    let mut prefixes = HashSet::new();

    for result in rdr.records() {
        let record = result?;
        if let Some(word) = record.get(0) {
            let word_lower = word.trim().to_lowercase();
            if !word_lower.is_empty() {
                words.insert(word_lower.clone());
                for i in 1..=word_lower.len() {
                    prefixes.insert(word_lower[..i].to_string());
                }
            }
        }
    }
    println!("Dictionary loaded: {} words, {} prefixes.", words.len(), prefixes.len());
    Ok(Dictionary { words, prefixes })
}


type Position = String;
type BoardState = HashMap<Position, Option<char>>;

fn create_empty_board() -> BoardState {
    let mut board = HashMap::new();
    for r in 'a'..='o' {
        for c in 1..=BOARD_SIZE {
            board.insert(format!("{}{}", r, c), None);
        }
    }
    board
}

fn pos_to_indices(pos: &str) -> Option<(usize, usize)> {
    if pos.len() < 2 { return None; }
    let row_char = pos.chars().next()?;
    let col_str = &pos[1..];
    let row = (row_char as u32).checked_sub('a' as u32)? as usize;
    let col = col_str.parse::<usize>().ok()?.checked_sub(1)?;
    if row < BOARD_SIZE && col < BOARD_SIZE {
        Some((row, col))
    } else {
        None
    }
}

fn indices_to_pos(row: usize, col: usize) -> Option<Position> {
    if row < BOARD_SIZE && col < BOARD_SIZE {
        let row_char = (('a' as u8) + row as u8) as char;
        Some(format!("{}{}", row_char, col + 1))
    } else {
        None
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize)]
enum Direction { Right, Down, Left, Up }


#[derive(Debug, Clone, PartialEq, Serialize)]
struct MoveDetail {
    word: String,
    start_space: Position,
    direction: Direction,
    score: u32,
    positions: Vec<Position>,
    blank_assignments: HashMap<Position, char>,
    cross_words: Vec<CrossWordInfo>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize)]
struct CrossWordInfo {
    word: String,
    start: Position,
    direction: Direction,
    positions: Vec<Position>,
    intersection_pos: Position,
}


#[derive(Debug, Clone)]
struct WordSegment {
    letter: char,
    pos: Position,
    is_from_rack: bool,
}

struct ScrabbleSolver {
    board_state: BoardState,
    player_hand: Vec<char>,
    dictionary: Arc<Dictionary>,
}

impl ScrabbleSolver {
    fn new(board_state: BoardState, player_hand: Vec<char>, dictionary: Arc<Dictionary>) -> Self {
        ScrabbleSolver {
            board_state,
            player_hand: player_hand.iter().map(|c| c.to_ascii_lowercase()).collect(),
            dictionary,
        }
    }

    fn get_anchor_squares(&self) -> Vec<Position> {
        if self.board_state.values().all(|opt| opt.is_none()) {
            return vec!["h8".to_string()];
        }

        let mut anchors = HashSet::new();
        for (pos, tile_opt) in &self.board_state {
            if tile_opt.is_some() {
                if let Some((r, c)) = pos_to_indices(pos) {
                    for (dr, dc) in [(0, 1), (0, -1), (1, 0), (-1, 0)] {
                        let nr = r as isize + dr;
                        let nc = c as isize + dc;
                        if nr >= 0 && nr < BOARD_SIZE as isize && nc >= 0 && nc < BOARD_SIZE as isize {
                           if let Some(npos) = indices_to_pos(nr as usize, nc as usize) {
                               if self.board_state.get(&npos).map_or(false, |opt| opt.is_none()) {
                                   anchors.insert(npos);
                               }
                           }
                        }
                    }
                }
            }
        }
        anchors.into_iter().collect()
    }

    fn process_anchor(&self, anchor: &str) -> Vec<MoveDetail> {
        let mut valid_moves_local = Vec::new();
        let mut seen_moves_local: HashSet<(String, Position, Direction)> = HashSet::new();
        let rack_letters = &self.player_hand;
        let rack_len = rack_letters.len();

        for &direction in &[Direction::Right, Direction::Down, Direction::Left, Direction::Up] {
            let max_space = self.get_max_space(anchor, direction);

            for num_rack_tiles_used in 1..=std::cmp::min(rack_len, max_space) {
                for combo_tuple in rack_letters.iter().copied().permutations(num_rack_tiles_used) {
                    let combo_list_original = combo_tuple;
                    let blanks_in_combo = combo_list_original.iter().filter(|&&c| c == '_').count();

                    let blank_sub_combinations: Box<dyn Iterator<Item = Vec<char>>> = if blanks_in_combo > 0 {
                        Box::new(
                            (0..blanks_in_combo)
                                .map(|_| ALPHABET.chars())
                                .multi_cartesian_product()
                        )
                    } else {
                        Box::new(std::iter::once(vec![]))
                    };

                    for substitutions in blank_sub_combinations {
                        let mut combo_list_modified = Vec::with_capacity(combo_list_original.len());
                        let mut current_blank_info = HashMap::new();
                        let mut sub_iter = substitutions.iter();

                        for (idx, &tile) in combo_list_original.iter().enumerate() {
                            if tile == '_' {
                                if let Some(&assigned_letter) = sub_iter.next() {
                                    combo_list_modified.push(assigned_letter);
                                    current_blank_info.insert(idx, assigned_letter);
                                } else {
                                     eprintln!("Error: Mismatch in blank substitutions");
                                     continue;
                                }
                            } else {
                                combo_list_modified.push(tile);
                            }
                        }

                        for anchor_idx in 0..combo_list_modified.len() {
                             self.try_place_with_anchor_at_position(
                                anchor,
                                direction,
                                &combo_list_modified,
                                &combo_list_original,
                                &current_blank_info,
                                anchor_idx,
                                &mut valid_moves_local,
                                &mut seen_moves_local,
                            );
                        }
                    }
                }
            }
        }
        valid_moves_local
    }


    fn get_max_space(&self, anchor: &str, direction: Direction) -> usize {
        pos_to_indices(anchor).map_or(0, |(r, c)| match direction {
            Direction::Right => BOARD_SIZE - 1 - c,
            Direction::Down => BOARD_SIZE - 1 - r,
            Direction::Left => c,
            Direction::Up => r,
        })
    }

    fn try_place_with_anchor_at_position(
        &self,
        anchor: &str,
        direction: Direction,
        combo_list_modified: &[char],
        combo_list_original: &[char],
        _blank_info_for_combo: &HashMap<usize, char>,
        anchor_idx_in_combo: usize,
        valid_moves: &mut Vec<MoveDetail>,
        seen_moves: &mut HashSet<(String, Position, Direction)>,
    ) {
        if let Some((anchor_r, anchor_c)) = pos_to_indices(anchor) {
            let (mut start_r, mut start_c) = (anchor_r as isize, anchor_c as isize);

            match direction {
                Direction::Right => start_c -= anchor_idx_in_combo as isize,
                Direction::Down => start_r -= anchor_idx_in_combo as isize,
                Direction::Left => start_c += anchor_idx_in_combo as isize - (combo_list_modified.len() as isize - 1),
                Direction::Up => start_r += anchor_idx_in_combo as isize - (combo_list_modified.len() as isize - 1),
            }

             if start_r < 0 || start_r >= BOARD_SIZE as isize || start_c < 0 || start_c >= BOARD_SIZE as isize {
                 return;
             }

             if let Some(start_pos) = indices_to_pos(start_r as usize, start_c as usize) {
                 self.validate_and_place_word(
                     &start_pos,
                     direction,
                     combo_list_modified,
                     combo_list_original,
                     valid_moves,
                     seen_moves,
                 );
             }
        }
    }


   fn validate_and_place_word(
       &self,
       start_pos: &str,
       placement_direction: Direction,
       combo_modified: &[char],
       combo_original: &[char],
       valid_moves: &mut Vec<MoveDetail>,
       seen_moves: &mut HashSet<(String, Position, Direction)>,
   ) {
        let (start_r, start_c) = pos_to_indices(start_pos).unwrap_or((BOARD_SIZE, BOARD_SIZE));
        if start_r >= BOARD_SIZE || start_c >= BOARD_SIZE { return; }

        let mut formed_word_segments: Vec<WordSegment> = Vec::new();
        let mut temp_rack_iter = combo_modified.iter();
        let mut current_blank_assignments = HashMap::new();
        let mut tiles_placed_count = 0;
        let mut combo_original_idx = 0;

        let mut current_r = start_r as isize;
        let mut current_c = start_c as isize;
        let (dr, dc) = match placement_direction {
            Direction::Right => (0, 1),
            Direction::Down => (1, 0),
            Direction::Left => (0, -1),
            Direction::Up => (-1, 0),
        };

        loop {
            if current_r < 0 || current_r >= BOARD_SIZE as isize || current_c < 0 || current_c >= BOARD_SIZE as isize {
                 break;
            }
            let current_pos = indices_to_pos(current_r as usize, current_c as usize).unwrap();

            if let Some(board_tile_opt) = self.board_state.get(&current_pos) {
                match board_tile_opt {
                    Some(board_char) => {
                        formed_word_segments.push(WordSegment {
                            letter: *board_char,
                            pos: current_pos.clone(),
                            is_from_rack: false,
                        });
                    }
                    None => {
                         if let Some(&rack_char) = temp_rack_iter.next() {
                            formed_word_segments.push(WordSegment {
                                letter: rack_char,
                                pos: current_pos.clone(),
                                is_from_rack: true,
                            });
                            tiles_placed_count += 1;

                            if combo_original_idx < combo_original.len() && combo_original[combo_original_idx] == '_' {
                                current_blank_assignments.insert(current_pos.clone(), rack_char);
                            }
                            combo_original_idx += 1;

                         } else {
                             break;
                         }
                    }
                }
            } else {
                 break;
            }

            let (current_word_str_for_prefix, _) = self.get_word_string_and_positions(&formed_word_segments, placement_direction);
            if !self.could_form_valid_word(&current_word_str_for_prefix) {
                return;
            }

            current_r += dr;
            current_c += dc;

             if temp_rack_iter.len() == 0 {
                let next_r = current_r;
                let next_c = current_c;
                 if next_r < 0 || next_r >= BOARD_SIZE as isize || next_c < 0 || next_c >= BOARD_SIZE as isize {
                     break;
                 }
                 if let Some(next_pos) = indices_to_pos(next_r as usize, next_c as usize) {
                     if self.board_state.get(&next_pos).map_or(true, |opt| opt.is_none()) {
                         break;
                     }
                 } else {
                     break;
                 }
             }
        }

        if tiles_placed_count == 0 {
             return;
        }

        self.extend_segments_backwards(&mut formed_word_segments, placement_direction);
        self.extend_segments_forwards(&mut formed_word_segments, placement_direction);


        let (final_word, final_positions) = self.get_word_string_and_positions(&formed_word_segments, placement_direction);
        let final_word_lower = final_word.to_lowercase();

        if final_word.len() < 2 || !self.dictionary.words.contains(&final_word_lower) {
            return;
        }

        let (display_start_space, display_direction) = if final_positions.is_empty() {
            return;
        } else {
             let first_pos = &final_positions[0];
             let (r1, c1) = pos_to_indices(first_pos).unwrap();
             if final_positions.len() > 1 {
                 let second_pos = &final_positions[1];
                 let (r2, c2) = pos_to_indices(second_pos).unwrap();
                 if r1 == r2 {
                     (first_pos.clone(), Direction::Right)
                 } else if c1 == c2 {
                      (first_pos.clone(), Direction::Down)
                 } else {
                      return;
                 }
             } else {
                 (first_pos.clone(), Direction::Right)
             }
        };


        let is_first_move = self.board_state.values().all(|opt| opt.is_none());

        if is_first_move && !final_positions.contains(&"h8".to_string()) {
            return;
        }

        let uses_existing_tile = formed_word_segments.iter().any(|seg| !seg.is_from_rack);
        if !is_first_move && !uses_existing_tile {
             let touches_existing = formed_word_segments.iter().any(|seg| {
                 if seg.is_from_rack {
                     self.is_adjacent_to_existing_tile(&seg.pos)
                 } else {
                     false
                 }
             });
             if !touches_existing {
                 return;
             }
        }


        let cross_words_result = self.check_cross_words(&formed_word_segments, &final_positions, &current_blank_assignments);
        let (cross_words_valid, cross_words_details) = match cross_words_result {
            Ok(details) => (true, details),
            Err(_) => (false, vec![]),
        };

        if !cross_words_valid {
            return;
        }

        if !is_first_move && !uses_existing_tile && cross_words_details.is_empty() {
             return;
        }


        let score = self.calculate_score(&formed_word_segments, &final_positions, &current_blank_assignments, &cross_words_details);


        let move_key = (final_word.clone(), display_start_space.clone(), display_direction);
        if !seen_moves.contains(&move_key) {
            seen_moves.insert(move_key);
            valid_moves.push(MoveDetail {
                word: final_word,
                start_space: display_start_space,
                direction: display_direction,
                score,
                positions: final_positions,
                blank_assignments: current_blank_assignments,
                cross_words: cross_words_details,
            });
        }
    }

    fn get_word_string_and_positions(
        &self,
        segments: &[WordSegment],
        placement_direction: Direction
    ) -> (String, Vec<Position>) {
         if segments.is_empty() {
             return ("".to_string(), vec![]);
         }

         let (first_r, _first_c) = pos_to_indices(&segments[0].pos).unwrap_or((BOARD_SIZE, BOARD_SIZE));
         let _is_naturally_horizontal = if segments.len() > 1 {
             let (second_r, _) = pos_to_indices(&segments[1].pos).unwrap_or((BOARD_SIZE, BOARD_SIZE));
             first_r == second_r
         } else {
             placement_direction == Direction::Right || placement_direction == Direction::Left
         };

         let needs_reversal = match placement_direction {
             Direction::Left => true,
             Direction::Up => true,
             Direction::Right | Direction::Down => false,
         };

        let final_segments: Vec<_> = if needs_reversal {
             segments.iter().rev().cloned().collect()
         } else {
             segments.to_vec()
         };

         let final_word = final_segments.iter().map(|seg| seg.letter).collect::<String>();
         let final_positions = final_segments.iter().map(|seg| seg.pos.clone()).collect::<Vec<Position>>();

         (final_word, final_positions)
     }


    fn is_adjacent_to_existing_tile(&self, pos: &str) -> bool {
        if let Some((r, c)) = pos_to_indices(pos) {
            for (dr, dc) in [(0, 1), (0, -1), (1, 0), (-1, 0)] {
                let nr = r as isize + dr;
                let nc = c as isize + dc;
                if nr >= 0 && nr < BOARD_SIZE as isize && nc >= 0 && nc < BOARD_SIZE as isize {
                   if let Some(npos) = indices_to_pos(nr as usize, nc as usize) {
                       if self.board_state.get(&npos).map_or(false, |opt| opt.is_some()) {
                           return true;
                       }
                   }
                }
            }
        }
        false
    }


    fn extend_segments_backwards(&self, segments: &mut Vec<WordSegment>, direction: Direction) {
        if segments.is_empty() { return; }
        let (mut current_r, mut current_c) = pos_to_indices(&segments[0].pos)
            .map(|(r,c)| (r as isize, c as isize))
            .unwrap_or((-1, -1));

        let (dr, dc) = match direction {
            Direction::Right => (0, -1),
            Direction::Down => (-1, 0),
            Direction::Left => (0, 1),
            Direction::Up => (1, 0),
        };

        loop {
            current_r += dr;
            current_c += dc;
             if current_r < 0 || current_r >= BOARD_SIZE as isize || current_c < 0 || current_c >= BOARD_SIZE as isize { break; }

             if let Some(prev_pos) = indices_to_pos(current_r as usize, current_c as usize) {
                 if let Some(Some(tile)) = self.board_state.get(&prev_pos) {
                     segments.insert(0, WordSegment { letter: *tile, pos: prev_pos, is_from_rack: false });
                 } else {
                     break;
                 }
             } else {
                 break;
             }
        }
    }

    fn extend_segments_forwards(&self, segments: &mut Vec<WordSegment>, direction: Direction) {
        if segments.is_empty() { return; }
        let (mut current_r, mut current_c) = pos_to_indices(segments.last().unwrap().pos.as_str())
             .map(|(r,c)| (r as isize, c as isize))
             .unwrap_or((-1, -1));

        let (dr, dc) = match direction {
            Direction::Right => (0, 1),
            Direction::Down => (1, 0),
            Direction::Left => (0, -1),
            Direction::Up => (-1, 0),
        };

        loop {
             current_r += dr;
             current_c += dc;
             if current_r < 0 || current_r >= BOARD_SIZE as isize || current_c < 0 || current_c >= BOARD_SIZE as isize { break; }

             if let Some(next_pos) = indices_to_pos(current_r as usize, current_c as usize) {
                 if let Some(Some(tile)) = self.board_state.get(&next_pos) {
                     segments.push(WordSegment { letter: *tile, pos: next_pos, is_from_rack: false });
                 } else {
                     break;
                 }
             } else {
                 break;
             }
         }
    }

     fn could_form_valid_word(&self, prefix: &str) -> bool {
        self.dictionary.prefixes.contains(&prefix.to_lowercase())
    }

    fn check_cross_words(
        &self,
        main_word_segments: &[WordSegment],
        main_word_positions: &[Position],
        _placed_blank_assignments: &HashMap<Position, char>,
    ) -> Result<Vec<CrossWordInfo>, ()> {
        let mut cross_words_details = Vec::new();

        let mut temp_board = self.board_state.clone();
        for segment in main_word_segments {
            if segment.is_from_rack {
                temp_board.insert(segment.pos.clone(), Some(segment.letter));
            }
        }

        for segment in main_word_segments {
             if !segment.is_from_rack { continue; }

             let pos = &segment.pos;
             let (r, c) = pos_to_indices(pos).ok_or(())?;

             let main_is_horizontal = if main_word_positions.len() > 1 {
                 let (r1, _) = pos_to_indices(&main_word_positions[0]).ok_or(())?;
                 let (r2, _) = pos_to_indices(&main_word_positions[1]).ok_or(())?;
                 r1 == r2
             } else {
                  let has_horizontal_neighbor =
                       (c > 0 && temp_board.get(&indices_to_pos(r, c-1).unwrap()).map_or(false, Option::is_some)) ||
                       (c < BOARD_SIZE - 1 && temp_board.get(&indices_to_pos(r, c+1).unwrap()).map_or(false, Option::is_some));
                  let has_vertical_neighbor =
                        (r > 0 && temp_board.get(&indices_to_pos(r-1, c).unwrap()).map_or(false, Option::is_some)) ||
                        (r < BOARD_SIZE - 1 && temp_board.get(&indices_to_pos(r+1, c).unwrap()).map_or(false, Option::is_some));


                   if has_horizontal_neighbor && !has_vertical_neighbor {
                        false
                   } else {
                        true
                   }
             };


             if main_is_horizontal {
                 if let Some(cross_info) = self.form_and_validate_cross_word(r, c, Direction::Down, &temp_board)? {
                    cross_words_details.push(cross_info);
                 }
             } else {
                  if let Some(cross_info) = self.form_and_validate_cross_word(r, c, Direction::Right, &temp_board)? {
                    cross_words_details.push(cross_info);
                  }
             }
        }

        cross_words_details.sort_by_key(|info| (info.start.clone(), info.direction));
        cross_words_details.dedup_by_key(|info| (info.start.clone(), info.direction));


        Ok(cross_words_details)
    }


    fn form_and_validate_cross_word(
        &self,
        intersect_r: usize,
        intersect_c: usize,
        cross_direction: Direction,
        temp_board: &BoardState,
    ) -> Result<Option<CrossWordInfo>, ()> {

        let (mut current_r, mut current_c) = (intersect_r as isize, intersect_c as isize);
        let (dr, dc) = match cross_direction {
            Direction::Right => (0, 1),
            Direction::Down => (1, 0),
            _ => return Ok(None),
        };
        let (bdr, bdc) = (-dr, -dc);


        while let Some(prev_pos) = indices_to_pos((current_r + bdr) as usize, (current_c + bdc) as usize) {
             if temp_board.get(&prev_pos).map_or(false, |opt| opt.is_some()) {
                 current_r += bdr;
                 current_c += bdc;
             } else {
                 break;
             }
        }

        let start_pos = indices_to_pos(current_r as usize, current_c as usize).ok_or(())?;
        let mut cross_word_chars = Vec::new();
        let mut cross_positions = Vec::new();

        loop {
             if current_r < 0 || current_r >= BOARD_SIZE as isize || current_c < 0 || current_c >= BOARD_SIZE as isize { break; }
             let pos = indices_to_pos(current_r as usize, current_c as usize).ok_or(())?;

             if let Some(Some(tile)) = temp_board.get(&pos) {
                 cross_word_chars.push(*tile);
                 cross_positions.push(pos);
                 current_r += dr;
                 current_c += dc;
             } else {
                 break;
             }
        }


        if cross_word_chars.len() > 1 {
            let cross_word: String = cross_word_chars.iter().collect();
            if self.dictionary.words.contains(&cross_word.to_lowercase()) {
                Ok(Some(CrossWordInfo {
                     word: cross_word,
                     start: start_pos,
                     direction: cross_direction,
                     positions: cross_positions,
                     intersection_pos: indices_to_pos(intersect_r, intersect_c).unwrap(),
                 }))
            } else {
                 Err(())
            }
        } else {
            Ok(None)
        }
    }


    fn calculate_score(
        &self,
        main_word_segments: &[WordSegment],
        _main_word_positions_in_reading_order: &[Position],
        placed_blank_assignments: &HashMap<Position, char>,
        cross_words_details: &[CrossWordInfo],
    ) -> u32 {
        let mut total_score = 0;
        let mut main_word_score = 0;
        let mut main_word_multiplier = 1;

        for segment in main_word_segments {
            let letter = segment.letter;
            let pos = &segment.pos;
            let is_blank = placed_blank_assignments.contains_key(pos);
            let letter_base_score = if is_blank { 0 } else { *LETTER_POINT_VALUES.get(&letter.to_ascii_lowercase()).unwrap_or(&0) };
            let mut letter_multiplier = 1;

             if segment.is_from_rack {
                if let Some(special) = SPECIAL_SPACES.get(pos) {
                    match special {
                        SpecialSpace::DoubleLetter => letter_multiplier = 2,
                        SpecialSpace::TripleLetter => letter_multiplier = 3,
                        SpecialSpace::DoubleWord => main_word_multiplier *= 2,
                        SpecialSpace::TripleWord => main_word_multiplier *= 3,
                    }
                }
            }
             main_word_score += letter_base_score * letter_multiplier;
        }
        main_word_score *= main_word_multiplier;
        total_score += main_word_score;


        for cross_info in cross_words_details {
             let mut cross_word_score = 0;
             let mut cross_word_multiplier = 1;
             let intersection_pos = &cross_info.intersection_pos;

             for (idx, cross_pos) in cross_info.positions.iter().enumerate() {
                 let cross_letter = cross_info.word.chars().nth(idx).unwrap();
                 let is_intersecting_blank = placed_blank_assignments.contains_key(cross_pos);
                 let cross_base_score = if is_intersecting_blank { 0 } else { *LETTER_POINT_VALUES.get(&cross_letter.to_ascii_lowercase()).unwrap_or(&0) };
                 let mut cross_letter_multiplier = 1;

                 if cross_pos == intersection_pos {
                     if let Some(special) = SPECIAL_SPACES.get(cross_pos.as_str()) {
                         match special {
                             SpecialSpace::DoubleLetter => cross_letter_multiplier = 2,
                             SpecialSpace::TripleLetter => cross_letter_multiplier = 3,
                             SpecialSpace::DoubleWord => cross_word_multiplier *= 2,
                             SpecialSpace::TripleWord => cross_word_multiplier *= 3,
                         }
                     }
                 }
                 cross_word_score += cross_base_score * cross_letter_multiplier;
             }
             cross_word_score *= cross_word_multiplier;
             total_score += cross_word_score;
        }

        let tiles_placed_count = main_word_segments.iter().filter(|seg| seg.is_from_rack).count();
        if tiles_placed_count == RACK_SIZE {
            total_score += 50;
        }

        total_score
    }

    pub fn get_all_valid_moves(&self) -> Vec<MoveDetail> {
        let anchor_squares = self.get_anchor_squares();
        if anchor_squares.is_empty() {
            return vec![];
        }

        let results: Vec<Vec<MoveDetail>> = anchor_squares
            .par_iter()
            .map(|anchor| {
                self.process_anchor(anchor)
            })
            .collect();

        let mut valid_moves = Vec::new();
        let mut seen_moves_global: HashSet<(String, Position, Direction)> = HashSet::new();

        for sublist in results {
            for mv in sublist {
                let move_key = (mv.word.clone(), mv.start_space.clone(), mv.direction);
                if !seen_moves_global.contains(&move_key) {
                    seen_moves_global.insert(move_key);
                    valid_moves.push(mv);
                }
            }
        }

        valid_moves.sort_unstable_by(|a, b| b.score.cmp(&a.score));

        valid_moves
    }
}


#[pyfunction]
fn get_valid_moves_rs(py_board_state: &PyDict, py_player_hand: &PyList) -> PyResult<PyObject> {
    let mut board_state: BoardState = BoardState::new();
    for (key, value) in py_board_state.iter() {
        let pos: String = key.extract()?;
        if !value.is_none() {
             let tile: char = value.extract()?;
             board_state.insert(pos, Some(tile));
        } else {
             board_state.insert(pos, None);
        }
    }

    for r in 'a'..='o' {
         for c in 1..=BOARD_SIZE {
             let pos_str = format!("{}{}", r, c);
             board_state.entry(pos_str).or_insert(None);
         }
     }

    let player_hand: Vec<char> = py_player_hand.extract()?;

    let dictionary = match &*DICTIONARY {
        Ok(dict_arc) => Arc::clone(dict_arc),
        Err(e) => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Dictionary failed to load: {}",
                e
            )));
        }
    };

    let solver = ScrabbleSolver::new(board_state, player_hand, dictionary);
    let valid_moves: Vec<MoveDetail> = solver.get_all_valid_moves();

    Python::with_gil(|py| {
        let py_moves = PyList::empty(py);
        for mv in valid_moves {
            let move_dict = PyDict::new(py);
            move_dict.set_item("word", mv.word)?;
            move_dict.set_item("start_space", mv.start_space)?;
            let direction_str = match mv.direction {
                Direction::Right => "right",
                Direction::Down => "down",
                _ => "unknown",
            };
            move_dict.set_item("direction", direction_str)?;
            move_dict.set_item("score", mv.score)?;
            move_dict.set_item("positions", mv.positions)?;
            let py_blanks = PyDict::new(py);
            for (pos, letter) in mv.blank_assignments {
                 py_blanks.set_item(pos, letter)?;
             }
            move_dict.set_item("blank_assignments", py_blanks)?;

            py_moves.append(move_dict)?;
        }
        Ok(py_moves.to_object(py))
    })
}


#[pymodule]
fn scrabble_valid_moves_rust(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(get_valid_moves_rs, m)?)?;
    Ok(())
}