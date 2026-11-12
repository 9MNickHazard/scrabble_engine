use scrabble_valid_moves_rust::{
    ScrabbleSolver,
    create_empty_board,
    DICTIONARY
};
use std::{sync::Arc, time::Instant};

fn main() {
    let dictionary = match &*DICTIONARY {
       Ok(dict_arc) => Arc::clone(dict_arc),
       Err(e) => {
           eprintln!("Fatal: Dictionary failed to load: {}", e);
           std::process::exit(1);
       }
   };

   let start_time = Instant::now();

   let mut board_state = create_empty_board();
   board_state.insert("g9".to_string(), Some('b'));
   board_state.insert("h8".to_string(), Some('r'));
   board_state.insert("h9".to_string(), Some('e'));
   board_state.insert("h10".to_string(), Some('s'));
   board_state.insert("h11".to_string(), Some('e'));
   board_state.insert("h12".to_string(), Some('t'));
   board_state.insert("i9".to_string(), Some('l'));
   board_state.insert("j9".to_string(), Some('i'));
   board_state.insert("k9".to_string(), Some('e'));
   board_state.insert("l9".to_string(), Some('v'));
   board_state.insert("m6".to_string(), Some('t'));
   board_state.insert("m7".to_string(), Some('a'));
   board_state.insert("m8".to_string(), Some('t'));
   board_state.insert("m9".to_string(), Some('e'));


   let player_hand = vec!['a', 'b', 'd', 'u', 'c', 't', 's'];
   // let player_hand = vec!['q', 'u', 'a', 'k', 'i', 'n', 'g'];
   // let player_hand = vec!['a', 'e', 'r', 't', 's', '_', 'n'];

   println!("Testing with hand: {:?}", player_hand);

   let solver = ScrabbleSolver::new(board_state, player_hand, dictionary);

   let valid_moves = solver.get_all_valid_moves();

   let duration = start_time.elapsed();

   println!("\nFound {} valid moves.", valid_moves.len());
   println!("Execution time: {:.3?} seconds", duration.as_secs_f64());

   println!("\nTop 20 Moves (by score):");
   for (i, mv) in valid_moves.iter().take(20).enumerate() {
       print!(
           "  {}. {} @ {} ({:?}) - Score: {}",
           i + 1,
           mv.word,
           mv.start_space,
           mv.direction,
           mv.score
       );
       if !mv.blank_assignments.is_empty() {
           let blanks_str: String = mv
               .blank_assignments
               .iter()
               .map(|(pos, letter)| format!("{}={}", pos, letter))
               .collect::<Vec<_>>()
               .join(", ");
           print!(" (Blanks: {})", blanks_str);
       }
        if !mv.cross_words.is_empty() {
           print!(" Cross: {:?}", mv.cross_words.iter().map(|cw| &cw.word).collect::<Vec<_>>());
        }
       println!();
   }
}