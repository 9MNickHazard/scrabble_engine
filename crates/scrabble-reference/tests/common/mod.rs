//! Shared helpers for the tests and the timing example: a small seeded random generator, the tile
//! bag from RULES.md, a made-up lexicon of ordinary words, and boards built by playing games.
#![allow(dead_code)]

use std::path::PathBuf;

use scrabble_reference::{legal_plays, RefBoard, RefLexicon, RefPlay};

/// SplitMix64: small, seeded, good enough to pick test positions.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng(seed)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform-ish in `0..n`.
    pub fn below(&mut self, n: usize) -> usize {
        assert!(n > 0);
        (self.next_u64() % n as u64) as usize
    }

    pub fn shuffle<T>(&mut self, items: &mut [T]) {
        for i in (1..items.len()).rev() {
            let j = self.below(i + 1);
            items.swap(i, j);
        }
    }
}

/// Tile counts from RULES.md "Tiles" (`?` is the blank).
pub const DISTRIBUTION: [(u8, usize); 27] = [
    (b'A', 9),
    (b'B', 2),
    (b'C', 2),
    (b'D', 4),
    (b'E', 12),
    (b'F', 2),
    (b'G', 3),
    (b'H', 2),
    (b'I', 9),
    (b'J', 1),
    (b'K', 1),
    (b'L', 4),
    (b'M', 2),
    (b'N', 6),
    (b'O', 8),
    (b'P', 2),
    (b'Q', 1),
    (b'R', 6),
    (b'S', 4),
    (b'T', 6),
    (b'U', 4),
    (b'V', 2),
    (b'W', 2),
    (b'X', 1),
    (b'Y', 2),
    (b'Z', 1),
    (b'?', 2),
];

/// The 100 tiles, unshuffled.
pub fn full_bag() -> Vec<u8> {
    let mut bag = Vec::with_capacity(100);
    for &(tile, count) in &DISTRIBUTION {
        bag.extend(std::iter::repeat_n(tile, count));
    }
    bag
}

/// A rack of `letters` random letter tiles (drawn from the bag without blanks) plus `blanks` blanks.
pub fn random_rack(rng: &mut Rng, letters: usize, blanks: usize) -> String {
    let mut bag: Vec<u8> = full_bag().into_iter().filter(|&t| t != b'?').collect();
    rng.shuffle(&mut bag);
    let mut rack: Vec<u8> = bag[..letters].to_vec();
    rack.extend(std::iter::repeat_n(b'?', blanks));
    rack.sort();
    String::from_utf8(rack).unwrap()
}

/// How a simulated player picks its move.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Choice {
    /// Any legal play, uniformly.
    Random,
    /// The highest-scoring play.
    Greedy,
}

/// One player drawing from a shuffled bag and making up to `max_moves` placements. Returns the
/// board before the first move and after every move. When the rack has no play, it is thrown back
/// and redrawn (up to 10 times, while the bag has 7 tiles).
pub fn play_game(
    lexicon: &RefLexicon,
    rng: &mut Rng,
    max_moves: usize,
    choice: Choice,
) -> Vec<RefBoard> {
    let mut bag = full_bag();
    rng.shuffle(&mut bag);
    let mut rack: Vec<u8> = Vec::new();
    refill(&mut rack, &mut bag);
    let mut board = RefBoard::empty();
    let mut boards = vec![board.clone()];
    let mut redraws = 0;
    while boards.len() <= max_moves {
        let rack_text = String::from_utf8(rack.clone()).unwrap();
        let plays: Vec<RefPlay> = legal_plays(&board, &rack_text, lexicon);
        if plays.is_empty() {
            if bag.len() < 7 || redraws >= 10 {
                break;
            }
            redraws += 1;
            bag.append(&mut rack);
            rng.shuffle(&mut bag);
            refill(&mut rack, &mut bag);
            continue;
        }
        let play = match choice {
            Choice::Random => &plays[rng.below(plays.len())],
            Choice::Greedy => plays.iter().max_by_key(|p| p.score).unwrap(),
        };
        board.place(&play.tiles);
        for tile in &play.tiles {
            let wanted = if tile.blank { b'?' } else { tile.letter };
            let at = rack
                .iter()
                .position(|&t| t == wanted)
                .expect("legal_plays returned a play the rack cannot make");
            rack.remove(at);
        }
        refill(&mut rack, &mut bag);
        boards.push(board.clone());
        if rack.is_empty() {
            break;
        }
    }
    boards
}

fn refill(rack: &mut Vec<u8>, bag: &mut Vec<u8>) {
    while rack.len() < 7 {
        match bag.pop() {
            Some(tile) => rack.push(tile),
            None => break,
        }
    }
}

/// `SCRABBLE_LEXICON` if set, otherwise `data/lexicon/NSWL2023.txt` under the workspace root.
pub fn real_lexicon_path() -> PathBuf {
    if let Some(path) = std::env::var_os("SCRABBLE_LEXICON") {
        return PathBuf::from(path);
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("data")
        .join("lexicon")
        .join("NSWL2023.txt")
}

/// A made-up lexicon of ordinary English words, for positions that do not need the real list.
pub fn small_lexicon() -> RefLexicon {
    RefLexicon::from_words(SMALL_WORDS.split_whitespace())
}

pub const SMALL_WORDS: &str = "
AD AH AM AN AS AT AX BE BY DO GO HA HE HI HO ID IF IN IS IT LA LO MA ME MY NO OF OH ON OR OW OX
PA PI RE SO TO UP US WE YE YO
ACE ACT ADD AGE AGO AID AIM AIR ALE ANT ANY APE ARC ARE ARM ART ASH ASK ATE AWE AXE BAD BAG BAN
BAR BAT BED BEE BET BID BIG BIN BIT BOX BOY BUD BUG BUN BUS BUT BUY CAB CAN CAP CAR CAT COD COG
CON COT COW CRY CUB CUE CUP CUT DAD DAM DEN DEW DID DIE DIG DIM DIN DIP DOE DOG DOT DRY DUE DUG
EAR EAT EEL EGG ELF ELK EMU END ERA EVE EWE EYE FAN FAR FAT FED FEE FEW FIG FIN FIR FIT FIX FLU
FLY FOE FOG FOR FOX FRY FUN FUR GAP GAS GEL GEM GET GIN GOT GUM GUN GUT GUY HAD HAM HAS HAT HAY
HEN HER HID HIM HIP HIS HIT HOE HOG HOP HOT HOW HUB HUE HUG HUM HUT ICE ICY ILL INK INN ION IRE
ITS IVY JAB JAM JAR JAW JET JOB JOG JOT JOY JUG KEG KEY KID KIN KIT LAB LAD LAG LAP LAW LAY LED
LEG LET LID LIE LIP LIT LOG LOT LOW MAD MAN MAP MAT MAY MEN MET MID MIX MOB MOP MUD MUG NAG NAP
NET NEW NIB NIL NIP NOD NOR NOT NOW NUT OAK OAR OAT ODD ODE OFF OIL OLD ONE OPT ORB ORE OUR OUT
OWE OWL OWN PAD PAL PAN PAT PAW PAY PEA PEG PEN PET PIE PIG PIN PIT POD POT PRY PUB PUN PUP PUT
RAG RAM RAN RAP RAT RAW RAY RED RIB RID RIG RIM RIP ROB ROD ROE ROT ROW RUB RUG RUN RYE SAD SAG
SAP SAT SAW SAY SEA SEE SET SEW SHE SHY SIN SIP SIR SIT SIX SKI SKY SLY SOB SON SOW SOY SPA SPY
SUB SUM SUN TAB TAG TAN TAP TAR TAX TEA TEE TEN THE TIE TIN TIP TOE TON TOO TOP TOW TOY TRY TUB
TUG TWO URN USE VAN VAT VET VEX VIA VOW WAG WAR WAS WAX WAY WEB WED WET WHO WHY WIG WIN WIT WOE
WON WOO WOW YAK YAM YES YET YOU ZAP ZIP ZOO
ABLE ACHE ACRE AREA BAKE BALL BAND BANK BARE BARN BASE BATH BEAD BEAN BEAR BEAT BELL BEST BIKE
BIRD BITE BLUE BOAT BODY BOIL BOLD BONE BOOK BOOT BORN BOTH BOWL CAGE CAKE CALL CALM CAME CAMP
CANE CARD CARE CART CASE CASH CAST CATS COAT CODE COIN COLD COME CONE COOK COOL COPY CORE CORN
COST CREW CROP CURE DARE DARK DATE DAWN DEAL DEAR DEED DEEP DEER DENT DESK DIAL DICE DIET DINE
DIRT DISH DIVE DOES DONE DOOR DOSE DOTE DOVE DOWN DRAW DROP DRUM DUCK DUST EACH EARN EASE EAST
EASY EDGE EDIT EVEN EVER EXIT FACE FACT FADE FAIR FALL FARM FAST FATE FEAR FEED FEEL FEET FERN
FILE FILL FILM FIND FINE FIRE FISH FLAG FLAT FLEA FOAM FOLD FOOD FOOT FORM FORT FOUR FREE FROG
GAIN GAME GATE GAVE GEAR GIFT GIVE GLAD GLOW GLUE GOAL GOAT GOLD GONE GOOD GRIN GROW HAIR HALF
HALL HAND HARD HARE HATE HAVE HEAD HEAL HEAR HEAT HEEL HELD HELP HERB HERD HERE HERO HIDE HIKE
HILL HINT HIRE HOLD HOLE HOME HOOD HOOK HOPE HORN HOSE HOST HOUR HUNT HURT IDEA INTO IRON ITEM
JOIN JOKE JUMP JUST KEEP KICK KIND KING KITE KNEE KNOT LACE LAKE LAMB LAMP LAND LANE LAST LATE
LEAD LEAF LEAN LEFT LEND LIFE LIFT LIKE LIME LINE LINK LION LIST LIVE LOAD LOAN LOCK LONG LOOK
LORD LOSE LOST LOUD LOVE LUCK MADE MAIL MAIN MAKE MANY MARK MASK MATE MEAL MEAN MEAT MEET MELT
MILD MILE MILK MIND MINE MINT MIST MOON MORE MOST MOVE NAME NEAR NEAT NECK NEED NEST NEWS NICE
NINE NONE NOON NOSE NOTE OATS ONCE ONLY OPEN OVEN OVER PACE PACK PAGE PAID PAIN PAIR PALE PARK
PART PAST PATH PEAR PINE PINK PIPE PLAN PLAY PLOT POET POLE POND POOL POOR PORT POSE POST POUR
RACE RAIN RATE READ REAL REST RICE RICH RIDE RING RISE ROAD ROCK RODE ROLE ROOF ROOM ROOT ROPE
ROSE RULE SAFE SAID SAIL SALE SALT SAME SAND SAVE SEAL SEAT SEED SEEN SELF SEND SENT SHIP SHOE
SHOP SHOT SHOW SIDE SIGN SING SINK SITE SIZE SKIN SLOW SNOW SOAP SOFT SOIL SOLD SOME SONG SOON
SORT SOUP SPOT STAR STAY STEP STOP SUIT SURE SWIM TAIL TAKE TALE TALK TALL TAME TAPE TEAM TEAR
TELL TENT TEST TIDE TILE TIME TINY TIRE TOAD TONE TOOL TOWN TRAP TREE TRIP TRUE TUBE TUNE TURN
UNIT VASE VERY VEST VIEW VINE VOTE WADE WAIT WAKE WALK WALL WANT WARM WASH WAVE WEAR WEED WEEK
WELL WENT WEST WIDE WILD WIND WINE WING WIRE WISE WISH WOLF WOOD WOOL WORD WORE WORK YARD YEAR
ZERO ZONE
";
