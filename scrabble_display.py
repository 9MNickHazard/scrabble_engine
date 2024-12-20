import pygame
import sys

pygame.init()

# constants
CELL_SIZE = 50
HAND_CELL_SIZE = 40
BOARD_SIZE = 15
BOARD_PIXELS = CELL_SIZE * BOARD_SIZE
HAND_HEIGHT = 120
LABEL_MARGIN = 25
SCORE_HEIGHT = 40
WINDOW_SIZE = (
    BOARD_PIXELS + (LABEL_MARGIN * 2),
    BOARD_PIXELS + HAND_HEIGHT + LABEL_MARGIN + SCORE_HEIGHT
)
TILE_COLOR = (219, 190, 143)
BOARD_COLOR = (245, 245, 220)
GRID_COLOR = (139, 69, 19)
FONT_COLOR = (0, 0, 0)
HAND_BG_COLOR = (230, 230, 230)
LETTER_POINT_VALUES = {
    'a': 1, 'b': 3, 'c': 3, 'd': 2, 'e': 1, 'f': 4, 'g': 2, 'h': 4, 'i': 1, 'j': 8, 
    'k': 5, 'l': 1, 'm': 3, 'n': 1, 'o': 1, 'p': 3, 'q': 10, 'r': 1, 's': 1, 't': 1, 
    'u': 1, 'v': 4, 'w': 4, 'x': 8, 'y': 4, 'z': 10
}

# special squares colors
TRIPLE_WORD = (255, 0, 0)      # red
DOUBLE_WORD = (255, 182, 193)  # pink
TRIPLE_LETTER = (0, 0, 255)    # blue
DOUBLE_LETTER = (173, 216, 230) # light blue

# initialize window
screen = pygame.display.set_mode(WINDOW_SIZE)
pygame.display.set_caption("Scrabble Board")

# initialize fonts
board_font = pygame.font.Font(None, 36)
hand_font = pygame.font.Font(None, 32)
player_font = pygame.font.Font(None, 32)
point_font = pygame.font.Font(None, 20)

def get_square_color(row, col):
    # triple word score squares
    if (row, col) in [(0, 0), (0, 7), (0, 14), (7, 0), (7, 14), (14, 0), (14, 7), (14, 14)]:
        return TRIPLE_WORD
    
    # double word score squares
    if (row, col) in [(1, 1), (2, 2), (3, 3), (4, 4), (7, 7), (13, 13), (12, 12), (11, 11), (10, 10),
                      (1, 13), (2, 12), (3, 11), (4, 10), (13, 1), (12, 2), (11, 3), (10, 4)]:
        return DOUBLE_WORD
    
    # triple letter score squares
    if (row, col) in [(1, 5), (1, 9), (5, 1), (5, 5), (5, 9), (5, 13),
                      (9, 1), (9, 5), (9, 9), (9, 13), (13, 5), (13, 9)]:
        return TRIPLE_LETTER
    
    # double letter score squares
    if (row, col) in [(0, 3), (0, 11), (2, 6), (2, 8), (3, 0), (3, 7), (3, 14),
                      (6, 2), (6, 6), (6, 8), (6, 12), (7, 3), (7, 11),
                      (8, 2), (8, 6), (8, 8), (8, 12), (11, 0), (11, 7), (11, 14),
                      (12, 6), (12, 8), (14, 3), (14, 11)]:
        return DOUBLE_LETTER
    
    return BOARD_COLOR

def draw_grid_labels():
    # draw rows
    for i in range(15):
        label = player_font.render(str(i + 1), True, FONT_COLOR)
        label_rect = label.get_rect(center=(LABEL_MARGIN + (i * CELL_SIZE) + CELL_SIZE//2, LABEL_MARGIN//2 + SCORE_HEIGHT))
        screen.blit(label, label_rect)

    # draw columns
    for i in range(15):
        label = player_font.render(chr(65 + i), True, FONT_COLOR)
        label_rect = label.get_rect(center=(LABEL_MARGIN//2, LABEL_MARGIN + (i * CELL_SIZE) + CELL_SIZE//2 + SCORE_HEIGHT))
        screen.blit(label, label_rect)


def draw_scores(player_1_score, player_2_score):
    # draw background for scores
    pygame.draw.rect(screen, HAND_BG_COLOR, (0, 0, WINDOW_SIZE[0], SCORE_HEIGHT))
    pygame.draw.line(screen, GRID_COLOR, (0, SCORE_HEIGHT), (WINDOW_SIZE[0], SCORE_HEIGHT), 2)
    
    # draw scores
    p1_score = player_font.render(f"Player 1 Score: {player_1_score}", True, FONT_COLOR)
    p2_score = player_font.render(f"Player 2 Score: {player_2_score}", True, FONT_COLOR)
    
    screen.blit(p1_score, (10, 10))
    screen.blit(p2_score, (WINDOW_SIZE[0] - p2_score.get_rect().width - 10, 10))


def draw_tile(x, y, letter, size, font):
    pygame.draw.rect(screen, TILE_COLOR, (x, y, size, size))
    pygame.draw.rect(screen, GRID_COLOR, (x, y, size, size), 1)
    
    if letter:
        # draw the main letter
        text = font.render(letter.upper(), True, FONT_COLOR)
        text_rect = text.get_rect(center=(x + size//2, y + size//2))
        screen.blit(text, text_rect)
        
        # draw the point value
        point_value = LETTER_POINT_VALUES[letter.lower()]
        point_text = point_font.render(str(point_value), True, FONT_COLOR)
        point_rect = point_text.get_rect(bottomright=(x + size - 2, y + size - 2))  # -2 for padding
        screen.blit(point_text, point_rect)


def draw_board(is_tile_present):
    # draw board squares w/ special colors
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            x = col * CELL_SIZE + LABEL_MARGIN
            y = row * CELL_SIZE + LABEL_MARGIN + SCORE_HEIGHT
            color = get_square_color(row, col)
            pygame.draw.rect(screen, color, (x, y, CELL_SIZE, CELL_SIZE))
            pygame.draw.rect(screen, GRID_COLOR, (x, y, CELL_SIZE, CELL_SIZE), 1)
    
    # draw tiles and letters
    for position, (is_present, letter) in is_tile_present.items():
        if is_present:
            row = ord(position[0].lower()) - ord('a')
            col = int(position[1:]) - 1
            x = col * CELL_SIZE + LABEL_MARGIN
            y = row * CELL_SIZE + LABEL_MARGIN + SCORE_HEIGHT
            draw_tile(x, y, letter, CELL_SIZE, board_font)


def draw_hands(player_1_hand, player_2_hand):
    # draw background for hands section
    pygame.draw.rect(screen, HAND_BG_COLOR, (0, BOARD_PIXELS, WINDOW_SIZE[0], HAND_HEIGHT))
    
    # draw horizontal line separating board from hands
    pygame.draw.line(screen, GRID_COLOR, (0, BOARD_PIXELS), (WINDOW_SIZE[0], BOARD_PIXELS), 2)
    
    # draw vertical line separating hands
    midpoint_x = WINDOW_SIZE[0] // 2
    pygame.draw.line(screen, (0, 0, 0), 
                    (midpoint_x, BOARD_PIXELS), 
                    (midpoint_x, WINDOW_SIZE[1]), 
                    3)  # Thicker black line
    
    # starting positions for each players hand
    hand_y = BOARD_PIXELS + 10
    p1_start_x = 10
    p2_start_x = WINDOW_SIZE[0] // 2 + 10
    
    # draw player labels
    p1_label = player_font.render("Player 1:", True, FONT_COLOR)
    p2_label = player_font.render("Player 2:", True, FONT_COLOR)
    screen.blit(p1_label, (p1_start_x, hand_y))
    screen.blit(p2_label, (p2_start_x, hand_y))
    
    # draw P1 hand
    for i, letter in enumerate(player_1_hand):
        x = p1_start_x + (i * (HAND_CELL_SIZE + 5))
        y = hand_y + 30
        draw_tile(x, y, letter, HAND_CELL_SIZE, hand_font)
    
    # draw P2 hand
    for i, letter in enumerate(player_2_hand):
        x = p2_start_x + (i * (HAND_CELL_SIZE + 5))
        y = hand_y + 30
        draw_tile(x, y, letter, HAND_CELL_SIZE, hand_font)


def display_loop(is_tile_present, player_1_hand, player_2_hand, player_1_score=0, player_2_score=0):
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
    
    screen.fill(BOARD_COLOR)
    draw_scores(player_1_score, player_2_score)
    draw_grid_labels()
    draw_board(is_tile_present)
    draw_hands(player_1_hand, player_2_hand)
    pygame.display.flip()

