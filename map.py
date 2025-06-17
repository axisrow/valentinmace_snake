# Valentin Macé
# valentin.mace@kedgebs.com
# Developed for fun
# Feel free to use this code as you wish as long as you quote me as author

"""
map.py
~~~~~~~~~~

This module is for building map for the snake game

The map:
- Contains its structure in a matrix form (see MAP in constants.py)
- Contains a instance of a snake and an instance of a food
- Is in charge of managing collisions, creation of food and giving vision to the snake

Notes:
- Some choices might seems weird in term of conception but I built it with priority for performance
      since the GA is greedy
"""

import math
import random
from snake import *


class Map:
    """Map class"""

    def __init__(self, snake, game=None):
        self.structure = MAP                                            # matrix of 0 and 1 representing the map
        self.snake = snake                                              # snake evolving in the map
        self.game = game                                                # reference to game for score tracking
        self.food = self.find_free_position()                          # food at a safe position

    def update(self):
        """
        Checks for collision between snake's head and walls or food
        Takes the right action in case of collision
        """
        snake_head_x, snake_head_y = self.snake.head
        snake_pos = self.structure[snake_head_y][snake_head_x]
        if [snake_head_x, snake_head_y] == self.food:                   # if snake's head is on food
            self.snake.grow()                                           # snake grows and new food is created
            if self.game:                                               # increase score if game reference exists
                self.game.increase_score()
            self.add_food()
        elif snake_pos == WALL:                                         # if snake's head is on wall, snek is ded
            self.snake.alive = False

    def find_free_position(self):
        """
        Finds a free position for food that is not occupied by snake's body or walls
        
        :return: list [x, y] coordinates of free position
        """
        max_attempts = 100  # Safety limit to prevent infinite loop
        attempts = 0
        
        while attempts < max_attempts:
            x = random.randint(1, SPRITE_NUMBER - 2)
            y = random.randint(1, SPRITE_NUMBER - 2)
            
            # Check if position is not a wall
            if self.structure[y][x] != WALL:
                # Check if position is not occupied by snake's body
                if [x, y] not in self.snake.body:
                    return [x, y]
            
            attempts += 1
        
        # Fallback: find any free position systematically if random failed
        for y in range(1, SPRITE_NUMBER - 1):
            for x in range(1, SPRITE_NUMBER - 1):
                if self.structure[y][x] != WALL and [x, y] not in self.snake.body:
                    return [x, y]
        
        # Last resort: return a safe position (should never happen in normal game)
        return [1, 1]

    def reset(self, snake, game=None):
        """
        Reset map to initial state for object reuse optimization
        
        :param snake: snake evolving in the map
        :param game: reference to game for score tracking
        """
        self.structure = MAP                                            # matrix of 0 and 1 representing the map
        self.snake = snake                                              # snake evolving in the map
        self.game = game                                                # reference to game for score tracking
        self.food = self.find_free_position()                          # food at a safe position

    def add_food(self):
        """
        Adds food at a free position that is not occupied by snake's body or walls
        """
        self.food = self.find_free_position()

    def render(self, window):
        """
        Renders the map (background, walls and food) on the game window and calls render() of snake
        Very very very unoptimized since render does not affect the genetic algorithm

        :param window: surface window
        """
        wall = pygame.image.load(IMAGE_WALL).convert()          # loading images
        food = pygame.image.load(IMAGE_FOOD).convert_alpha()

        window.fill([0,0,0])                # painting background
        num_line = 0
        for line in self.structure:         # running through the map structure
            num_case = 0
            for sprite in line:
                x = num_case * SPRITE_SIZE
                y = num_line * SPRITE_SIZE
                if sprite == 1:                         # displaying wall
                    window.blit(wall, (x, y))
                if self.food == [num_case, num_line]:   # displaying food
                    window.blit(food, (x, y))
                num_case += 1
            num_line += 1
        self.snake.render(window)         # snake will be rendered on above the map



    def scan(self):
        """
        Optimized scanning of the snake's environment into vision
        Uses pre-allocated arrays and cached calculations for performance

        Notes:
        - 7 first inputs are for walls, 7 next for food, 7 last for itself (its body)
        - Food is seen across all the map, walls and body are seen in range of 10 blocks max
        - Optimized to minimize function call overhead and distance calculations

        :return: nothing but gives vision to the snake
        """
        # Use pre-allocated scan array to avoid memory allocation
        if not hasattr(self, '_scan_cache'):
            self._scan_cache = [[0.0] for _ in range(21)]
        scan = self._scan_cache
        
        # Reset all values to 0
        for i in range(21):
            scan[i][0] = 0.0
            
        # Cache frequently accessed values
        structure = self.structure
        snake_body = self.snake.body                
        head_x = self.snake.head[0]
        head_y = self.snake.head[1]
        food_x = self.food[0]
        food_y = self.food[1]

        # Pre-calculate direction vectors
        forward_x = self.snake.direction[0]         
        forward_y = self.snake.direction[1]         
        right_x = -forward_y
        right_y = forward_x
        left_x = forward_y                          
        left_y = -forward_x                         
        forward_right_x = forward_x + right_x
        forward_right_y = forward_y + right_y
        forward_left_x = forward_x + left_x
        forward_left_y = forward_y + left_y         
        backward_right_x = -forward_left_x
        backward_right_y = -forward_left_y
        backward_left_x = -forward_right_x
        backward_left_y = -forward_right_y

        # Pre-calculate ranges to avoid repeated computation
        forward_range = min((20 - (forward_x * head_x + forward_y * head_y) - 1) % 19 + 1, 10)
        backward_range = min(21 - forward_range, 10)
        right_range = min((20 - (right_x * head_x + right_y * head_y) - 1) % 19 + 1, 10)
        left_range = min(21 - right_range, 10)
        forward_right_range = min(forward_range, right_range)
        forward_left_range = min(forward_range, left_range)
        backward_right_range = min(backward_range, right_range)
        backward_left_range = min(backward_range, left_range)

        # Optimized direction vectors and ranges
        directions = [
            (forward_x, forward_y, forward_range),
            (right_x, right_y, right_range),
            (left_x, left_y, left_range),
            (forward_right_x, forward_right_y, forward_right_range),
            (forward_left_x, forward_left_y, forward_left_range),
            (backward_right_x, backward_right_y, backward_right_range),
            (backward_left_x, backward_left_y, backward_left_range)
        ]

        # Optimized scanning with reduced function call overhead
        for idx, (dx, dy, max_range) in enumerate(directions):
            # Scan walls
            for i in range(1, min(max_range, 10)):
                step_x = head_x + i * dx
                step_y = head_y + i * dy
                if structure[step_y][step_x] == WALL:
                    scan[idx][0] = 1.0 / i  # Use simple distance instead of euclidean
                    break
            
            # Scan food
            for i in range(1, max_range):
                if food_x == (head_x + i * dx) and food_y == (head_y + i * dy):
                    scan[idx + 7][0] = 1.0
                    break
            
            # Scan snake body  
            for i in range(1, min(max_range, 10)):
                step_x = head_x + i * dx
                step_y = head_y + i * dy
                if [step_x, step_y] in snake_body:
                    scan[idx + 14][0] = max(scan[idx + 14][0], 1.0 / i)
                    break

        self.snake.vision = scan    # gives snake vision


@jit(nopython=True)
def distance(p1, p2):
    """
    Gives euclidian distance between two points
    @jit is used to speed up computation

    :param p1: origin point
    :param p2: end point
    :return: distance
    """
    return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)


