from map import *
from pygame.locals import QUIT, KEYDOWN, K_ESCAPE
from snake import *
from game_menu import GameMenu

class Game:
    """ Game Class """

    def __init__(self):
        self.game_score = 0     # contains the snake fitness at the end of the game
        self.game_time = 0      # number of iteration de game has been played (useful to stop long games)
        self.score = 0          # current game score (1 point per food eaten)
        self._cached_snake = None  # reusable snake object for performance
        self._cached_map = None    # reusable map object for performance

    def start(self, display=False, neural_net=None, speed=20, show_menu=True):
        """
        Wraps run_invisible and run_visible for simplicity when starting a game

        Note:
        Arguments are not checked for the sake of performance when starting a big amount
        of games (in the GA typically).

        :param display: boolean display or not the game
        :param neural_net: NeuralNetwork to provide for the game
        :param speed: game speed for displayed games
        :param show_menu: boolean show menu before game
        :return: int score achieved
        """
        if not display:
            return self.run_invisible(neural_net=neural_net)
        else:
            if show_menu:
                return self.run_with_menu(speed=speed)
            else:
                return self.run_visible(neural_net=neural_net, speed=speed)

    def run_invisible(self, neural_net=None, max_game_time=1000):
        """
        Runs an undisplayed game played by a neural network.

        The game is played as fast as possible with a time limit to prevent infinite loops.
        Uses object pooling to avoid memory allocation overhead.

        :param neural_net: NeuralNetwork that will play the game
        :param max_game_time: Maximum number of game iterations before timeout
        :return: int score achieved
        """
        self.score = 0                              # reset score for new game
        self.game_time = 0                          # reset game time
        
        # Object pooling optimization - reuse objects instead of creating new ones
        if self._cached_snake is None:
            self._cached_snake = Snake(neural_net=neural_net)
        else:
            self._cached_snake.reset(neural_net=neural_net)
            
        if self._cached_map is None:
            self._cached_map = Map(self._cached_snake, self)
        else:
            self._cached_map.reset(self._cached_snake, self)
        
        snake = self._cached_snake
        map = self._cached_map

        # optimized main game loop with time limit and early exit conditions
        snake_alive = snake.alive
        while snake_alive and self.game_time < max_game_time:
            self.game_time += 1
            map.scan()                              # gives vision of the environment to the snake
            snake.AI()                              # decision of the neural net (brain of the snake)
            snake.update()                          # snake is moving and aging
            map.update()                            # checking for collision of snake with walls or food
            snake_alive = snake.alive               # cache alive status for next iteration
            
            # early exit if snake is starving (stuck in loop)
            if snake.starve < 50:
                break
            
        self.game_score = snake.fitness()           # if the game is over, returns the score
        return self.game_score

    def increase_score(self):
        """
        Increases the game score by 1 point (called when food is eaten)
        """
        self.score += 1

    def run_visible(self, neural_net, speed=20):
        """
        Runs a displayed game played by a neural network.
        Enhanced with proper pygame cleanup to prevent hanging.

        :param neural_net: NeuralNetwork to provide for the game
        :param speed: game speed
        :return: int score achieved
        """
        pygame_initialized = False
        try:
            self.score = 0                                                              # reset score for new game
            pygame.init()                                                               # pygame initialization
            pygame_initialized = True
            game_window = pygame.display.set_mode((int(WINDOW_SIZE*2), WINDOW_SIZE))    # opens window
            pygame.display.set_caption(WINDOW_TITLE)

            snake = Snake(neural_net=neural_net)
            map = Map(snake, self)

            cont = [True]
            clock = pygame.time.Clock()  # Create clock object once
            
            while cont[0]:                               # main game loop
                clock.tick(speed)                           # display speed control
                self.inputs_management(cont)                # inputs handling (for quitting)
                
                # Only continue game logic if still running
                if not cont[0]:
                    break
                    
                map.scan()                                  # gives vision to the snake
                snake.AI()                                  # snake makes decision
                self.render(game_window, map)               # render the game
                snake.update()
                map.update()
                
                if not snake.alive:
                    cont[0] = False

            self.game_score = snake.fitness()            # if the game is over, returns the score
            return self.game_score
            
        except Exception as e:
            print(f"Error in run_visible: {e}")
            return 0
        finally:
            # Safe pygame cleanup
            if pygame_initialized:
                try:
                    pygame.display.quit()  # Close display first
                    pygame.quit()          # Then quit pygame
                except:
                    pass  # Ignore cleanup errors

    def run_with_menu(self, speed=20):
        """
        Runs the game with a graphical menu for selecting neural networks.
        
        :param speed: game speed
        :return: int score achieved
        """
        pygame.init()
        game_window = pygame.display.set_mode((int(WINDOW_SIZE*2), WINDOW_SIZE))
        pygame.display.set_caption(WINDOW_TITLE + " - Menu")
        
        menu = GameMenu()
        clock = pygame.time.Clock()
        
        while True:
            events = pygame.event.get()
            
            # Handle quit events
            for event in events:
                if event.type == QUIT:
                    pygame.quit()
                    return 0
                elif event.type == KEYDOWN and event.key == K_ESCAPE:
                    pygame.quit()
                    return 0
            
            # Handle menu events
            action, data = menu.handle_events(events)
            
            if action == 'exit':
                pygame.quit()
                return 0
            elif action == 'network':
                if isinstance(data, str):
                    network = menu.load_network(data)
                    if network:
                        pygame.display.set_caption(WINDOW_TITLE + f" - {menu.networks[data]['name']}")
                        score = self.run_visible(neural_net=network, speed=speed)
                        pygame.display.set_caption(WINDOW_TITLE + " - Menu")
                        # Return to menu after game
                        continue
            
            # Draw menu
            menu.draw(game_window)
            pygame.display.flip()
            clock.tick(60)

    def inputs_management(self, cont):
        """
        Enhanced keyboard and window event management (for quitting the game).
        Handles both ESC key and window close button properly.
        """
        for event in pygame.event.get():
            if event.type == QUIT:
                # Window close button clicked - exit immediately
                cont[0] = False
                return  # Early return to prevent further event processing
            elif event.type == KEYDOWN:
                if event.key == K_ESCAPE:       # escape key
                    cont[0] = False
                    return  # Early return for immediate exit

    def render(self, window, map):
        """
        Renders the game.
        Works by calling render() for the map which in turn will call render for the snake etc.
        """
        map.render(window)
        self.render_score(window)
        pygame.display.flip()

    def render_score(self, window):
        """
        Renders the current score on the game window.
        """
        font = pygame.font.Font(None, 36)
        score_text = font.render(f"Score: {self.score}", True, (255, 255, 255))
        # Position in top-left corner of the game area
        window.blit(score_text, (10, 10))