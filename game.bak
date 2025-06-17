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

    def run_invisible(self, neural_net=None):
        """
        Runs an undisplayed game played by a neural network.

        The game is played as fast as possible, you might want to fix a limit for the duration of the game,
        I advise to add 'or self.game_time > x' to the end condition.

        :param neural_net: NeuralNetwork that will play the game
        :return: int score achieved
        """
        self.score = 0                              # reset score for new game
        snake = Snake(neural_net=neural_net)        # creation of the snake and of its little brain
        map = Map(snake, self)                      # map creation with game reference

        cont = True                                 # main game loop
        while cont:
            self.game_time += 1
            map.scan()                              # gives vision of the environment to the snake
            snake.AI()                              # decision of the neural net (brain of the snake)
            snake.update()                          # snake is moving and aging
            map.update()                            # checking for collision of snake with walls or food
            if not snake.alive:
                cont = False
                self.game_time = 0
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

        :param neural_net: NeuralNetwork to provide for the game
        :param speed: game speed
        :return: int score achieved
        """
        self.score = 0                                                              # reset score for new game
        pygame.init()                                                               # pygame initialization
        game_window = pygame.display.set_mode((int(WINDOW_SIZE*2), WINDOW_SIZE))    # opens window
        pygame.display.set_caption(WINDOW_TITLE)

        snake = Snake(neural_net=neural_net)
        map = Map(snake, self)

        cont = [True]
        while cont[0]:                               # main game loop
            pygame.time.Clock().tick(speed)             # display speed
            self.inputs_management(cont)                # inputs handling (for quitting)
            map.scan()                                  # gives vision to the snake
            snake.AI()                                  # snake makes decision
            self.render(game_window, map)               # render the game
            snake.update()
            map.update()
            if not snake.alive:
                cont[0] = False

        self.game_score = snake.fitness()            # if the game is over, returns the score
        return self.game_score

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
        Keyboard inputs management (for quitting the game).
        """
        for event in pygame.event.get():
            if event.type == QUIT:
                cont[0] = False
            elif event.type == KEYDOWN:
                if event.key == K_ESCAPE:       # escape
                    cont[0] = False

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