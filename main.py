# Valentin Macé
# valentin.mace@kedgebs.com
# Developed for fun
# Feel free to use this code as you wish as long as you quote me as author

"""
main.py
~~~~~~~~~~

Main file for this project

Interactive menu to choose which neural network to watch or play manually
"""

from game import Game

if __name__ == "__main__":
    game = Game()
    game.start(display=True, show_menu=True)