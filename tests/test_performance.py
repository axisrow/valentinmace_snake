# tests/test_performance.py
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game_menu import GameMenu
import pygame

def test_menu_setup_performance(benchmark):
    # Инициализация pygame для корректной работы GameMenu
    pygame.init()
    
    # Тестируем время создания меню
    benchmark(GameMenu)
    
    pygame.quit()

def test_menu_draw_performance(benchmark):
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    menu = GameMenu()
    
    # Тестируем время отрисовки меню
    benchmark(menu.draw, screen)
    
    pygame.quit()

