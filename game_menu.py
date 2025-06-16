# Valentin Macé
# valentin.mace@kedgebs.com
# Developed for fun
# Feel free to use this code as you wish as long as you quote me as author

"""
game_menu.py
~~~~~~~~~~

Graphical menu for selecting neural networks or game modes using pygame
"""

import pygame
from constants import *
from neural_network import NeuralNetwork


class Button:
    """Simple button class for menu"""
    
    def __init__(self, x, y, width, height, text, color=(70, 70, 70), hover_color=(100, 100, 100), text_color=(255, 255, 255)):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.color = color
        self.hover_color = hover_color
        self.text_color = text_color
        self.font = pygame.font.Font(None, 32)
        self.hovered = False
    
    def handle_event(self, event):
        """Handle mouse events"""
        if event.type == pygame.MOUSEMOTION:
            self.hovered = self.rect.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                return True
        return False
    
    def draw(self, surface):
        """Draw the button"""
        color = self.hover_color if self.hovered else self.color
        pygame.draw.rect(surface, color, self.rect)
        pygame.draw.rect(surface, (255, 255, 255), self.rect, 2)
        
        text_surface = self.font.render(self.text, True, self.text_color)
        text_rect = text_surface.get_rect(center=self.rect.center)
        surface.blit(text_surface, text_rect)


class GameMenu:
    """Graphical menu for neural network selection"""
    
    def __init__(self):
        self.networks = {
            'joseph': {
                'name': 'Joseph',
                'description': 'Funniest to watch, always does something cool',
                'weights': 'saved/joseph_weights.npy',
                'biases': 'saved/joseph_biases.npy'
            },
            'valentin': {
                'name': 'Valentin',
                'description': 'Safe and precise',
                'weights': 'saved/valentin_weights.npy',
                'biases': 'saved/valentin_biases.npy'
            },
            'larry': {
                'name': 'Larry',
                'description': 'Very safe but also the best network',
                'weights': 'saved/larry_weights.npy',
                'biases': 'saved/larry_biases.npy'
            },
            'gen35': {
                'name': 'Gen 35',
                'description': 'Trained network from generation 35',
                'weights': 'gen_35_weights.npy',
                'biases': 'gen_35_biases.npy'
            }
        }
        
        self.buttons = []
        self.setup_buttons()
    
    def setup_buttons(self):
        """Create menu buttons"""
        button_width = 400
        button_height = 50
        start_y = 150
        spacing = 70
        center_x = WINDOW_SIZE - button_width // 2
        
        # Neural network buttons
        y_pos = start_y
        for key, network in self.networks.items():
            button = Button(
                center_x, y_pos, button_width, button_height,
                f"{network['name']} - {network['description'][:25]}...",
                color=(50, 100, 50), hover_color=(70, 140, 70)
            )
            button.network_key = key
            self.buttons.append(button)
            y_pos += spacing
        
        # Manual play button
        manual_button = Button(
            center_x, y_pos, button_width, button_height,
            "Play Manually",
            color=(100, 50, 50), hover_color=(140, 70, 70)
        )
        manual_button.action = 'manual'
        self.buttons.append(manual_button)
        y_pos += spacing
        
        # Exit button
        exit_button = Button(
            center_x, y_pos, button_width, button_height,
            "Exit",
            color=(80, 80, 80), hover_color=(120, 120, 120)
        )
        exit_button.action = 'exit'
        self.buttons.append(exit_button)
    
    def handle_events(self, events):
        """Handle menu events"""
        for event in events:
            for button in self.buttons:
                if button.handle_event(event):
                    if hasattr(button, 'network_key'):
                        return 'network', button.network_key
                    elif hasattr(button, 'action'):
                        return button.action, None
        return None, None
    
    def draw(self, surface):
        """Draw the menu"""
        # Background
        surface.fill((30, 30, 30))
        
        # Title
        title_font = pygame.font.Font(None, 64)
        title_text = title_font.render("SNAKE AI", True, (255, 255, 255))
        title_rect = title_text.get_rect(center=(WINDOW_SIZE, 100))
        surface.blit(title_text, title_rect)
        
        # Subtitle
        subtitle_font = pygame.font.Font(None, 32)
        subtitle_text = subtitle_font.render("Choose Neural Network", True, (200, 200, 200))
        subtitle_rect = subtitle_text.get_rect(center=(WINDOW_SIZE, 140))
        surface.blit(subtitle_text, subtitle_rect)
        
        # Buttons
        for button in self.buttons:
            button.draw(surface)
    
    def load_network(self, network_key):
        """Load and return selected neural network"""
        if network_key in self.networks:
            network_info = self.networks[network_key]
            net = NeuralNetwork()
            try:
                net.load(filename_weights=network_info['weights'], 
                        filename_biases=network_info['biases'])
                return net
            except FileNotFoundError:
                print(f"Error: Could not find network files for {network_info['name']}")
                return None
        return None