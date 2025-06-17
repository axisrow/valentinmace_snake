#!/usr/bin/env python3
"""
network_loader.py
~~~~~~~~~~~~~~~~~

Neural Network Loader for Snake Game
Provides a text-based menu to select and load neural networks from the versions/ folder.

Usage: python network_loader.py
"""

import os
import sys
from neural_network import NeuralNetwork
from game import Game


class NetworkLoader:
    """Text-based neural network loader and runner"""
    
    def __init__(self, versions_folder="versions"):
        self.versions_folder = versions_folder
        self.available_networks = {}
        self.scan_networks()
    
    def scan_networks(self):
        """Scan the versions folder for available neural networks"""
        if not os.path.exists(self.versions_folder):
            print(f"Error: {self.versions_folder} folder not found!")
            return
        
        files = os.listdir(self.versions_folder)
        
        # Find matching weight and bias files
        weights_files = [f for f in files if f.endswith('_weights.npy')]
        
        for weights_file in weights_files:
            base_name = weights_file.replace('_weights.npy', '')
            bias_file = f"{base_name}_biases.npy"
            
            if bias_file in files:
                weights_path = os.path.join(self.versions_folder, weights_file)
                bias_path = os.path.join(self.versions_folder, bias_file)
                
                self.available_networks[base_name] = {
                    'weights': weights_path,
                    'biases': bias_path,
                    'display_name': self.format_display_name(base_name)
                }
    
    def format_display_name(self, base_name):
        """Format network name for display"""
        if base_name.startswith('gen_'):
            # Extract generation number
            gen_num = base_name.replace('gen_', '')
            return f"Generation {gen_num}"
        else:
            # Custom named networks
            return base_name.replace('_', ' ').title()
    
    def display_menu(self):
        """Display the network selection menu"""
        if not self.available_networks:
            print("No neural networks found in versions/ folder!")
            print("Make sure you have matching *_weights.npy and *_biases.npy files.")
            return None
        
        print("\n" + "="*50)
        print("🐍 SNAKE AI - NEURAL NETWORK LOADER")
        print("="*50)
        print("\nAvailable Neural Networks:")
        print("-" * 30)
        
        # Sort networks by generation number if possible
        sorted_networks = sorted(self.available_networks.items(), 
                                key=lambda x: self.sort_key(x[0]))
        
        for i, (key, info) in enumerate(sorted_networks, 1):
            print(f"{i:2d}. {info['display_name']}")
        
        print(f"\n{len(sorted_networks) + 1:2d}. Exit")
        print("-" * 30)
        
        return sorted_networks
    
    def sort_key(self, network_name):
        """Generate sort key for network ordering"""
        if network_name.startswith('gen_'):
            try:
                # Extract number and sort numerically
                num_str = network_name.replace('gen_', '')
                # Handle names like "gen_alex35" 
                if num_str.isdigit():
                    return (0, int(num_str))  # Generations first
                else:
                    return (1, num_str)  # Named generations second
            except:
                return (2, network_name)  # Fallback
        else:
            return (3, network_name)  # Other networks last
    
    def get_user_choice(self, max_choice):
        """Get and validate user input"""
        while True:
            try:
                choice = input(f"\nSelect network (1-{max_choice}): ").strip()
                choice_num = int(choice)
                
                if 1 <= choice_num <= max_choice:
                    return choice_num
                else:
                    print(f"Please enter a number between 1 and {max_choice}")
            except ValueError:
                print("Please enter a valid number")
            except KeyboardInterrupt:
                print("\nExiting...")
                return None
    
    def load_network(self, network_key):
        """Load a neural network from files"""
        try:
            network_info = self.available_networks[network_key]
            
            # Create neural network and load weights/biases
            neural_net = NeuralNetwork([21, 16, 3])  # Default shape
            neural_net.load(network_info['weights'], network_info['biases'])
            
            print(f"✅ Successfully loaded: {network_info['display_name']}")
            return neural_net
            
        except Exception as e:
            print(f"❌ Error loading network: {e}")
            return None
    
    def run_game(self, neural_net, network_name):
        """Run the game with selected neural network"""
        print(f"\n🎮 Starting game with {network_name}...")
        print("Press ESC to quit the game or close the window")
        print("-" * 30)
        
        game_finished = False
        try:
            game = Game()
            score = game.start(display=True, neural_net=neural_net, speed=10, show_menu=False)
            game_finished = True
            print(f"\n🏆 Game finished! Final score: {score}")
            
        except KeyboardInterrupt:
            print("\n⚠️ Game interrupted by user")
            game_finished = True
        except Exception as e:
            print(f"❌ Error running game: {e}")
            game_finished = True
        finally:
            # Ensure we always return to a clean state
            if not game_finished:
                print("\n⚠️ Game terminated unexpectedly")
            # Force cleanup of any remaining pygame resources
            try:
                import pygame
                if pygame.get_init():
                    pygame.quit()
            except:
                pass
    
    def main_loop(self):
        """Main program loop"""
        while True:
            try:
                sorted_networks = self.display_menu()
                
                if not sorted_networks:
                    break
                
                max_choice = len(sorted_networks) + 1
                choice = self.get_user_choice(max_choice)
                
                if choice is None:  # Ctrl+C
                    break
                elif choice == max_choice:  # Exit option
                    print("Goodbye! 🐍")
                    break
                else:
                    # Load and run selected network
                    network_key, network_info = sorted_networks[choice - 1]
                    neural_net = self.load_network(network_key)
                    
                    if neural_net:
                        self.run_game(neural_net, network_info['display_name'])
                        # Give user option to continue or exit after game
                        try:
                            user_input = input("\nPress Enter to return to menu (or 'q' to quit): ").strip().lower()
                            if user_input == 'q':
                                print("Goodbye! 🐍")
                                break
                        except (KeyboardInterrupt, EOFError):
                            print("\nGoodbye! 🐍")
                            break
            except KeyboardInterrupt:
                print("\nProgram interrupted. Goodbye! 🐍")
                break
            except Exception as e:
                print(f"Unexpected error in main loop: {e}")
                print("Continuing...")
                continue


def main():
    """Main entry point"""
    # Suppress macOS pygame warnings
    import os
    os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
    
    print("Initializing Snake AI Network Loader...")
    
    try:
        loader = NetworkLoader()
        loader.main_loop()
    except KeyboardInterrupt:
        print("\nProgram interrupted by user. Goodbye! 🐍")
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()