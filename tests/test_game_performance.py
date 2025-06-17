# tests/test_game_performance.py
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game import Game

def test_run_invisible(benchmark):
    """Test performance of Game.run_invisible() without neural network in headless mode"""
    game = Game()
    
    # Benchmark the run_invisible method without neural network
    result = benchmark(game.run_invisible, neural_net=None)
    
    # Ensure the method returns a valid score  
    assert isinstance(result, (int, float))
    assert result >= 0