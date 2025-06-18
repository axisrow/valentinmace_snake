#!/usr/bin/env python3
# Valentin Macé
# Quick benchmark script to compare CPU vs MPS performance

"""
benchmark_devices.py
~~~~~~~~~~

Quick benchmark to analyze why CPU might be faster than MPS
for small neural networks in Snake AI training.
"""

import sys
import os

# Disable pygame messages
os.environ['SDL_VIDEODRIVER'] = 'dummy'
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'

from performance_profiler import compare_cpu_vs_mps
import argparse

def main():
    parser = argparse.ArgumentParser(description='Benchmark CPU vs MPS performance for Snake AI')
    parser.add_argument('--network-size', type=str, choices=['small', 'medium', 'large'], 
                        default='small', help='Neural network size to test')
    parser.add_argument('--operations', type=int, default=1000,
                        help='Number of operations to benchmark')
    args = parser.parse_args()
    
    # Define network shapes
    network_shapes = {
        'small': [21, 16, 3],      # Current Snake AI size
        'medium': [21, 64, 32, 3], # Larger network
        'large': [21, 128, 64, 32, 3]  # Much larger network
    }
    
    shape = network_shapes[args.network_size]
    
    print("🔬 DEVICE PERFORMANCE BENCHMARK")
    print("=" * 50)
    print(f"Network shape: {shape}")
    print(f"Total parameters: ~{sum(shape[i]*shape[i+1] for i in range(len(shape)-1))}")
    print(f"Operations: {args.operations}")
    print()
    
    # Run comparison
    compare_cpu_vs_mps(shape, args.operations)
    
    print("\n📝 ANALYSIS:")
    print("-" * 30)
    print("If CPU is faster:")
    print("  • Network too small for GPU overhead")
    print("  • MPS initialization cost > computation gain")  
    print("  • Memory transfer overhead dominates")
    print()
    print("If MPS is faster:")
    print("  • GPU parallelization effective")
    print("  • Network size optimal for MPS")
    print("  • Memory bandwidth advantage")

if __name__ == "__main__":
    main()