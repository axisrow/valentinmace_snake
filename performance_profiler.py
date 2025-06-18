# Valentin Macé
# valentin.mace@kedgebs.com
# Performance profiling utilities for analyzing CPU vs MPS performance

"""
performance_profiler.py
~~~~~~~~~~

Utilities for profiling performance of different operations
to understand why CPU might be faster than MPS for small neural networks
"""

import time
from collections import defaultdict
from contextlib import contextmanager
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

class PerformanceProfiler:
    """Profile performance of different operations"""
    
    def __init__(self, enabled=True):
        self.enabled = enabled
        self.timings = defaultdict(list)
        self.current_timers = {}
        
    def reset(self):
        """Reset all timing data"""
        self.timings.clear()
        self.current_timers.clear()
        
    @contextmanager
    def timer(self, operation_name):
        """Context manager for timing operations"""
        if not self.enabled:
            yield
            return
            
        start_time = time.perf_counter()
        try:
            yield
        finally:
            end_time = time.perf_counter()
            duration = end_time - start_time
            self.timings[operation_name].append(duration)
            
    def get_stats(self, operation_name):
        """Get statistics for an operation"""
        if operation_name not in self.timings:
            return None
            
        times = self.timings[operation_name]
        return {
            'count': len(times),
            'total': sum(times),
            'mean': sum(times) / len(times),
            'min': min(times),
            'max': max(times)
        }
        
    def print_summary(self):
        """Print performance summary"""
        if not self.timings:
            print("No timing data collected")
            return
            
        print("\n" + "="*60)
        print("PERFORMANCE PROFILING SUMMARY")
        print("="*60)
        
        total_time = 0
        for operation in sorted(self.timings.keys()):
            stats = self.get_stats(operation)
            total_time += stats['total']
            print(f"\n{operation}:")
            print(f"  Count: {stats['count']}")
            print(f"  Total: {stats['total']:.3f}s")
            print(f"  Mean:  {stats['mean']:.3f}s")
            print(f"  Min:   {stats['min']:.3f}s")
            print(f"  Max:   {stats['max']:.3f}s")
            
        print(f"\nTotal measured time: {total_time:.3f}s")
        print("="*60)

def benchmark_neural_network_operations(device_type='cpu', network_shape=[21, 16, 3], num_operations=1000):
    """
    Benchmark basic neural network operations on different devices
    """
    if not TORCH_AVAILABLE:
        print("PyTorch not available for benchmarking")
        return None
        
    from neural_network import NeuralNetwork
    import numpy as np
    
    print(f"\n🔬 BENCHMARKING {device_type.upper()} PERFORMANCE")
    print(f"Network shape: {network_shape}")
    print(f"Operations: {num_operations}")
    print("-" * 50)
    
    # Setup device
    if device_type == 'mps' and torch.backends.mps.is_available():
        device = torch.device('mps')
    elif device_type == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        device_type = 'cpu'
    
    profiler = PerformanceProfiler()
    
    # Test 1: Network creation
    with profiler.timer('network_creation'):
        net = NeuralNetwork(network_shape, device=device, use_torch=(device_type != 'cpu'))
    
    # Test 2: Forward pass operations
    test_input = np.random.randn(network_shape[0], 1).astype(np.float32)
    
    # Warmup
    for _ in range(10):
        _ = net.feed_forward(test_input)
    
    # Benchmark forward passes
    for i in range(num_operations):
        with profiler.timer('feed_forward'):
            output = net.feed_forward(test_input)
            
        # Ensure computation is complete (for GPU)
        if hasattr(output, 'cpu'):
            with profiler.timer('gpu_sync'):
                _ = output.cpu()
    
    # Test 3: Network cloning
    with profiler.timer('network_clone'):
        for _ in range(10):
            clone = net.clone()
    
    # Test 4: Device transfer (if GPU)
    if device_type in ['mps', 'cuda']:
        cpu_device = torch.device('cpu')
        with profiler.timer('device_transfer_to_cpu'):
            for _ in range(10):
                cpu_net = net.clone()
                cpu_net.to_device(cpu_device)
        
        with profiler.timer('device_transfer_to_gpu'):
            for _ in range(10):
                gpu_net = cpu_net.clone()
                gpu_net.to_device(device)
    
    profiler.print_summary()
    return profiler

def compare_cpu_vs_mps(network_shape=[21, 16, 3], num_operations=1000):
    """
    Direct comparison of CPU vs MPS performance
    """
    if not TORCH_AVAILABLE or not torch.backends.mps.is_available():
        print("MPS not available for comparison")
        return
        
    print("\n" + "🏁" * 20)
    print("CPU vs MPS PERFORMANCE COMPARISON")
    print("🏁" * 20)
    
    # Benchmark CPU
    cpu_profiler = benchmark_neural_network_operations('cpu', network_shape, num_operations)
    
    # Benchmark MPS  
    mps_profiler = benchmark_neural_network_operations('mps', network_shape, num_operations)
    
    # Compare results
    print("\n" + "📊 COMPARISON RESULTS")
    print("-" * 50)
    
    operations = ['feed_forward', 'network_creation', 'network_clone']
    
    for op in operations:
        cpu_stats = cpu_profiler.get_stats(op)
        mps_stats = mps_profiler.get_stats(op)
        
        if cpu_stats and mps_stats:
            speedup = cpu_stats['mean'] / mps_stats['mean']
            winner = "MPS" if speedup > 1 else "CPU"
            print(f"\n{op}:")
            print(f"  CPU mean: {cpu_stats['mean']:.6f}s")
            print(f"  MPS mean: {mps_stats['mean']:.6f}s")
            print(f"  Speedup: {abs(speedup):.2f}x ({winner} faster)")

if __name__ == "__main__":
    # Run benchmark when script is executed directly
    compare_cpu_vs_mps()