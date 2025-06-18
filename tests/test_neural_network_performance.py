# tests/test_neural_network_performance.py
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
import time
import psutil
import gc
from neural_network import NeuralNetwork

try:
    import torch
    from device_utils import get_device
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class TestNeuralNetworkPerformance:
    """Тесты производительности нейронной сети"""
    
    @pytest.fixture
    def network_shapes(self):
        """Различные архитектуры сетей для тестирования"""
        return {
            'small': [21, 16, 3],
            'medium': [21, 64, 32, 3],
            'large': [21, 128, 64, 32, 3]
        }
    
    @pytest.fixture
    def test_input(self):
        """Тестовый входной вектор"""
        return np.random.randn(21, 1).astype(np.float32)
    
    def get_memory_usage(self):
        """Получить текущее использование памяти в МБ"""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    
    @pytest.mark.benchmark(group="network_creation")
    @pytest.mark.parametrize("size", ["small", "medium", "large"])
    def test_network_creation_performance(self, benchmark, network_shapes, size):
        """Тест производительности создания нейронной сети"""
        shape = network_shapes[size]
        
        def create_network():
            return NeuralNetwork(shape, use_torch=False)
        
        result = benchmark(create_network)
        assert result is not None
        assert len(result.weights) == len(shape) - 1
    
    @pytest.mark.benchmark(group="feed_forward")
    @pytest.mark.parametrize("size", ["small", "medium", "large"])
    def test_feed_forward_performance(self, benchmark, network_shapes, test_input, size):
        """Тест производительности прямого прохода"""
        shape = network_shapes[size]
        network = NeuralNetwork(shape, use_torch=False)
        
        # Прогрев
        for _ in range(10):
            network.feed_forward(test_input)
        
        result = benchmark(network.feed_forward, test_input)
        assert result.shape == (shape[-1], 1)
    
    @pytest.mark.benchmark(group="network_clone")
    @pytest.mark.parametrize("size", ["small", "medium", "large"])
    def test_network_clone_performance(self, benchmark, network_shapes, size):
        """Тест производительности клонирования сети"""
        shape = network_shapes[size]
        network = NeuralNetwork(shape, use_torch=False)
        
        result = benchmark(network.clone)
        assert result is not None
        assert len(result.weights) == len(network.weights)
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    @pytest.mark.benchmark(group="cpu_vs_gpu")
    @pytest.mark.parametrize("size", ["small", "medium", "large"])
    def test_cpu_vs_gpu_performance(self, benchmark, network_shapes, test_input, size):
        """Сравнение производительности CPU vs GPU"""
        shape = network_shapes[size]
        device = get_device()
        
        # Пропускаем если нет GPU
        if device.type == 'cpu':
            pytest.skip("No GPU available for comparison")
        
        # CPU версия
        cpu_network = NeuralNetwork(shape, use_torch=False)
        cpu_times = []
        
        for _ in range(100):
            start = time.perf_counter()
            cpu_network.feed_forward(test_input)
            cpu_times.append(time.perf_counter() - start)
        
        # GPU версия
        gpu_network = NeuralNetwork(shape, device=device, use_torch=True)
        gpu_times = []
        
        # Прогрев GPU
        for _ in range(10):
            gpu_network.feed_forward(test_input)
        
        for _ in range(100):
            start = time.perf_counter()
            output = gpu_network.feed_forward(test_input)
            # Синхронизация для точного измерения
            if hasattr(output, 'cpu'):
                _ = output.cpu()
            gpu_times.append(time.perf_counter() - start)
        
        cpu_mean = np.mean(cpu_times)
        gpu_mean = np.mean(gpu_times)
        speedup = cpu_mean / gpu_mean
        
        print(f"\n{size.upper()} Network Performance:")
        print(f"  CPU mean time: {cpu_mean*1000:.3f}ms")
        print(f"  GPU mean time: {gpu_mean*1000:.3f}ms")
        print(f"  Speedup: {speedup:.2f}x")
        
        # Для больших сетей ожидаем ускорение на GPU
        if size == "large":
            assert speedup > 1.0, f"GPU should be faster for large networks, got {speedup:.2f}x"
    
    @pytest.mark.benchmark(group="batch_processing")
    @pytest.mark.parametrize("batch_size", [1, 10, 100, 1000])
    def test_batch_processing_scalability(self, benchmark, test_input, batch_size):
        """Тест масштабируемости при пакетной обработке"""
        network = NeuralNetwork([21, 64, 32, 3], use_torch=False)
        
        def process_batch():
            results = []
            for _ in range(batch_size):
                results.append(network.feed_forward(test_input))
            return results
        
        result = benchmark(process_batch)
        assert len(result) == batch_size
    
    @pytest.mark.benchmark(group="memory_usage")
    @pytest.mark.parametrize("size", ["small", "medium", "large"])
    def test_memory_usage(self, network_shapes, size):
        """Тест использования памяти для разных размеров сетей"""
        shape = network_shapes[size]
        
        # Сборка мусора перед измерением
        gc.collect()
        initial_memory = self.get_memory_usage()
        
        # Создаем несколько сетей
        networks = []
        for _ in range(10):
            networks.append(NeuralNetwork(shape, use_torch=False))
        
        # Измеряем память после создания
        gc.collect()
        final_memory = self.get_memory_usage()
        
        memory_per_network = (final_memory - initial_memory) / 10
        total_params = sum(shape[i] * shape[i+1] for i in range(len(shape)-1))
        
        print(f"\n{size.upper()} Network Memory Usage:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Memory per network: {memory_per_network:.2f} MB")
        print(f"  Bytes per parameter: {memory_per_network * 1024 * 1024 / total_params:.1f}")
        
        # Проверяем что использование памяти разумное
        assert memory_per_network < 10, f"Network using too much memory: {memory_per_network:.2f} MB"
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    @pytest.mark.benchmark(group="device_transfer")
    def test_device_transfer_performance(self, benchmark, network_shapes):
        """Тест производительности переноса между устройствами"""
        if not torch.cuda.is_available() and not torch.backends.mps.is_available():
            pytest.skip("No GPU available for device transfer test")
        
        shape = network_shapes['medium']
        device = get_device()
        
        # Создаем сеть на CPU
        cpu_network = NeuralNetwork(shape, device=torch.device('cpu'), use_torch=True)
        
        def transfer_to_gpu_and_back():
            # На GPU
            cpu_network.to_device(device)
            # Обратно на CPU
            cpu_network.to_device(torch.device('cpu'))
        
        benchmark(transfer_to_gpu_and_back)
    
    def test_performance_regression(self, network_shapes, test_input):
        """Тест на регрессию производительности"""
        shape = network_shapes['medium']
        network = NeuralNetwork(shape, use_torch=False)
        
        # Измеряем время 1000 прямых проходов
        start = time.perf_counter()
        for _ in range(1000):
            network.feed_forward(test_input)
        elapsed = time.perf_counter() - start
        
        # Ожидаемое время не более 1 секунды для 1000 проходов
        assert elapsed < 1.0, f"Performance regression detected: {elapsed:.2f}s for 1000 forward passes"
        
        print(f"\n1000 forward passes completed in {elapsed:.3f}s")
        print(f"Average time per pass: {elapsed/1000*1000:.3f}ms")


if __name__ == "__main__":
    # Запуск тестов с подробной статистикой
    pytest.main([__file__, "-v", "--benchmark-only", "--benchmark-columns=min,max,mean,stddev"])