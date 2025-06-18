# Valentin Macé
# valentin.mace@kedgebs.com
# Developed for fun
# Feel free to use this code as you wish as long as you quote me as author

"""
train.py
~~~~~~~~~~

Headless training script for neural networks using genetic algorithm
Run this script to train new neural networks without GUI
"""

import os
os.environ['SDL_VIDEODRIVER'] = 'dummy'  # Disable pygame video
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'  # Hide pygame messages

import argparse
import time
from genetic_algorithm import GeneticAlgorithm
try:
    from device_utils import get_device, print_device_info
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available. Training will use CPU only.")

def choose_device_interactive():
    """
    Интерактивный выбор устройства для обучения
    """
    print("\n" + "🚀" + " ВЫБОР УСТРОЙСТВА ДЛЯ ОБУЧЕНИЯ " + "🚀")
    print("=" * 50)
    
    options = []
    
    # Проверяем доступные устройства
    if TORCH_AVAILABLE:
        if torch.cuda.is_available():
            cuda_name = torch.cuda.get_device_name(0)
            options.append(("cuda", f"🚀 NVIDIA GPU: {cuda_name}"))
        if torch.backends.mps.is_available():
            options.append(("mps", "🍎 Apple Silicon (MPS)"))
    
    options.append(("cpu", "💻 CPU только"))
    options.append(("auto", "🤖 Автовыбор лучшего устройства"))
    
    print("\nДоступные устройства:")
    for i, (device, desc) in enumerate(options, 1):
        print(f"  {i}. {desc}")
    
    print("\nРекомендации:")
    print("  • GPU (CUDA/MPS) - быстрее для больших сетей")
    print("  • CPU - оптимально для малых сетей")
    print("  • Автовыбор - выберет лучшее доступное")
    
    while True:
        try:
            choice = input(f"\nВаш выбор (1-{len(options)}): ").strip()
            choice_num = int(choice)
            
            if 1 <= choice_num <= len(options):
                device_choice = options[choice_num - 1][0]
                device_name = options[choice_num - 1][1]
                print(f"✅ Выбрано: {device_name}")
                return device_choice
            else:
                print(f"❌ Введите число от 1 до {len(options)}")
        except ValueError:
            print("❌ Введите корректное число")
        except KeyboardInterrupt:
            print("\n\n👋 Обучение отменено")
            exit(0)

def choose_network_size_interactive():
    """
    Интерактивный выбор размера нейронной сети
    """
    print("\n" + "🧠" + " ВЫБОР РАЗМЕРА НЕЙРОННОЙ СЕТИ " + "🧠")
    print("=" * 50)
    
    options = [
        ("small", "🐁 Маленькая: 21→16→3 (~400 параметров)", [21, 16, 3]),
        ("medium", "🐕 Средняя: 21→64→32→3 (~3,488 параметров)", [21, 64, 32, 3]),
        ("large", "🐘 Большая: 21→128→64→32→3 (~13,024 параметров)", [21, 128, 64, 32, 3]),
        ("custom", "🔧 Пользовательская архитектура", None)
    ]
    
    print("\nДоступные размеры:")
    for i, (size, desc, _) in enumerate(options, 1):
        print(f"  {i}. {desc}")
    
    print("\nРекомендации по выбору размера:")
    print("  • 🐁 Маленькая - быстрое обучение, хорошо для CPU")
    print("  • 🐕 Средняя - баланс скорости и качества, эффективна на GPU")
    print("  • 🐘 Большая - лучшие результаты, требует GPU")
    print("  • 🔧 Пользовательская - для экспериментов")
    
    while True:
        try:
            choice = input(f"\nВаш выбор (1-{len(options)}): ").strip()
            choice_num = int(choice)
            
            if 1 <= choice_num <= len(options):
                size_choice, desc, shape = options[choice_num - 1]
                
                if size_choice == "custom":
                    print("\nВведите архитектуру сети через запятую")
                    print("Пример: 21,64,64,32,3")
                    custom_input = input("Архитектура: ").strip()
                    try:
                        shape = [int(x) for x in custom_input.split(',')]
                        if len(shape) < 2:
                            print("❌ Нужно минимум 2 слоя")
                            continue
                        if shape[0] != 21:
                            print("⚠️  Входной слой изменён с 21 на", shape[0])
                        if shape[-1] != 3:
                            print("⚠️  Выходной слой изменён с 3 на", shape[-1])
                    except:
                        print("❌ Неверный формат. Используйте числа через запятую")
                        continue
                
                total_params = sum(shape[i] * shape[i+1] for i in range(len(shape)-1))
                print(f"\n✅ Выбрана архитектура: {' → '.join(map(str, shape))}")
                print(f"   Всего параметров: ~{total_params:,}")
                return size_choice, shape
            else:
                print(f"❌ Введите число от 1 до {len(options)}")
        except ValueError:
            print("❌ Введите корректное число")
        except KeyboardInterrupt:
            print("\n\n👋 Обучение отменено")
            exit(0)

def parse_args():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description='Train Snake AI with GPU acceleration')
    parser.add_argument('--device', type=str, choices=['auto', 'cuda', 'mps', 'cpu', 'interactive'], 
                        default='interactive', help='Device to use: interactive (menu), auto, cuda, mps, cpu')
    parser.add_argument('--population-size', type=int, default=1000,
                        help='Number of networks per generation')
    parser.add_argument('--generations', type=int, default=100,
                        help='Number of generations to train')
    parser.add_argument('--crossover-rate', type=float, default=0.3,
                        help='Proportion of children produced')
    parser.add_argument('--mutation-rate', type=float, default=0.7,
                        help='Proportion of population to mutate')
    parser.add_argument('--no-gpu', action='store_true',
                        help='Force CPU-only training (disable GPU acceleration)')
    parser.add_argument('--mps-tournament-mode', type=str, choices=['hybrid', 'full'], 
                        default='hybrid', help='MPS tournament mode: hybrid (CPU tournaments) or full (MPS everywhere)')
    parser.add_argument('--profile', action='store_true',
                        help='Enable detailed performance profiling')
    parser.add_argument('--network-size', type=str, choices=['small', 'medium', 'large', 'custom'], 
                        default='small', help='Neural network size: small (21,16,3), medium (21,64,32,3), large (21,128,64,32,3)')
    parser.add_argument('--custom-layers', type=str, 
                        help='Custom network layers (e.g., "21,64,64,32,3")')
    return parser.parse_args()

def main():
    """
    Main training function
    Runs genetic algorithm in headless mode (no display)
    """
    args = parse_args()
    
    print("Starting neural network training...")
    print("This will run in headless mode (no graphics)")
    print("-" * 60)
    
    # Device setup
    device = None
    use_torch = TORCH_AVAILABLE and not args.no_gpu
    
    # Определяем устройство на основе аргументов
    if args.device == 'interactive':
        # Показываем интерактивное меню
        device_choice = choose_device_interactive()
    else:
        # Используем устройство из командной строки
        device_choice = args.device
        print(f"Устройство выбрано из командной строки: {device_choice}")
    
    if use_torch:
        if device_choice == 'auto':
            device = get_device()
        else:
            device = get_device(device_choice)
        print_device_info(device)
        
        if device.type in ['cuda', 'mps']:
            print(f"🚀 GPU acceleration enabled! Expected speedup: {get_speedup_estimate(device.type)}")
        else:
            print("ℹ️  Using CPU (consider installing PyTorch with CUDA/MPS for faster training)")
    else:
        print("ℹ️  Using CPU only (PyTorch not available or disabled)")
    
    # Determine network shape
    network_shapes = {
        'small': [21, 16, 3],           # Original: ~400 parameters
        'medium': [21, 64, 32, 3],      # Medium: ~3,488 parameters  
        'large': [21, 128, 64, 32, 3]   # Large: ~13,024 parameters
    }
    
    # Интерактивный выбор размера сети если device был interactive
    if args.device == 'interactive':
        size_choice, network_shape = choose_network_size_interactive()
    elif args.network_size == 'custom' and args.custom_layers:
        try:
            network_shape = [int(x) for x in args.custom_layers.split(',')]
            if len(network_shape) < 2:
                raise ValueError("Need at least 2 layers")
        except:
            print("❌ Invalid custom layers format. Using small network.")
            network_shape = network_shapes['small']
    else:
        network_shape = network_shapes.get(args.network_size, network_shapes['small'])
    
    # Calculate total parameters
    total_params = sum(network_shape[i] * network_shape[i+1] for i in range(len(network_shape)-1))
    
    print(f"Network architecture: {' → '.join(map(str, network_shape))}")
    print(f"Total parameters: ~{total_params:,}")
    print(f"Population size: {args.population_size}")
    print(f"Generations: {args.generations}")
    print(f"Crossover rate: {args.crossover_rate}")
    print(f"Mutation rate: {args.mutation_rate}")
    print("-" * 60)
    
    # Create genetic algorithm with optimized parameters
    start_time = time.time()
    gen = GeneticAlgorithm(
        networks_shape=network_shape,
        population_size=args.population_size,
        generation_number=args.generations,
        crossover_rate=args.crossover_rate,
        crossover_method='neuron',
        mutation_rate=args.mutation_rate,
        mutation_method='weight',
        device=device,
        use_torch=use_torch,
        mps_tournament_mode=args.mps_tournament_mode,
        enable_profiling=args.profile
    )
    
    # Start training
    gen.start()
    
    # Performance summary
    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print("Training completed!")
    print(f"Total training time: {total_time:.2f} seconds ({total_time/60:.1f} minutes)")
    print(f"Average time per generation: {total_time/args.generations:.2f} seconds")
    print("Best networks saved in current directory as gen_X_weights.npy and gen_X_biases.npy")
    
    if use_torch and device is not None and device.type in ['cuda', 'mps']:
        print(f"\n🎉 GPU acceleration helped speed up training!")
        print(f"Device used: {device}")

def get_speedup_estimate(device_type):
    """
    Provide rough speedup estimates for different devices
    """
    if device_type == 'cuda':
        return "10-50x vs CPU"
    elif device_type == 'mps':
        return "3-10x vs CPU"
    else:
        return "1x (baseline)"

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        print("Partial results may be saved in current directory.")
    except Exception as e:
        print(f"\n\nError during training: {e}")
        print("Please check your PyTorch installation and GPU drivers.")
        raise
