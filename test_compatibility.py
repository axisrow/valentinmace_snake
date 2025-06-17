#!/usr/bin/env python3
# Valentin Macé
# valentin.mace@kedgebs.com
# Test script to verify compatibility with existing saved models

"""
Test script to verify that the updated neural network implementation
can load and work with existing saved models (backward compatibility)
"""

import os
import numpy as np
from neural_network import NeuralNetwork
try:
    import torch
    from device_utils import get_device
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

def test_model_loading():
    """
    Test loading existing models with both NumPy and PyTorch implementations
    """
    print("Testing model compatibility...")
    print("=" * 50)
    
    # Find existing model files
    model_files = []
    for file in os.listdir("."):
        if file.endswith("_weights.npy"):
            base_name = file.replace("_weights.npy", "")
            bias_file = base_name + "_biases.npy"
            if os.path.exists(bias_file):
                model_files.append((file, bias_file, base_name))
    
    if not model_files:
        print("No existing model files found. Creating test model...")
        # Create a test model
        test_net = NeuralNetwork([21, 16, 3], use_torch=False)
        test_net.save("test_model")
        model_files = [("test_model_weights.npy", "test_model_biases.npy", "test_model")]
    
    print(f"Found {len(model_files)} model(s) to test")
    
    # Test each model
    for weights_file, biases_file, model_name in model_files[:3]:  # Test first 3 models
        print(f"\nTesting model: {model_name}")
        
        # Test NumPy loading
        try:
            net_numpy = NeuralNetwork(use_torch=False)
            net_numpy.load(weights_file, biases_file)
            print(f"  ✓ NumPy loading: SUCCESS")
            
            # Test forward pass
            test_input = np.random.randn(21, 1).astype(np.float32)
            output_numpy = net_numpy.feed_forward(test_input)
            print(f"  ✓ NumPy forward pass: SUCCESS (output shape: {output_numpy.shape})")
            
        except Exception as e:
            print(f"  ✗ NumPy loading: FAILED - {e}")
            continue
        
        # Test PyTorch loading if available
        if TORCH_AVAILABLE:
            try:
                device = get_device()
                net_torch = NeuralNetwork(use_torch=True, device=device)
                net_torch.load(weights_file, biases_file)
                print(f"  ✓ PyTorch loading: SUCCESS (device: {device})")
                
                # Test forward pass
                test_input_torch = torch.tensor(test_input, device=device, dtype=torch.float32)
                output_torch = net_torch.feed_forward(test_input_torch)
                output_torch_np = output_torch.detach().cpu().numpy()
                print(f"  ✓ PyTorch forward pass: SUCCESS (output shape: {output_torch.shape})")
                
                # Compare outputs (should be very similar)
                diff = np.abs(output_numpy - output_torch_np)
                max_diff = np.max(diff)
                print(f"  ✓ Output consistency: Max difference = {max_diff:.6f}")
                
                if max_diff < 1e-5:
                    print(f"  ✓ Perfect consistency between NumPy and PyTorch!")
                elif max_diff < 1e-3:
                    print(f"  ✓ Good consistency (small numerical differences)")
                else:
                    print(f"  ⚠ Large differences detected - check implementation")
                    
            except Exception as e:
                print(f"  ✗ PyTorch loading: FAILED - {e}")
        else:
            print(f"  - PyTorch testing: SKIPPED (not available)")
    
    print("\n" + "=" * 50)
    print("Compatibility test completed!")

def test_new_model_creation():
    """
    Test creating new models with both implementations
    """
    print("\nTesting new model creation...")
    print("-" * 30)
    
    # Test NumPy implementation
    try:
        net_numpy = NeuralNetwork([21, 16, 3], use_torch=False)
        test_input = np.random.randn(21, 1).astype(np.float32)
        output = net_numpy.feed_forward(test_input)
        print(f"✓ NumPy model creation: SUCCESS")
    except Exception as e:
        print(f"✗ NumPy model creation: FAILED - {e}")
    
    # Test PyTorch implementation
    if TORCH_AVAILABLE:
        try:
            device = get_device()
            net_torch = NeuralNetwork([21, 16, 3], use_torch=True, device=device)
            test_input = torch.randn(21, 1, device=device, dtype=torch.float32)
            output = net_torch.feed_forward(test_input)
            print(f"✓ PyTorch model creation: SUCCESS (device: {device})")
        except Exception as e:
            print(f"✗ PyTorch model creation: FAILED - {e}")
    else:
        print(f"- PyTorch model creation: SKIPPED (not available)")

if __name__ == "__main__":
    test_model_loading()
    test_new_model_creation()
    
    print("\n🎉 All tests completed!")
    if TORCH_AVAILABLE:
        device = get_device()
        if device.type in ['cuda', 'mps']:
            print(f"🚀 GPU acceleration is ready: {device}")
        else:
            print(f"💻 Running on CPU: {device}")
    else:
        print(f"💻 PyTorch not available - CPU only mode")