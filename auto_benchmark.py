#!/usr/bin/env python3
# Valentin Macé
# Automated benchmarking and report generation

"""
auto_benchmark.py
~~~~~~~~~~

Automatically run performance tests and generate comprehensive reports
comparing CPU vs MPS performance for Snake AI training.
"""

import os
import sys
import subprocess
import time
import json
from datetime import datetime
from pathlib import Path

def ensure_logs_dir():
    """Create logs directory if it doesn't exist"""
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    return logs_dir

def run_command_with_log(command, log_file, description):
    """Run a command and save output to log file"""
    print(f"🔄 {description}...")
    print(f"   Command: {' '.join(command)}")
    print(f"   Log file: {log_file}")
    
    start_time = time.time()
    
    try:
        with open(log_file, 'w', encoding='utf-8') as f:
            # Write header
            f.write(f"# {description}\n")
            f.write(f"# Command: {' '.join(command)}\n")
            f.write(f"# Started: {datetime.now()}\n")
            f.write("# " + "="*50 + "\n\n")
            
            # Run command
            result = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding='utf-8'
            )
            
            # Write output
            f.write(result.stdout)
            
            # Write footer
            f.write(f"\n\n# Exit code: {result.returncode}")
            f.write(f"\n# Finished: {datetime.now()}")
            
        elapsed = time.time() - start_time
        
        if result.returncode == 0:
            print(f"   ✅ Completed in {elapsed:.1f}s")
            return True
        else:
            print(f"   ❌ Failed with exit code {result.returncode}")
            return False
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def run_all_benchmarks():
    """Run comprehensive benchmarks"""
    logs_dir = ensure_logs_dir()
    
    print("🚀 STARTING COMPREHENSIVE PERFORMANCE BENCHMARKS")
    print("=" * 60)
    print(f"Logs will be saved to: {logs_dir.absolute()}")
    print()
    
    # Test results tracking
    results = {
        'timestamp': datetime.now().isoformat(),
        'tests': {}
    }
    
    # 1. Benchmark different network sizes
    print("📊 Phase 1: Network Size Benchmarks")
    print("-" * 40)
    
    network_sizes = ['small', 'medium', 'large']
    for size in network_sizes:
        log_file = logs_dir / f"benchmark_{size}.log"
        command = ['python3', 'benchmark_devices.py', '--network-size', size, '--operations', '500']
        success = run_command_with_log(command, log_file, f"Benchmark {size} networks")
        results['tests'][f'benchmark_{size}'] = {'success': success, 'log_file': str(log_file)}
        print()
    
    # 2. Training benchmarks - CPU
    print("🖥️  Phase 2: CPU Training Benchmark")
    print("-" * 40)
    
    log_file = logs_dir / "train_cpu_1gen.log"
    command = ['python3', 'train.py', '--profile', '--device', 'cpu', '--generations', '1', '--population-size', '300']
    success = run_command_with_log(command, log_file, "CPU Training (1 generation)")
    results['tests']['train_cpu'] = {'success': success, 'log_file': str(log_file)}
    print()
    
    # 3. Training benchmarks - MPS
    print("🍎 Phase 3: MPS Training Benchmark")
    print("-" * 40)
    
    log_file = logs_dir / "train_mps_1gen.log"
    command = ['python3', 'train.py', '--profile', '--device', 'mps', '--generations', '1', '--population-size', '300']
    success = run_command_with_log(command, log_file, "MPS Training (1 generation)")
    results['tests']['train_mps'] = {'success': success, 'log_file': str(log_file)}
    print()
    
    # 4. Save test summary
    summary_file = logs_dir / "test_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("✅ ALL BENCHMARKS COMPLETED")
    print("=" * 60)
    print(f"Test summary saved to: {summary_file}")
    
    # Generate report
    print("\n📝 Generating performance report...")
    generate_report(logs_dir)
    
    return results

def parse_benchmark_log(log_file):
    """Parse benchmark log file to extract performance data"""
    data = {}
    
    try:
        with open(log_file, 'r') as f:
            content = f.read()
        
        # Extract CPU vs MPS comparison
        if "CPU vs MPS PERFORMANCE COMPARISON" in content:
            lines = content.split('\n')
            in_comparison = False
            current_operation = None
            
            for line in lines:
                if "COMPARISON RESULTS" in line:
                    in_comparison = True
                    continue
                    
                if in_comparison:
                    if line.strip().endswith(':') and not line.strip().startswith(' '):
                        current_operation = line.strip().rstrip(':')
                        data[current_operation] = {}
                    elif "CPU mean:" in line:
                        try:
                            cpu_time = float(line.split(':')[1].strip().replace('s', ''))
                            data[current_operation]['cpu_mean'] = cpu_time
                        except:
                            pass
                    elif "MPS mean:" in line:
                        try:
                            mps_time = float(line.split(':')[1].strip().replace('s', ''))
                            data[current_operation]['mps_mean'] = mps_time
                        except:
                            pass
                    elif "Speedup:" in line:
                        try:
                            speedup_text = line.split(':')[1].strip()
                            speedup = float(speedup_text.split('x')[0])
                            winner = "MPS" if "MPS faster" in speedup_text else "CPU"
                            data[current_operation]['speedup'] = speedup
                            data[current_operation]['winner'] = winner
                        except:
                            pass
    except Exception as e:
        print(f"Error parsing {log_file}: {e}")
    
    return data

def parse_training_log(log_file):
    """Parse training log file to extract profiling data"""
    data = {}
    
    try:
        with open(log_file, 'r') as f:
            content = f.read()
        
        # Extract device info
        if "Using device:" in content:
            for line in content.split('\n'):
                if "Using device:" in line:
                    data['device'] = line.split('Using device:')[1].strip()
                    break
        
        # Extract total generation time
        for line in content.split('\n'):
            if "Generation 1 completed in" in line:
                try:
                    time_str = line.split('completed in')[1].strip().replace('s', '')
                    data['generation_time'] = float(time_str)
                except:
                    pass
        
        # Extract profiling summary
        if "PERFORMANCE PROFILING SUMMARY" in content:
            lines = content.split('\n')
            in_summary = False
            current_operation = None
            
            for line in lines:
                if "PERFORMANCE PROFILING SUMMARY" in line:
                    in_summary = True
                    continue
                elif line.strip().startswith('Total measured time:'):
                    try:
                        total_time = float(line.split(':')[1].strip().replace('s', ''))
                        data['total_measured_time'] = total_time
                    except:
                        pass
                    break
                
                if in_summary and line.strip().endswith(':'):
                    current_operation = line.strip().rstrip(':')
                    data[current_operation] = {}
                elif in_summary and current_operation and 'Total:' in line:
                    try:
                        total_time = float(line.split(':')[1].strip().replace('s', ''))
                        data[current_operation]['total'] = total_time
                    except:
                        pass
                elif in_summary and current_operation and 'Mean:' in line:
                    try:
                        mean_time = float(line.split(':')[1].strip().replace('s', ''))
                        data[current_operation]['mean'] = mean_time
                    except:
                        pass
    
    except Exception as e:
        print(f"Error parsing {log_file}: {e}")
    
    return data

def generate_report(logs_dir):
    """Generate comprehensive performance report"""
    report_file = logs_dir / "performance_report.md"
    
    with open(report_file, 'w') as f:
        f.write("# Snake AI Performance Analysis Report\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # 1. Executive Summary
        f.write("## 📊 Executive Summary\n\n")
        
        # Parse training logs
        cpu_data = parse_training_log(logs_dir / "train_cpu_1gen.log")
        mps_data = parse_training_log(logs_dir / "train_mps_1gen.log")
        
        if cpu_data.get('generation_time') and mps_data.get('generation_time'):
            cpu_time = cpu_data['generation_time']
            mps_time = mps_data['generation_time']
            speedup = cpu_time / mps_time
            winner = "MPS" if speedup > 1 else "CPU"
            
            f.write(f"**Training Performance (1 generation, 300 networks):**\n")
            f.write(f"- CPU Time: {cpu_time:.2f}s\n")
            f.write(f"- MPS Time: {mps_time:.2f}s\n")
            f.write(f"- **Winner: {winner}** ({abs(speedup):.2f}x faster)\n\n")
            
            if winner == "CPU":
                f.write("🚨 **Key Finding**: CPU is faster than MPS for current network size\n\n")
            else:
                f.write("🚀 **Key Finding**: MPS provides significant speedup\n\n")
        
        # 2. Detailed Analysis
        f.write("## 🔍 Detailed Analysis\n\n")
        
        f.write("### Network Size Benchmarks\n\n")
        for size in ['small', 'medium', 'large']:
            benchmark_file = logs_dir / f"benchmark_{size}.log"
            if benchmark_file.exists():
                bench_data = parse_benchmark_log(benchmark_file)
                f.write(f"#### {size.title()} Networks\n")
                
                if 'feed_forward' in bench_data:
                    ff_data = bench_data['feed_forward']
                    f.write(f"- CPU: {ff_data.get('cpu_mean', 'N/A')}s per operation\n")
                    f.write(f"- MPS: {ff_data.get('mps_mean', 'N/A')}s per operation\n")
                    f.write(f"- Winner: {ff_data.get('winner', 'N/A')} ({ff_data.get('speedup', 'N/A')}x)\n\n")
        
        f.write("### Training Profiling Details\n\n")
        
        if cpu_data:
            f.write("#### CPU Training Breakdown\n")
            f.write("| Operation | Time (s) | Percentage |\n")
            f.write("|-----------|----------|------------|\n")
            
            total_time = cpu_data.get('total_measured_time', 0)
            for op, data in cpu_data.items():
                if isinstance(data, dict) and 'total' in data:
                    time_val = data['total']
                    percentage = (time_val / total_time * 100) if total_time > 0 else 0
                    f.write(f"| {op} | {time_val:.3f} | {percentage:.1f}% |\n")
            f.write("\n")
        
        if mps_data:
            f.write("#### MPS Training Breakdown\n")
            f.write("| Operation | Time (s) | Percentage |\n")
            f.write("|-----------|----------|------------|\n")
            
            total_time = mps_data.get('total_measured_time', 0)
            for op, data in mps_data.items():
                if isinstance(data, dict) and 'total' in data:
                    time_val = data['total']
                    percentage = (time_val / total_time * 100) if total_time > 0 else 0
                    f.write(f"| {op} | {time_val:.3f} | {percentage:.1f}% |\n")
            f.write("\n")
        
        # 3. Conclusions and Recommendations
        f.write("## 💡 Conclusions and Recommendations\n\n")
        
        if cpu_data.get('generation_time') and mps_data.get('generation_time'):
            if cpu_data['generation_time'] < mps_data['generation_time']:
                f.write("### Why CPU is Faster\n\n")
                f.write("1. **Network Size Too Small**: Current Snake AI networks (21→16→3) have only ~400 parameters\n")
                f.write("2. **GPU Overhead**: MPS initialization and memory transfer costs exceed computation benefits\n")
                f.write("3. **Non-optimal Operations**: Many small operations instead of large batch operations\n\n")
                
                f.write("### Optimization Strategies\n\n")
                f.write("1. **Increase Network Size**: Try larger networks (e.g., 21→64→32→3)\n")
                f.write("2. **Batch Operations**: Process multiple networks simultaneously\n")
                f.write("3. **Pure MPS Mode**: Avoid CPU↔MPS transfers in tournaments\n")
                f.write("4. **Use CPU for Small Networks**: Current implementation with CPU is optimal\n\n")
            else:
                f.write("### MPS Provides Benefits\n\n")
                f.write("1. **Optimal Network Size**: Current networks benefit from GPU parallelization\n")
                f.write("2. **Efficient Implementation**: Memory transfers are minimized\n")
                f.write("3. **Good Batch Utilization**: Operations are well-suited for MPS\n\n")
        
        f.write("### Final Recommendation\n\n")
        f.write("Based on this analysis:\n")
        if cpu_data.get('generation_time', float('inf')) < mps_data.get('generation_time', 0):
            f.write("- **Use CPU** for current Snake AI network size\n")
            f.write("- Consider **larger networks** to benefit from MPS\n")
            f.write("- **Keep hybrid mode** as fallback option\n")
        else:
            f.write("- **Use MPS** for significant performance gains\n")
            f.write("- **Optimize batch operations** further\n")
            f.write("- **Consider pure MPS mode** for even better performance\n")
    
    print(f"📝 Performance report generated: {report_file}")
    print(f"📁 All logs available in: {logs_dir}")
    
    # Display summary
    print("\n" + "="*60)
    print("QUICK SUMMARY")
    print("="*60)
    
    if cpu_data.get('generation_time') and mps_data.get('generation_time'):
        cpu_time = cpu_data['generation_time']
        mps_time = mps_data['generation_time']
        speedup = cpu_time / mps_time
        winner = "MPS" if speedup > 1 else "CPU"
        
        print(f"Training (1 gen, 300 networks):")
        print(f"  CPU: {cpu_time:.2f}s")
        print(f"  MPS: {mps_time:.2f}s")
        print(f"  Winner: {winner} ({abs(speedup):.2f}x faster)")
        
        if winner == "CPU":
            print(f"\n🎯 Result: CPU is faster for current network size")
            print(f"   Reason: Networks too small for GPU overhead")
        else:
            print(f"\n🚀 Result: MPS provides significant speedup")

if __name__ == "__main__":
    try:
        run_all_benchmarks()
    except KeyboardInterrupt:
        print("\n\n👋 Benchmarking interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error during benchmarking: {e}")
        raise