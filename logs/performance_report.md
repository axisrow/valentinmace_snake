# Snake AI Performance Analysis Report

Generated: 2025-06-17 10:31:03

## 📊 Executive Summary

**Training Performance (1 generation, 300 networks):**
- CPU Time: 9.84s
- MPS Time: 41.14s
- **Winner: CPU** (0.24x faster)

🚨 **Key Finding**: CPU is faster than MPS for current network size

## 🔍 Detailed Analysis

### Network Size Benchmarks

#### Small Networks
- CPU: 6e-06s per operation
- MPS: 0.000607s per operation
- Winner: CPU (0.01x)

#### Medium Networks
- CPU: 1.1e-05s per operation
- MPS: 0.000714s per operation
- Winner: CPU (0.02x)

#### Large Networks
- CPU: 2.2e-05s per operation
- MPS: 0.000895s per operation
- Winner: CPU (0.03x)

### Training Profiling Details

#### CPU Training Breakdown
| Operation | Time (s) | Percentage |
|-----------|----------|------------|
| evaluation_round_1 | 7.562 | 40.6% |
| evaluation_round_2 | 0.566 | 3.0% |
| evaluation_round_3 | 0.492 | 2.6% |
| evaluation_round_4 | 0.518 | 2.8% |
| evaluation_total | 9.143 | 49.1% |
| population_creation | 0.005 | 0.0% |
| score_calculation | 0.005 | 0.0% |
| single_network_creation | 0.005 | 0.0% |
| tournament_native | 0.343 | 1.8% |

#### MPS Training Breakdown
| Operation | Time (s) | Percentage |
|-----------|----------|------------|
| cpu_copies_creation | 0.484 | 1.0% |
| evaluation_round_1 | 5.969 | 12.0% |
| evaluation_round_2 | 5.927 | 11.9% |
| evaluation_round_3 | 5.458 | 10.9% |
| evaluation_round_4 | 6.654 | 13.3% |
| evaluation_total | 24.011 | 48.2% |
| population_creation | 0.141 | 0.3% |
| score_calculation | 0.003 | 0.0% |
| single_cpu_copy | 0.483 | 1.0% |
| single_network_creation | 0.140 | 0.3% |
| tournament_cpu | 0.595 | 1.2% |

## 💡 Conclusions and Recommendations

### Why CPU is Faster

1. **Network Size Too Small**: Current Snake AI networks (21→16→3) have only ~400 parameters
2. **GPU Overhead**: MPS initialization and memory transfer costs exceed computation benefits
3. **Non-optimal Operations**: Many small operations instead of large batch operations

### Optimization Strategies

1. **Increase Network Size**: Try larger networks (e.g., 21→64→32→3)
2. **Batch Operations**: Process multiple networks simultaneously
3. **Pure MPS Mode**: Avoid CPU↔MPS transfers in tournaments
4. **Use CPU for Small Networks**: Current implementation with CPU is optimal

### Final Recommendation

Based on this analysis:
- **Use CPU** for current Snake AI network size
- Consider **larger networks** to benefit from MPS
- **Keep hybrid mode** as fallback option
