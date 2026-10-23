#!/usr/bin/env python3
"""
scripts/plot_results.py

Generates benchmark performance plots from results/matmul.csv for the
CPU vs GPU Parallel Computing Benchmark Suite.

Outputs:
  - results/plots/runtime_by_matrix_size.png
  - results/plots/speedup_comparison.png
  - results/plots/cpu_comparison.png
"""

import os
import sys
import csv
import matplotlib.pyplot as plt

def load_data(csv_path):
    if not os.path.isfile(csv_path):
        print(f"Error: CSV file not found at {csv_path}", file=sys.stderr)
        sys.exit(1)

    data = {
        'N': [],
        'seq_ms': [],
        'tiled_ms': [],
        'omp_ms': [],
        'tiled_omp_ms': [],
        'gpu_kernel_ms': [],
        'gpu_e2e_ms': [],
        'tiled_speedup': [],
        'omp_speedup': [],
        'tiled_omp_speedup': [],
        'gpu_kernel_speedup': [],
        'gpu_e2e_speedup': [],
        'speedup_base': [],
        'seq_correct': [],
        'tiled_correct': [],
        'omp_correct': [],
        'tiled_omp_correct': [],
        'gpu_correct': [],
    }

    with open(csv_path, mode='r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data['N'].append(int(row['N']))
            data['seq_ms'].append(float(row['seq_ms']))
            data['tiled_ms'].append(float(row['tiled_ms']))
            data['omp_ms'].append(float(row['omp_ms']))
            
            # Optional tiled_omp columns (Phase 4)
            if 'tiled_omp_ms' in row:
                data['tiled_omp_ms'].append(float(row['tiled_omp_ms']))
            
            data['gpu_kernel_ms'].append(float(row['gpu_kernel_ms']))
            data['gpu_e2e_ms'].append(float(row['gpu_e2e_ms']))
            
            # Support both column naming conventions (_vs_seq and _vs_base)
            tiled_spd_col = 'tiled_speedup_vs_base' if 'tiled_speedup_vs_base' in row else 'tiled_speedup_vs_seq'
            omp_spd_col = 'omp_speedup_vs_base' if 'omp_speedup_vs_base' in row else 'omp_speedup_vs_seq'
            tomp_spd_col = 'tiled_omp_speedup_vs_base' if 'tiled_omp_speedup_vs_base' in row else 'tiled_omp_speedup_vs_seq'
            gkern_spd_col = 'gpu_kernel_speedup_vs_base' if 'gpu_kernel_speedup_vs_base' in row else 'gpu_kernel_speedup_vs_seq'
            ge2e_spd_col = 'gpu_e2e_speedup_vs_base' if 'gpu_e2e_speedup_vs_base' in row else 'gpu_e2e_speedup_vs_seq'
            
            data['tiled_speedup'].append(float(row[tiled_spd_col]))
            data['omp_speedup'].append(float(row[omp_spd_col]))
            if tomp_spd_col in row:
                data['tiled_omp_speedup'].append(float(row[tomp_spd_col]))
                
            data['gpu_kernel_speedup'].append(float(row[gkern_spd_col]))
            data['gpu_e2e_speedup'].append(float(row[ge2e_spd_col]))
            
            data['speedup_base'].append(row.get('speedup_base', 'seq'))
            data['seq_correct'].append(row.get('seq_correct', ''))
            data['tiled_correct'].append(row.get('tiled_correct', ''))
            data['omp_correct'].append(row.get('omp_correct', ''))
            data['tiled_omp_correct'].append(row.get('tiled_omp_correct', ''))
            data['gpu_correct'].append(row.get('gpu_correct', ''))

    return data


def plot_runtime_by_matrix_size(data, out_dir):
    """
    Plot 1: Runtime by matrix size for Tiled CPU, OpenMP CPU, Parallel Tiled CPU, GPU Kernel, and GPU End-to-End.
    """
    fig, ax = plt.subplots(figsize=(10, 6.5), dpi=300)

    N_vals = data['N']

    ax.plot(N_vals, data['tiled_ms'], marker='s', linewidth=2, color='#2ca02c',
            label='Tiled CPU (single-thread, BS=64)')
    ax.plot(N_vals, data['omp_ms'], marker='^', linewidth=2, color='#ff7f0e',
            label='OpenMP CPU (naïve loop, 12 threads)')
    
    if data['tiled_omp_ms']:
        ax.plot(N_vals, data['tiled_omp_ms'], marker='P', linewidth=2.2, color='#17becf',
                label='Parallel Tiled CPU (OpenMP, 12 threads, BS=64)')

    ax.plot(N_vals, data['gpu_e2e_ms'], marker='D', linewidth=2, color='#1f77b4',
            label='GPU End-to-End (RTX 4060, incl. transfers)')
    ax.plot(N_vals, data['gpu_kernel_ms'], marker='o', linewidth=2, linestyle='--', color='#9467bd',
            label='GPU Kernel-Only (RTX 4060, compute only)')

    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_xticks(N_vals)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())

    ax.set_xlabel('Matrix Dimension N (N × N)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Median Execution Time (ms) [Log Scale]', fontsize=12, fontweight='bold')
    ax.set_title('Matrix Multiplication Runtime by Matrix Size\n'
                 'Hardware: Intel Core i5-12450H & NVIDIA GeForce RTX 4060 Laptop GPU',
                 fontsize=13, fontweight='bold', pad=12)

    ax.grid(True, which='both', linestyle=':', alpha=0.6)
    ax.legend(fontsize=10, loc='upper left', framealpha=0.9)

    plt.tight_layout()
    out_path = os.path.join(out_dir, 'runtime_by_matrix_size.png')
    plt.savefig(out_path)
    plt.close()
    print(f"Generated: {out_path}")


def plot_speedup_comparison(data, out_dir):
    """
    Plot 2: Speedup relative to the applicable documented baseline.
    """
    fig, ax = plt.subplots(figsize=(10, 6.5), dpi=300)

    N_vals = data['N']

    ax.plot(N_vals, data['tiled_speedup'], marker='s', linewidth=2, color='#2ca02c',
            label='Tiled CPU Speedup (single-thread)')
    ax.plot(N_vals, data['omp_speedup'], marker='^', linewidth=2, color='#ff7f0e',
            label='OpenMP CPU Speedup (naïve 12T)')
    
    if data['tiled_omp_speedup']:
        ax.plot(N_vals, data['tiled_omp_speedup'], marker='P', linewidth=2.2, color='#17becf',
                label='Parallel Tiled CPU Speedup (12T)')

    ax.plot(N_vals, data['gpu_e2e_speedup'], marker='D', linewidth=2, color='#1f77b4',
            label='GPU End-to-End Speedup (incl. transfers)')
    ax.plot(N_vals, data['gpu_kernel_speedup'], marker='o', linewidth=2, linestyle='--', color='#9467bd',
            label='GPU Kernel-Only Speedup (compute only)')

    # Baseline line (1.0x)
    ax.axhline(1.0, color='gray', linestyle='-.', linewidth=1.2, label='Baseline (1.0×)')

    # Vertical line indicating baseline transition at N=2048
    ax.axvline(2048, color='crimson', linestyle=':', alpha=0.7)
    ax.text(1950, 0.2, 'N=2048 Baseline:\nTiled CPU (1T)', color='crimson', fontsize=9,
            fontweight='bold', ha='right', va='bottom',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='crimson', alpha=0.8))

    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_xticks(N_vals)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())

    ax.set_xlabel('Matrix Dimension N (N × N)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Speedup Relative to Baseline [Log Scale]', fontsize=12, fontweight='bold')
    ax.set_title('Matrix Multiplication Speedup by Implementation\n'
                 'Baseline: Sequential CPU (N ≤ 1024) | Tiled CPU 1T (N = 2048)',
                 fontsize=13, fontweight='bold', pad=12)

    ax.grid(True, which='both', linestyle=':', alpha=0.6)
    ax.legend(fontsize=10, loc='upper left', framealpha=0.9)

    plt.tight_layout()
    out_path = os.path.join(out_dir, 'speedup_comparison.png')
    plt.savefig(out_path)
    plt.close()
    print(f"Generated: {out_path}")


def plot_cpu_comparison(data, out_dir):
    """
    Plot 3: CPU implementation comparison, including the naïve sequential baseline
    where it was measured (skipping N=2048 where seq was omitted).
    """
    fig, ax = plt.subplots(figsize=(10, 6.5), dpi=300)

    # Filter sequential points where it was actually run (seq_ms > 0)
    seq_N = [n for n, t, s in zip(data['N'], data['seq_ms'], data['seq_correct']) if t > 0 and s != 'skip']
    seq_t = [t for t, s in zip(data['seq_ms'], data['seq_correct']) if t > 0 and s != 'skip']

    all_N = data['N']

    ax.plot(seq_N, seq_t, marker='o', linewidth=2, color='#d62728',
            label='Sequential CPU (naïve i-j-k, 1 thread)')
    ax.plot(all_N, data['tiled_ms'], marker='s', linewidth=2, color='#2ca02c',
            label='Tiled CPU (cache-blocked i-k-j, BS=64, 1 thread)')
    ax.plot(all_N, data['omp_ms'], marker='^', linewidth=2, color='#ff7f0e',
            label='OpenMP CPU (naïve i-j-k, 12 threads)')

    if data['tiled_omp_ms']:
        ax.plot(all_N, data['tiled_omp_ms'], marker='P', linewidth=2.2, color='#17becf',
                label='Parallel Tiled CPU (OpenMP, 12 threads, BS=64)')

    # Add annotation for skipped sequential run at N=2048
    ax.scatter([2048], [data['tiled_ms'][-1]], color='#2ca02c', s=60, zorder=5)
    ax.annotate('Sequential skipped at N=2048\n(estimated >40s due to O(N³))',
                xy=(2048, data['omp_ms'][-1]), xytext=(700, 15000),
                arrowprops=dict(facecolor='black', arrowstyle='->', lw=1),
                fontsize=9, bbox=dict(boxstyle='round,pad=0.3', facecolor='#ffffcc', edgecolor='gray'))

    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_xticks(all_N)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())

    ax.set_xlabel('Matrix Dimension N (N × N)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Median Execution Time (ms) [Log Scale]', fontsize=12, fontweight='bold')
    ax.set_title('CPU Implementation Comparison: Sequential vs Tiled vs OpenMP vs Parallel Tiled\n'
                 'Hardware: Intel Core i5-12450H (8 cores / 12 threads)',
                 fontsize=13, fontweight='bold', pad=12)

    ax.grid(True, which='both', linestyle=':', alpha=0.6)
    ax.legend(fontsize=10, loc='upper left', framealpha=0.9)

    plt.tight_layout()
    out_path = os.path.join(out_dir, 'cpu_comparison.png')
    plt.savefig(out_path)
    plt.close()
    print(f"Generated: {out_path}")


def main():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    csv_path = os.path.join(project_root, 'results', 'matmul.csv')
    plots_dir = os.path.join(project_root, 'results', 'plots')

    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    if len(sys.argv) > 2:
        plots_dir = sys.argv[2]

    os.makedirs(plots_dir, exist_ok=True)
    print(f"Reading benchmark data from: {csv_path}")
    print(f"Saving plots to: {plots_dir}")

    data = load_data(csv_path)

    plot_runtime_by_matrix_size(data, plots_dir)
    plot_speedup_comparison(data, plots_dir)
    plot_cpu_comparison(data, plots_dir)

    print("All plots generated successfully.")

if __name__ == '__main__':
    main()
