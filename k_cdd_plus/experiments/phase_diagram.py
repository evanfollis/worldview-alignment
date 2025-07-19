"""Phase diagram experiments for K C D D-PLUS."""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import argparse
import os
from multiprocessing import Pool
import time

from ..core.world import Simulation
from ..core.config import SimulationConfig, AgentConfig, KernelConfig


def run_single_simulation(params: Dict) -> Dict:
    """
    Run a single simulation with given parameters.
    
    Args:
        params: Dictionary with simulation parameters
        
    Returns:
        Dictionary with results
    """
    lambda_align = params['lambda_align']
    kappa_noise = params['kappa_noise']
    n_agents = params['n_agents']
    n_steps = params['n_steps']
    seed = params['seed']
    
    # Create configuration
    config = SimulationConfig(
        n_agents=n_agents,
        state_dim=2,
        worldview_dim=3,
        seed=seed,
        use_vectorized_update=True,
        agent=AgentConfig(
            lambda_align=lambda_align,
            kappa_noise=kappa_noise,
            gamma=0.5,
            alpha=0.01,
            use_analytic_gradient=True
        )
    )
    
    sim = Simulation(config=config)
    
    # Run simulation with sparse metrics for performance
    metrics = sim.run(n_steps, verbose=False, metrics_every=20)
    
    # Extract final metrics
    if len(metrics['n_clusters']) > 0:
        final_clusters = metrics['n_clusters'][-1]
        final_ir = metrics['mean_irrationality'][-1] if metrics['mean_irrationality'] else 0.0
        final_diversity = metrics['diversity'][-1] if metrics['diversity'] else 0.0
        
        # Compute time-averaged metrics (last 25% of simulation)
        start_idx = max(0, len(metrics['n_clusters']) - len(metrics['n_clusters']) // 4)
        avg_clusters = np.mean(metrics['n_clusters'][start_idx:])
        avg_ir = np.mean(metrics['mean_irrationality'][start_idx:]) if metrics['mean_irrationality'] else 0.0
    else:
        final_clusters = avg_clusters = 1
        final_ir = avg_ir = 0.0
        final_diversity = 0.0
    
    return {
        'lambda_align': lambda_align,
        'kappa_noise': kappa_noise,
        'final_clusters': final_clusters,
        'avg_clusters': avg_clusters,
        'final_ir': final_ir,
        'avg_ir': avg_ir,
        'final_diversity': final_diversity,
        'seed': seed
    }


def run_phase_diagram_experiment(
    lambda_range: Tuple[float, float] = (0.5, 4.0),
    lambda_steps: int = 8,
    kappa_range: Tuple[float, float] = (0.0, 3.0),
    kappa_steps: int = 8,
    n_agents: int = 100,
    n_steps: int = 2000,
    n_replicates: int = 3,
    n_processes: int = 4,
    seed_base: int = 42,
    save_path: str = "outputs/phase_diagram.npz"
) -> Dict[str, np.ndarray]:
    """
    Run full phase diagram experiment (B-1 from specification).
    
    Args:
        lambda_range: Range of social alignment strength values
        lambda_steps: Number of lambda values to test
        kappa_range: Range of crowd noise scaling values
        kappa_steps: Number of kappa values to test
        n_agents: Number of agents per simulation
        n_steps: Number of steps per simulation
        n_replicates: Number of replicate runs per parameter combination
        n_processes: Number of parallel processes
        seed_base: Base random seed
        save_path: Path to save results
        
    Returns:
        Dictionary with results arrays
    """
    print("Running K C D D-PLUS Phase Diagram Experiment (B-1)")
    print(f"Parameter space: λ∈{lambda_range} (×{lambda_steps}), κ∈{kappa_range} (×{kappa_steps})")
    print(f"Each point: {n_replicates} runs × {n_agents} agents × {n_steps} steps")
    print(f"Total simulations: {lambda_steps * kappa_steps * n_replicates}")
    
    # Create parameter grid
    lambda_values = np.linspace(lambda_range[0], lambda_range[1], lambda_steps)
    kappa_values = np.linspace(kappa_range[0], kappa_range[1], kappa_steps)
    
    # Prepare simulation parameters
    sim_params = []
    for i, lambda_val in enumerate(lambda_values):
        for j, kappa_val in enumerate(kappa_values):
            for rep in range(n_replicates):
                sim_params.append({
                    'lambda_align': lambda_val,
                    'kappa_noise': kappa_val,
                    'n_agents': n_agents,
                    'n_steps': n_steps,
                    'seed': seed_base + i * 1000 + j * 100 + rep
                })
    
    print(f"Starting {len(sim_params)} simulations on {n_processes} processes...")
    start_time = time.time()
    
    # Run simulations in parallel
    with Pool(n_processes) as pool:
        results = pool.map(run_single_simulation, sim_params)
    
    elapsed = time.time() - start_time
    print(f"Completed in {elapsed:.1f}s ({elapsed/len(sim_params):.2f}s per simulation)")
    
    # Organize results into arrays
    clusters_grid = np.zeros((lambda_steps, kappa_steps, n_replicates))
    ir_grid = np.zeros((lambda_steps, kappa_steps, n_replicates))
    diversity_grid = np.zeros((lambda_steps, kappa_steps, n_replicates))
    
    for result in results:
        # Find grid indices
        i = np.argmin(np.abs(lambda_values - result['lambda_align']))
        j = np.argmin(np.abs(kappa_values - result['kappa_noise']))
        
        # Find replicate index
        rep_idx = 0
        for rep in range(n_replicates):
            if clusters_grid[i, j, rep] == 0:  # Unfilled slot
                rep_idx = rep
                break
        
        clusters_grid[i, j, rep_idx] = result['avg_clusters']
        ir_grid[i, j, rep_idx] = result['avg_ir']
        diversity_grid[i, j, rep_idx] = result['final_diversity']
    
    # Compute statistics
    clusters_mean = np.mean(clusters_grid, axis=2)
    clusters_std = np.std(clusters_grid, axis=2)
    ir_mean = np.mean(ir_grid, axis=2)
    ir_std = np.std(ir_grid, axis=2)
    diversity_mean = np.mean(diversity_grid, axis=2)
    
    results_dict = {
        'lambda_values': lambda_values,
        'kappa_values': kappa_values,
        'clusters_mean': clusters_mean,
        'clusters_std': clusters_std,
        'ir_mean': ir_mean,
        'ir_std': ir_std,
        'diversity_mean': diversity_mean,
        'raw_results': results
    }
    
    # Save results
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savez(save_path, **results_dict)
    print(f"Results saved to {save_path}")
    
    return results_dict


def plot_phase_diagram(results: Dict[str, np.ndarray], 
                      save_fig: str = "outputs/phase_diagram.png"):
    """
    Plot phase diagram results.
    
    Args:
        results: Results dictionary from run_phase_diagram_experiment
        save_fig: Path to save figure
    """
    lambda_values = results['lambda_values']
    kappa_values = results['kappa_values']
    clusters_mean = results['clusters_mean']
    ir_mean = results['ir_mean']
    diversity_mean = results['diversity_mean']
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Plot 1: Number of clusters
    im1 = axes[0].imshow(clusters_mean.T, origin='lower', aspect='auto', 
                        extent=[lambda_values[0], lambda_values[-1], 
                               kappa_values[0], kappa_values[-1]],
                        cmap='viridis')
    axes[0].set_xlabel('Social Alignment (λ)')
    axes[0].set_ylabel('Crowd Noise Scaling (κ)')
    axes[0].set_title('Average Number of Clusters')
    plt.colorbar(im1, ax=axes[0])
    
    # Add phase transition contours
    axes[0].contour(lambda_values, kappa_values, clusters_mean.T, 
                   levels=[1.5, 2.5, 3.5], colors='white', linewidths=1)
    
    # Plot 2: Mean irrationality
    im2 = axes[1].imshow(ir_mean.T, origin='lower', aspect='auto',
                        extent=[lambda_values[0], lambda_values[-1], 
                               kappa_values[0], kappa_values[-1]],
                        cmap='plasma')
    axes[1].set_xlabel('Social Alignment (λ)')
    axes[1].set_ylabel('Crowd Noise Scaling (κ)')
    axes[1].set_title('Mean Perceived Irrationality')
    plt.colorbar(im2, ax=axes[1])
    
    # Plot 3: Diversity
    im3 = axes[2].imshow(diversity_mean.T, origin='lower', aspect='auto',
                        extent=[lambda_values[0], lambda_values[-1], 
                               kappa_values[0], kappa_values[-1]],
                        cmap='RdYlBu')
    axes[2].set_xlabel('Social Alignment (λ)')
    axes[2].set_ylabel('Crowd Noise Scaling (κ)')
    axes[2].set_title('Worldview Diversity')
    plt.colorbar(im3, ax=axes[2])
    
    plt.tight_layout()
    plt.savefig(save_fig, dpi=150, bbox_inches='tight')
    print(f"Phase diagram saved to {save_fig}")
    plt.show()


def analyze_phase_transitions(results: Dict[str, np.ndarray]):
    """
    Analyze phase transitions in the parameter space.
    
    Args:
        results: Results dictionary from experiment
    """
    lambda_values = results['lambda_values']
    kappa_values = results['kappa_values']
    clusters_mean = results['clusters_mean']
    
    print("\nPhase Transition Analysis:")
    print("=" * 40)
    
    # Find monoculture regions (clusters ≈ 1)
    monoculture = clusters_mean < 1.5
    if np.any(monoculture):
        mono_lambda = lambda_values[np.any(monoculture, axis=1)]
        mono_kappa = kappa_values[np.any(monoculture, axis=0)]
        print(f"Monoculture regions: λ ∈ [{mono_lambda.min():.2f}, {mono_lambda.max():.2f}]")
        print(f"                     κ ∈ [{mono_kappa.min():.2f}, {mono_kappa.max():.2f}]")
    
    # Find pluralism regions (2-4 clusters)
    pluralism = (clusters_mean >= 1.5) & (clusters_mean <= 4.5)
    if np.any(pluralism):
        plur_lambda = lambda_values[np.any(pluralism, axis=1)]
        plur_kappa = kappa_values[np.any(pluralism, axis=0)]
        print(f"Pluralism regions:   λ ∈ [{plur_lambda.min():.2f}, {plur_lambda.max():.2f}]")
        print(f"                     κ ∈ [{plur_kappa.min():.2f}, {plur_kappa.max():.2f}]")
    
    # Find fragmentation regions (>4 clusters)
    fragmentation = clusters_mean > 4.5
    if np.any(fragmentation):
        frag_lambda = lambda_values[np.any(fragmentation, axis=1)]
        frag_kappa = kappa_values[np.any(fragmentation, axis=0)]
        print(f"Fragmentation:       λ ∈ [{frag_lambda.min():.2f}, {frag_lambda.max():.2f}]")
        print(f"                     κ ∈ [{frag_kappa.min():.2f}, {frag_kappa.max():.2f}]")
    
    # Find critical transitions
    print(f"\nOverall cluster range: {clusters_mean.min():.1f} - {clusters_mean.max():.1f}")
    print(f"Maximum IR: {results['ir_mean'].max():.3f}")
    print(f"Diversity range: {results['diversity_mean'].min():.2f} - {results['diversity_mean'].max():.2f}")


def main():
    """Main entry point for phase diagram experiments."""
    parser = argparse.ArgumentParser(description='Run K C D D-PLUS phase diagram experiments')
    parser.add_argument('--lambda_min', type=float, default=0.5, help='Minimum lambda value')
    parser.add_argument('--lambda_max', type=float, default=4.0, help='Maximum lambda value')
    parser.add_argument('--lambda_steps', type=int, default=8, help='Number of lambda steps')
    parser.add_argument('--kappa_min', type=float, default=0.0, help='Minimum kappa value')
    parser.add_argument('--kappa_max', type=float, default=3.0, help='Maximum kappa value')
    parser.add_argument('--kappa_steps', type=int, default=8, help='Number of kappa steps')
    parser.add_argument('--n_agents', type=int, default=100, help='Number of agents')
    parser.add_argument('--n_steps', type=int, default=2000, help='Number of steps')
    parser.add_argument('--n_replicates', type=int, default=3, help='Number of replicates')
    parser.add_argument('--n_processes', type=int, default=4, help='Number of parallel processes')
    parser.add_argument('--quick', action='store_true', help='Quick test run')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    if args.quick:
        args.lambda_steps = 4
        args.kappa_steps = 4
        args.n_steps = 500
        args.n_replicates = 2
    
    # Run experiment
    results = run_phase_diagram_experiment(
        lambda_range=(args.lambda_min, args.lambda_max),
        lambda_steps=args.lambda_steps,
        kappa_range=(args.kappa_min, args.kappa_max),
        kappa_steps=args.kappa_steps,
        n_agents=args.n_agents,
        n_steps=args.n_steps,
        n_replicates=args.n_replicates,
        n_processes=args.n_processes,
        seed_base=args.seed
    )
    
    # Plot and analyze
    plot_phase_diagram(results)
    analyze_phase_transitions(results)


if __name__ == '__main__':
    main()