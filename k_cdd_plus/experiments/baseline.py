"""Baseline experiments for K C D D-PLUS."""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional
import argparse
import os

from ..core.world import Simulation
from ..core.masks import IssueMasks


def run_baseline_simulation(
    n_agents: int = 50,
    state_dim: int = 2,
    worldview_dim: int = 3,
    n_steps: int = 1000,
    gamma: float = 0.5,
    lambda_align: float = 1.0,
    kappa_noise: float = 1.0,
    alpha: float = 0.01,
    event_prob: float = 0.1,
    event_intensity: float = 1.0,
    seed: Optional[int] = None,
    save_path: Optional[str] = None
) -> Dict[str, list]:
    """
    Run baseline simulation with specified parameters.
    
    Args:
        n_agents: Number of agents
        state_dim: State space dimension
        worldview_dim: Worldview space dimension
        n_steps: Number of simulation steps
        gamma: Distortion strength
        lambda_align: Social alignment strength
        kappa_noise: Crowd noise scaling
        alpha: Learning rate
        event_prob: Probability of event each step
        event_intensity: Event strength
        seed: Random seed
        save_path: Path to save results
        
    Returns:
        Metrics history
    """
    print(f"Running baseline simulation with {n_agents} agents for {n_steps} steps...")
    print(f"Parameters: γ={gamma}, λ={lambda_align}, κ={kappa_noise}")
    
    agent_params = {
        'alpha': alpha,
        'lambda_align': lambda_align,
        'kappa_noise': kappa_noise,
        'eta_momentum': 0.1,
        'sigma_base': 0.01
    }
    
    distortion_params = {
        'gamma': gamma
    }
    
    sim = Simulation(
        n_agents=n_agents,
        state_dim=state_dim,
        worldview_dim=worldview_dim,
        agent_params=agent_params,
        distortion_params=distortion_params,
        seed=seed
    )
    
    event_schedule = {}
    mask_gen = IssueMasks(worldview_dim)
    
    for t in range(n_steps):
        if np.random.rand() < event_prob:
            event = mask_gen.generate_event(intensity=event_intensity)
            event_schedule[t] = event
    
    metrics = sim.run(n_steps, event_schedule, verbose=True)
    
    if save_path:
        sim.save_state(save_path)
        print(f"Saved simulation state to {save_path}")
    
    return metrics


def plot_baseline_results(metrics: Dict[str, list], save_fig: Optional[str] = None):
    """
    Plot baseline simulation results.
    
    Args:
        metrics: Metrics history from simulation
        save_fig: Path to save figure
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    axes[0].plot(metrics['time'], metrics['n_clusters'])
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('Number of Clusters')
    axes[0].set_title('Worldview Clustering')
    axes[0].grid(True)
    
    axes[1].plot(metrics['time'], metrics['mean_irrationality'])
    axes[1].set_xlabel('Time')
    axes[1].set_ylabel('Mean Perceived Irrationality')
    axes[1].set_title('Cross-Agent Irrationality')
    axes[1].grid(True)
    
    axes[2].plot(metrics['time'], metrics['diversity'])
    axes[2].set_xlabel('Time')
    axes[2].set_ylabel('Worldview Diversity')
    axes[2].set_title('Population Diversity')
    axes[2].grid(True)
    
    axes[3].plot(metrics['time'], metrics['participation_rate'])
    axes[3].set_xlabel('Time')
    axes[3].set_ylabel('Participation Rate')
    axes[3].set_title('Agent Activation')
    axes[3].grid(True)
    
    axes[4].plot(metrics['time'], metrics['gradient_alignment'])
    axes[4].set_xlabel('Time')
    axes[4].set_ylabel('Gradient Alignment')
    axes[4].set_title('Perceived Gradient Similarity')
    axes[4].grid(True)
    
    running_avg_ir = np.convolve(metrics['mean_irrationality'], 
                                  np.ones(50)/50, mode='valid')
    axes[5].plot(range(len(running_avg_ir)), running_avg_ir)
    axes[5].set_xlabel('Time')
    axes[5].set_ylabel('Smoothed IR (50-step average)')
    axes[5].set_title('Irrationality Trend')
    axes[5].grid(True)
    
    plt.tight_layout()
    
    if save_fig:
        plt.savefig(save_fig, dpi=150)
        print(f"Saved figure to {save_fig}")
    
    plt.show()


def main():
    """Main entry point for baseline experiments."""
    parser = argparse.ArgumentParser(description='Run K C D D-PLUS baseline experiments')
    parser.add_argument('--n_agents', type=int, default=50, help='Number of agents')
    parser.add_argument('--n_steps', type=int, default=1000, help='Number of steps')
    parser.add_argument('--gamma', type=float, default=0.5, help='Distortion strength')
    parser.add_argument('--lambda_align', type=float, default=1.0, help='Social alignment')
    parser.add_argument('--kappa', type=float, default=1.0, help='Crowd noise scaling')
    parser.add_argument('--quick', action='store_true', help='Quick test run (1k steps)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--save', type=str, help='Save results to file')
    
    args = parser.parse_args()
    
    if args.quick:
        args.n_steps = 1000
        args.n_agents = 20
    
    os.makedirs('outputs', exist_ok=True)
    
    metrics = run_baseline_simulation(
        n_agents=args.n_agents,
        n_steps=args.n_steps,
        gamma=args.gamma,
        lambda_align=args.lambda_align,
        kappa_noise=args.kappa,
        seed=args.seed,
        save_path=args.save
    )
    
    plot_baseline_results(
        metrics,
        save_fig=f'outputs/baseline_g{args.gamma}_l{args.lambda_align}_k{args.kappa}.png'
    )


if __name__ == '__main__':
    main()