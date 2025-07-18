"""World module: Simulation environment and orchestration."""

import numpy as np
from typing import List, Dict, Optional, Tuple
from sklearn.cluster import DBSCAN
import time

from .agent import Agent
from .state import BaseUtility, StateSpace
from .distortions import DistortionTransform
from .kernels import SocialKernel
from .masks import IssueMasks


class Simulation:
    """
    Main simulation environment orchestrating all components.
    """
    
    def __init__(self,
                 n_agents: int,
                 state_dim: int,
                 worldview_dim: int,
                 utility_params: Optional[Dict] = None,
                 distortion_params: Optional[Dict] = None,
                 kernel_params: Optional[Dict] = None,
                 mask_params: Optional[Dict] = None,
                 agent_params: Optional[Dict] = None,
                 seed: Optional[int] = None):
        """
        Initialize simulation.
        
        Args:
            n_agents: Number of agents
            state_dim: Dimension of state space (d)
            worldview_dim: Dimension of worldview space (D)
            utility_params: Parameters for base utility
            distortion_params: Parameters for distortion transform
            kernel_params: Parameters for social kernel
            mask_params: Parameters for issue masks
            agent_params: Parameters for agent dynamics
            seed: Random seed for reproducibility
        """
        if seed is not None:
            np.random.seed(seed)
        
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.worldview_dim = worldview_dim
        self.t = 0
        
        utility_params = utility_params or {}
        self.utility = BaseUtility(dim=state_dim, **utility_params)
        self.state_space = StateSpace(dim=state_dim)
        
        distortion_params = distortion_params or {}
        self.distortion = DistortionTransform(
            state_dim=state_dim,
            worldview_dim=worldview_dim,
            **distortion_params
        )
        
        kernel_params = kernel_params or {}
        self.kernel = SocialKernel(**kernel_params)
        
        mask_params = mask_params or {}
        self.masks = IssueMasks(worldview_dim=worldview_dim, **mask_params)
        
        agent_params = agent_params or {}
        self.agents = []
        for i in range(n_agents):
            agent = Agent(
                agent_id=i,
                worldview_dim=worldview_dim,
                state_dim=state_dim,
                **agent_params
            )
            self.agents.append(agent)
        
        self.current_event = np.zeros(worldview_dim)
        
        self.metrics_history = {
            'time': [],
            'n_clusters': [],
            'mean_irrationality': [],
            'diversity': [],
            'participation_rate': [],
            'gradient_alignment': []
        }
    
    def get_worldview_positions(self) -> np.ndarray:
        """Get all agent worldview positions."""
        return np.array([agent.theta for agent in self.agents])
    
    def compute_clusters(self, eps: float = 0.5, min_samples: int = 2) -> Tuple[int, np.ndarray]:
        """
        Compute worldview clusters using DBSCAN.
        
        Args:
            eps: Maximum distance between samples in cluster
            min_samples: Minimum samples per cluster
            
        Returns:
            Tuple of (n_clusters, cluster_labels)
        """
        positions = self.get_worldview_positions()
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(positions)
        labels = clustering.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        return n_clusters, labels
    
    def compute_mean_irrationality(self) -> float:
        """Compute mean perceived irrationality across all agent pairs."""
        state = self.state_space.get_state()
        total_ir = 0.0
        count = 0
        
        for i, agent_i in enumerate(self.agents):
            for j, agent_j in enumerate(self.agents):
                if i != j:
                    ir = agent_i.compute_irrationality_to(
                        agent_j, state, self.utility, self.distortion
                    )
                    total_ir += ir
                    count += 1
        
        return total_ir / count if count > 0 else 0.0
    
    def compute_gradient_alignment(self) -> float:
        """Compute mean cosine similarity between perceived gradients."""
        gradients = []
        state = self.state_space.get_state()
        true_gradient = self.utility.gradient(state)
        
        for agent in self.agents:
            perceived = agent.perceive_gradient(true_gradient, self.distortion)
            gradients.append(perceived)
        
        alignments = []
        for i in range(len(gradients)):
            for j in range(i + 1, len(gradients)):
                g1, g2 = gradients[i], gradients[j]
                norm1, norm2 = np.linalg.norm(g1), np.linalg.norm(g2)
                if norm1 > 1e-10 and norm2 > 1e-10:
                    cos_sim = np.dot(g1, g2) / (norm1 * norm2)
                    alignments.append(cos_sim)
        
        return np.mean(alignments) if alignments else 0.0
    
    def compute_diversity(self) -> float:
        """Compute worldview diversity (mean pairwise distance)."""
        positions = self.get_worldview_positions()
        distances = []
        
        for i in range(self.n_agents):
            for j in range(i + 1, self.n_agents):
                dist = np.linalg.norm(positions[i] - positions[j])
                distances.append(dist)
        
        return np.mean(distances) if distances else 0.0
    
    def step(self, event: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Execute one simulation step.
        
        Args:
            event: Event vector (None = no event)
            
        Returns:
            Dictionary of current metrics
        """
        self.t += 1
        
        if event is None:
            event = np.zeros(self.worldview_dim)
        self.current_event = event
        
        state = self.state_space.get_state()
        positions = self.get_worldview_positions()
        densities = self.kernel.density_field(positions)
        
        for i, agent in enumerate(self.agents):
            local_density = densities[i]
            agent.update_worldview(
                other_agents=self.agents,
                state=state,
                event=event,
                utility=self.utility,
                distortion=self.distortion,
                kernel=self.kernel,
                masks=self.masks,
                local_density=local_density
            )
        
        n_clusters, _ = self.compute_clusters()
        mean_ir = self.compute_mean_irrationality()
        diversity = self.compute_diversity()
        participation = np.mean([agent.participation for agent in self.agents])
        gradient_align = self.compute_gradient_alignment()
        
        metrics = {
            'time': self.t,
            'n_clusters': n_clusters,
            'mean_irrationality': mean_ir,
            'diversity': diversity,
            'participation_rate': participation,
            'gradient_alignment': gradient_align
        }
        
        for key, value in metrics.items():
            self.metrics_history[key].append(value)
        
        return metrics
    
    def run(self, 
            n_steps: int,
            event_schedule: Optional[Dict[int, np.ndarray]] = None,
            verbose: bool = True) -> Dict[str, List]:
        """
        Run simulation for multiple steps.
        
        Args:
            n_steps: Number of steps to run
            event_schedule: Dictionary mapping time steps to events
            verbose: Whether to print progress
            
        Returns:
            Metrics history
        """
        event_schedule = event_schedule or {}
        
        start_time = time.time()
        
        for step in range(n_steps):
            event = event_schedule.get(step, None)
            metrics = self.step(event)
            
            if verbose and step % 100 == 0:
                elapsed = time.time() - start_time
                print(f"Step {step}/{n_steps} - "
                      f"Clusters: {metrics['n_clusters']}, "
                      f"IR: {metrics['mean_irrationality']:.3f}, "
                      f"Time: {elapsed:.1f}s")
        
        return self.metrics_history
    
    def save_state(self, filename: str):
        """Save simulation state to file."""
        state_dict = {
            'n_agents': self.n_agents,
            'state_dim': self.state_dim,
            'worldview_dim': self.worldview_dim,
            't': self.t,
            'agent_thetas': self.get_worldview_positions(),
            'agent_sensitivities': np.array([agent.sensitivity for agent in self.agents]),
            'metrics_history': self.metrics_history
        }
        np.savez(filename, **state_dict)
    
    def load_state(self, filename: str):
        """Load simulation state from file."""
        data = np.load(filename)
        
        for i, agent in enumerate(self.agents):
            agent.theta = data['agent_thetas'][i]
            agent.sensitivity = data['agent_sensitivities'][i]
        
        self.t = int(data['t'])
        self.metrics_history = {k: list(v) for k, v in data['metrics_history'].items()}