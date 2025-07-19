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
from .config import SimulationConfig
from .social_dynamics import VectorizedSocialDynamics


class Simulation:
    """
    Main simulation environment orchestrating all components.
    """
    
    def __init__(self,
                 config: Optional[SimulationConfig] = None,
                 n_agents: Optional[int] = None,
                 state_dim: Optional[int] = None,
                 worldview_dim: Optional[int] = None,
                 utility_params: Optional[Dict] = None,
                 distortion_params: Optional[Dict] = None,
                 kernel_params: Optional[Dict] = None,
                 mask_params: Optional[Dict] = None,
                 agent_params: Optional[Dict] = None,
                 seed: Optional[int] = None):
        """
        Initialize simulation.
        
        Args:
            config: Complete simulation configuration (preferred)
            n_agents: Number of agents (fallback)
            state_dim: Dimension of state space (fallback)
            worldview_dim: Dimension of worldview space (fallback)
            utility_params: Parameters for base utility (legacy)
            distortion_params: Parameters for distortion transform (legacy)
            kernel_params: Parameters for social kernel (legacy)
            mask_params: Parameters for issue masks (legacy)
            agent_params: Parameters for agent dynamics (legacy)
            seed: Random seed for reproducibility (legacy)
        """
        # Handle both new config-based and legacy parameter-based initialization
        if config is not None:
            self.config = config
            self.n_agents = config.n_agents
            self.state_dim = config.state_dim
            self.worldview_dim = config.worldview_dim
            seed = config.seed
        else:
            # Legacy initialization
            if n_agents is None or state_dim is None or worldview_dim is None:
                raise ValueError("Must provide either config or n_agents/state_dim/worldview_dim")
            
            self.n_agents = n_agents
            self.state_dim = state_dim
            self.worldview_dim = worldview_dim
            
            # Create default config
            from .config import UtilityConfig, DistortionConfig, KernelConfig, MaskConfig, AgentConfig
            self.config = SimulationConfig(
                n_agents=n_agents,
                state_dim=state_dim,
                worldview_dim=worldview_dim,
                seed=seed,
                utility=UtilityConfig(dim=state_dim, **(utility_params or {})),
                distortion=DistortionConfig(
                    state_dim=state_dim,
                    worldview_dim=worldview_dim,
                    **(distortion_params or {})
                ),
                kernel=KernelConfig(**(kernel_params or {})),
                mask=MaskConfig(worldview_dim=worldview_dim, **(mask_params or {})),
                agent=AgentConfig(**(agent_params or {}))
            )
        
        if seed is not None:
            np.random.seed(seed)
        
        self.t = 0
        self.rng = np.random.RandomState(seed)
        
        # Initialize components using config
        self.utility = BaseUtility(
            dim=self.config.utility.dim,
            utility_type=self.config.utility.utility_type,
            center=self.config.utility.center,
            scale=self.config.utility.scale
        )
        self.state_space = StateSpace(dim=self.state_dim)
        
        self.distortion = DistortionTransform(
            state_dim=self.config.distortion.state_dim,
            worldview_dim=self.config.distortion.worldview_dim,
            gamma=self.config.distortion.gamma,
            distortion_type=self.config.distortion.distortion_type
        )
        
        self.kernel = SocialKernel(
            sigma_align=self.config.kernel.sigma_align,
            sigma_density=self.config.kernel.sigma_density,
            kernel_type=self.config.kernel.kernel_type
        )
        
        self.masks = IssueMasks(
            worldview_dim=self.config.mask.worldview_dim,
            tau_value=self.config.mask.tau_value,
            tau_participation=self.config.mask.tau_participation,
            noise_scale=self.config.mask.noise_scale,
            rng=np.random.RandomState(seed + 1000 if seed is not None else None)
        )
        
        # Initialize agents
        self.agents = []
        for i in range(self.n_agents):
            agent = Agent(
                agent_id=i,
                worldview_dim=self.worldview_dim,
                state_dim=self.state_dim,
                alpha=self.config.agent.alpha,
                lambda_align=self.config.agent.lambda_align,
                eta_momentum=self.config.agent.eta_momentum,
                kappa_noise=self.config.agent.kappa_noise,
                sigma_base=self.config.agent.sigma_base,
                gamma=self.config.agent.gamma,
                use_analytic_gradient=self.config.agent.use_analytic_gradient,
                seed=seed
            )
            self.agents.append(agent)
        
        # Initialize vectorized social dynamics
        self.social_dynamics = VectorizedSocialDynamics(self.kernel, self.masks)
        self.use_vectorized_update = getattr(self.config, 'use_vectorized_update', True)
        
        self.current_event = np.zeros(self.worldview_dim)
        
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
    
    def _vectorized_worldview_update(self, state: np.ndarray, event: np.ndarray):
        """
        Vectorized worldview update for all agents simultaneously.
        Eliminates O(N²) per-agent loops for social force computation.
        """
        # Get current state
        positions = self.get_worldview_positions()  # (n_agents, worldview_dim)
        true_gradient = self.utility.gradient(state)
        
        # Compute value masks and participation for all agents
        value_masks = np.zeros((self.n_agents, self.worldview_dim))
        participations = np.zeros(self.n_agents)
        
        for i, agent in enumerate(self.agents):
            mask, participation = self.masks.compute_masks(agent.sensitivity, event)
            value_masks[i] = mask
            participations[i] = participation
            agent.value_mask = mask
            agent.participation = participation
        
        # Only update agents with significant participation
        active_agents = participations > 0.1
        n_active = np.sum(active_agents)
        
        if n_active == 0:
            return
        
        # Get positions and masks for active agents
        active_positions = positions[active_agents]
        active_masks = value_masks[active_agents]
        active_indices = np.where(active_agents)[0]
        
        # Compute social forces vectorized (only for active agents)
        if n_active > 1:
            social_forces = self.social_dynamics.compute_vectorized_social_forces(
                active_positions, active_masks, self.config.agent.lambda_align
            )
        else:
            social_forces = np.zeros_like(active_positions)
        
        # Compute irrationality gradients
        ir_gradients = np.zeros_like(active_positions)
        if self.config.agent.use_analytic_gradient and hasattr(self.agents[0], 'grad_computer'):
            for idx, agent_idx in enumerate(active_indices):
                agent = self.agents[agent_idx]
                if hasattr(agent, 'grad_computer'):
                    ir_gradients[idx] = agent.grad_computer.compute_total_irrationality_gradient(
                        agent_idx, positions, true_gradient, self.distortion
                    )
        else:
            # Use finite differences fallback
            ir_gradients = self.social_dynamics.compute_vectorized_irrationality_gradients(
                active_positions, true_gradient, self.distortion, None
            )
        
        # Compute momentum terms
        momentum_forces = np.zeros_like(active_positions)
        for idx, agent_idx in enumerate(active_indices):
            agent = self.agents[agent_idx]
            momentum_forces[idx] = -2 * self.config.agent.eta_momentum * (agent.theta - agent.theta_prev)
        
        # Compute noise
        densities = self.kernel.density_field(positions)
        active_densities = densities[active_agents]
        
        # Update worldviews
        alpha = self.config.agent.alpha
        
        for idx, agent_idx in enumerate(active_indices):
            agent = self.agents[agent_idx]
            
            # Deterministic update
            deterministic_update = (
                -alpha * ir_gradients[idx] +
                alpha * social_forces[idx] +
                alpha * momentum_forces[idx]
            )
            
            # Noise
            covariance = agent.compute_noise_covariance(active_densities[idx])
            noise = agent.sample_noise(covariance)
            
            # Update
            agent.theta_prev = agent.theta.copy()
            agent.theta = agent.theta + participations[agent_idx] * deterministic_update + noise
            
            # Update perceived gradient
            agent.perceive_gradient(true_gradient, self.distortion)
            
            # Update history
            agent.history['theta'].append(agent.theta.copy())
            agent.history['participation'].append(agent.participation)
            agent.history['perceived_gradient'].append(agent.perceived_gradient.copy())
    
    def step(self, 
             event: Optional[np.ndarray] = None, 
             compute_metrics: bool = True) -> Dict[str, float]:
        """
        Execute one simulation step.
        
        Args:
            event: Event vector (None = no event)
            compute_metrics: Whether to compute expensive metrics this step
            
        Returns:
            Dictionary of current metrics (empty if compute_metrics=False)
        """
        self.t += 1
        
        if event is None:
            event = np.zeros(self.worldview_dim)
        self.current_event = event
        
        state = self.state_space.get_state()
        
        # Use vectorized update for better performance
        if hasattr(self, 'use_vectorized_update') and self.use_vectorized_update:
            self._vectorized_worldview_update(state, event)
        else:
            # Original per-agent update (fallback)
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
        
        if not compute_metrics:
            return {}
        
        # Only compute expensive metrics when requested
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
            verbose: bool = True,
            metrics_every: int = 10) -> Dict[str, List]:
        """
        Run simulation for multiple steps.
        
        Args:
            n_steps: Number of steps to run
            event_schedule: Dictionary mapping time steps to events
            verbose: Whether to print progress
            metrics_every: Compute metrics every N steps (for performance)
            
        Returns:
            Metrics history
        """
        event_schedule = event_schedule or {}
        
        start_time = time.time()
        
        for step in range(n_steps):
            event = event_schedule.get(step, None)
            
            # Only compute metrics every metrics_every steps or at the end
            compute_metrics = (step % metrics_every == 0) or (step == n_steps - 1)
            metrics = self.step(event, compute_metrics=compute_metrics)
            
            if verbose and step % 100 == 0 and metrics:
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