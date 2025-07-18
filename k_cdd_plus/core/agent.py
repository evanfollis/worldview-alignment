"""Agent module: Individual agents with worldview dynamics."""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from .state import BaseUtility
from .distortions import DistortionTransform
from .kernels import SocialKernel
from .masks import IssueMasks
from .irrationality import IrrationalityComputer
from .irrationality_gradient import IrrationalityGradientComputer


@dataclass
class AgentState:
    """Stores agent state and history."""
    theta: np.ndarray
    theta_prev: np.ndarray
    sensitivity: np.ndarray
    value_mask: np.ndarray
    participation: float
    perceived_gradient: np.ndarray
    irrationality_to_others: Dict[int, float] = field(default_factory=dict)
    irrationality_from_others: Dict[int, float] = field(default_factory=dict)


class Agent:
    """
    Individual agent with worldview-dependent perception.
    """
    
    def __init__(self,
                 agent_id: int,
                 worldview_dim: int,
                 state_dim: int,
                 initial_theta: Optional[np.ndarray] = None,
                 sensitivity_profile: Optional[np.ndarray] = None,
                 alpha: float = 0.01,
                 lambda_align: float = 1.0,
                 eta_momentum: float = 0.1,
                 kappa_noise: float = 1.0,
                 sigma_base: float = 0.01,
                 gamma: float = 0.5,
                 use_analytic_gradient: bool = True,
                 seed: Optional[int] = None):
        """
        Initialize agent.
        
        Args:
            agent_id: Unique agent identifier
            worldview_dim: Dimension of worldview space (D)
            state_dim: Dimension of state space (d)
            initial_theta: Initial worldview position
            sensitivity_profile: Issue sensitivity vector a_i
            alpha: Learning rate
            lambda_align: Social alignment strength
            eta_momentum: Momentum coefficient
            kappa_noise: Crowd-dependent noise scaling
            sigma_base: Base noise level
        """
        self.id = agent_id
        self.worldview_dim = worldview_dim
        self.state_dim = state_dim
        
        self.alpha = alpha
        self.lambda_align = lambda_align
        self.eta_momentum = eta_momentum
        self.kappa_noise = kappa_noise
        self.sigma_base = sigma_base
        self.gamma = gamma
        self.use_analytic_gradient = use_analytic_gradient
        
        # Initialize gradient computer for irrationality
        self.ir_computer = IrrationalityComputer(state_dim)
        if use_analytic_gradient:
            self.grad_computer = IrrationalityGradientComputer(
                state_dim, worldview_dim, gamma=gamma
            )
        
        # Initialize per-agent random number generator for reproducibility
        self.rng = np.random.RandomState(seed=agent_id + 1000 if seed is None else seed + agent_id)
        
        if initial_theta is None:
            self.theta = self.rng.randn(worldview_dim) * 0.5
        else:
            self.theta = initial_theta.copy()
        
        self.theta_prev = self.theta.copy()
        
        if sensitivity_profile is None:
            self.sensitivity = self.rng.rand(worldview_dim)
        else:
            self.sensitivity = sensitivity_profile.copy()
        
        self.value_mask = np.ones(worldview_dim)
        self.participation = 1.0
        self.perceived_gradient = np.zeros(state_dim)
        
        self.history = {
            'theta': [self.theta.copy()],
            'participation': [self.participation],
            'perceived_gradient': [self.perceived_gradient.copy()]
        }
    
    def perceive_gradient(self,
                         true_gradient: np.ndarray,
                         distortion: DistortionTransform) -> np.ndarray:
        """
        Compute perceived gradient g_i = T(θ_i) g_true.
        
        Args:
            true_gradient: True utility gradient
            distortion: Distortion transform T(θ)
            
        Returns:
            Perceived gradient
        """
        T = distortion(self.theta)
        self.perceived_gradient = T @ true_gradient
        return self.perceived_gradient
    
    def compute_irrationality_to(self,
                               other_agent: 'Agent',
                               state: np.ndarray,
                               utility: BaseUtility,
                               distortion: DistortionTransform) -> float:
        """
        Compute perceived irrationality IR_{i←j,t} using KL divergence.
        
        Args:
            other_agent: Agent j being evaluated
            state: Current state
            utility: Base utility function
            distortion: Distortion transform
            
        Returns:
            Irrationality measure (KL divergence)
        """
        true_gradient = utility.gradient(state)
        
        # My perceived gradient
        my_perceived = self.perceive_gradient(true_gradient, distortion)
        
        # How I expect other agent to perceive (using their worldview)
        T_other = distortion(other_agent.theta)
        expected_other_gradient = T_other @ true_gradient
        
        # Compute KL divergence between action distributions
        return self.ir_computer.compute_irrationality(my_perceived, expected_other_gradient)
    
    def social_pull(self,
                   other_agents: List['Agent'],
                   kernel: SocialKernel,
                   masks: IssueMasks) -> np.ndarray:
        """
        Compute social alignment force.
        
        Args:
            other_agents: List of other agents
            kernel: Social kernel for computing weights
            masks: Issue masks for value dimensions
            
        Returns:
            Social pull vector in worldview space
        """
        pull = np.zeros(self.worldview_dim)
        
        for other in other_agents:
            if other.id != self.id:
                distance = np.linalg.norm(self.theta - other.theta)
                weight = kernel.alignment_weight(distance)
                
                mask_product = masks.mask_product(self.value_mask, other.value_mask)
                masked_diff = mask_product * (other.theta - self.theta)
                
                pull += weight * masked_diff
        
        return self.lambda_align * pull
    
    def irrationality_gradient(self,
                             other_agents: List['Agent'],
                             state: np.ndarray,
                             utility: BaseUtility,
                             distortion: DistortionTransform) -> np.ndarray:
        """
        Compute gradient of total perceived irrationality w.r.t. own worldview.
        Uses analytic gradient when available, falls back to finite differences.
        
        Args:
            other_agents: List of other agents
            state: Current state
            utility: Base utility function
            distortion: Distortion transform
            
        Returns:
            Gradient vector in worldview space
        """
        if self.use_analytic_gradient and hasattr(self, 'grad_computer'):
            # Use analytic gradient computation
            all_thetas = np.array([agent.theta for agent in [self] + other_agents])
            agent_idx = 0  # Self is first in the list
            
            return self.grad_computer.compute_total_irrationality_gradient(
                agent_idx, all_thetas, utility.gradient(state), distortion
            )
        else:
            # Fallback to finite differences (original implementation)
            eps = 1e-6
            grad = np.zeros(self.worldview_dim)
            
            base_ir = sum(self.compute_irrationality_to(other, state, utility, distortion)
                         for other in other_agents if other.id != self.id)
            
            for i in range(self.worldview_dim):
                theta_plus = self.theta.copy()
                theta_plus[i] += eps
                
                theta_original = self.theta.copy()
                self.theta = theta_plus
                
                ir_plus = sum(self.compute_irrationality_to(other, state, utility, distortion)
                             for other in other_agents if other.id != self.id)
                
                self.theta = theta_original
                
                grad[i] = (ir_plus - base_ir) / eps
            
            return grad
    
    def compute_noise_covariance(self, local_density: float) -> np.ndarray:
        """
        Compute noise covariance Σ = σ₀²(1 + κn_loc)I.
        
        Args:
            local_density: Local population density
            
        Returns:
            Covariance matrix
        """
        variance = self.sigma_base ** 2 * (1 + self.kappa_noise * local_density)
        return variance * np.eye(self.worldview_dim)
    
    def sample_noise(self, covariance: np.ndarray) -> np.ndarray:
        """
        Sample noise from covariance matrix using agent's RNG.
        
        Args:
            covariance: Noise covariance matrix
            
        Returns:
            Noise vector
        """
        return self.rng.multivariate_normal(np.zeros(self.worldview_dim), covariance)
    
    def update_worldview(self,
                        other_agents: List['Agent'],
                        state: np.ndarray,
                        event: np.ndarray,
                        utility: BaseUtility,
                        distortion: DistortionTransform,
                        kernel: SocialKernel,
                        masks: IssueMasks,
                        local_density: float) -> None:
        """
        Update worldview using full dynamics equation.
        
        Args:
            other_agents: List of all other agents
            state: Current external state
            event: Current event vector
            utility: Base utility function
            distortion: Distortion transform
            kernel: Social kernel
            masks: Issue mask generator
            local_density: Local population density
        """
        self.value_mask, self.participation = masks.compute_masks(self.sensitivity, event)
        
        if self.participation < 0.1:
            self.history['theta'].append(self.theta.copy())
            self.history['participation'].append(self.participation)
            self.history['perceived_gradient'].append(self.perceived_gradient.copy())
            return
        
        ir_gradient = self.irrationality_gradient(other_agents, state, utility, distortion)
        
        social_force = self.social_pull(other_agents, kernel, masks)
        
        momentum = -2 * self.eta_momentum * (self.theta - self.theta_prev)
        
        deterministic_update = (-self.alpha * ir_gradient + 
                               self.alpha * social_force + 
                               self.alpha * momentum)
        
        covariance = self.compute_noise_covariance(local_density)
        noise = self.sample_noise(covariance)
        
        theta_new = self.theta + self.participation * deterministic_update + noise
        
        self.theta_prev = self.theta.copy()
        self.theta = theta_new
        
        self.perceive_gradient(utility.gradient(state), distortion)
        
        self.history['theta'].append(self.theta.copy())
        self.history['participation'].append(self.participation)
        self.history['perceived_gradient'].append(self.perceived_gradient.copy())
    
    def get_state(self) -> AgentState:
        """Get current agent state."""
        return AgentState(
            theta=self.theta.copy(),
            theta_prev=self.theta_prev.copy(),
            sensitivity=self.sensitivity.copy(),
            value_mask=self.value_mask.copy(),
            participation=self.participation,
            perceived_gradient=self.perceived_gradient.copy()
        )