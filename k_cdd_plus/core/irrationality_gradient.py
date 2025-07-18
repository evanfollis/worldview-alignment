"""Analytic gradient computation for irrationality minimization."""

import numpy as np
from typing import List, Tuple
from .irrationality import softmax, kl_divergence


class IrrationalityGradientComputer:
    """
    Computes analytic gradients of perceived irrationality w.r.t. worldview.
    
    The key insight is that IR_{i←j} depends on θ_i through:
    1. Agent i's perceived gradient: g_i = T(θ_i) g_true
    2. Agent i's expectation of j: g_{j|i} = T(θ_j) g_true (from i's perspective)
    
    We compute ∂IR/∂θ_i analytically using chain rule.
    """
    
    def __init__(self,
                 state_dim: int,
                 worldview_dim: int,
                 n_discrete_actions: int = 10,
                 action_temperature: float = 1.0,
                 gamma: float = 0.5):
        """
        Initialize gradient computer.
        
        Args:
            state_dim: Dimension of state space
            worldview_dim: Dimension of worldview space
            n_discrete_actions: Number of discrete actions
            action_temperature: Temperature for action softmax
            gamma: Distortion strength parameter
        """
        self.state_dim = state_dim
        self.worldview_dim = worldview_dim
        self.n_discrete_actions = n_discrete_actions
        self.action_temperature = action_temperature
        self.gamma = gamma
        
        # Pre-compute action directions
        if state_dim == 2:
            angles = np.linspace(0, 2 * np.pi, n_discrete_actions, endpoint=False)
            self.action_directions = np.column_stack([np.cos(angles), np.sin(angles)])
        else:
            rng = np.random.RandomState(42)
            dirs = rng.randn(n_discrete_actions, state_dim)
            self.action_directions = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
        
        # Pre-compute A^T A for gradient computation
        self.AtA = self.action_directions.T @ self.action_directions
    
    def rotation_gradient_2d(self, theta: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        """
        Compute ∂(T(θ)g)/∂θ for 2D rotation.
        
        For rotation by angle φ = γ||θ||:
        ∂(Rg)/∂θ = ∂(Rg)/∂φ * ∂φ/∂θ
        
        Args:
            theta: Worldview vector
            gradient: True gradient being rotated
            
        Returns:
            Jacobian matrix of shape (state_dim, worldview_dim)
        """
        theta_norm = np.linalg.norm(theta)
        if theta_norm < 1e-10:
            return np.zeros((self.state_dim, self.worldview_dim))
        
        angle = self.gamma * theta_norm
        c, s = np.cos(angle), np.sin(angle)
        
        # Derivative of rotation matrix w.r.t. angle
        dR_dangle = np.array([[-s, -c], [c, -s]])
        
        # Derivative of angle w.r.t. theta
        dangle_dtheta = self.gamma * theta / theta_norm
        
        # Chain rule: ∂(Rg)/∂θ_k = dR/dφ * g * dφ/dθ_k
        jacobian = np.zeros((2, self.worldview_dim))
        rotated_grad_deriv = dR_dangle @ gradient
        
        for k in range(self.worldview_dim):
            jacobian[:, k] = rotated_grad_deriv * dangle_dtheta[k]
        
        return jacobian
    
    def compute_irrationality_gradient(self,
                                     theta_i: np.ndarray,
                                     theta_j: np.ndarray,
                                     true_gradient: np.ndarray,
                                     distortion_transform) -> np.ndarray:
        """
        Compute ∂IR_{i←j}/∂θ_i analytically.
        
        Args:
            theta_i: Agent i's worldview
            theta_j: Agent j's worldview
            true_gradient: True utility gradient
            distortion_transform: Distortion transform object
            
        Returns:
            Gradient vector of shape (worldview_dim,)
        """
        # Compute perceived gradients
        T_i = distortion_transform(theta_i)
        T_j = distortion_transform(theta_j)
        g_i = T_i @ true_gradient
        g_j = T_j @ true_gradient
        
        # Convert to action distributions
        u_i = self.action_directions @ g_i  # Utilities for each action
        u_j = self.action_directions @ g_j
        
        pi_i = softmax(u_i, self.action_temperature)
        pi_j = softmax(u_j, self.action_temperature)
        
        # Gradient of KL w.r.t. the first distribution
        # ∂KL(p||q)/∂p_k = log(p_k/q_k) + 1
        epsilon = 1e-10
        dKL_dpi = np.log((pi_i + epsilon) / (pi_j + epsilon)) + 1
        
        # Gradient of softmax w.r.t. utilities
        # ∂π_k/∂u_m = π_k(δ_km - π_m) / τ
        tau = self.action_temperature
        dsoftmax_du = pi_i[:, np.newaxis] * (np.eye(self.n_discrete_actions) - pi_i[np.newaxis, :]) / tau
        
        # Gradient of utilities w.r.t. perceived gradient
        # u = Ag where A is action_directions matrix
        du_dg = self.action_directions  # Shape: (n_actions, state_dim)
        
        # Gradient of perceived gradient w.r.t. theta_i
        if self.state_dim == 2:
            dg_dtheta = self.rotation_gradient_2d(theta_i, true_gradient)
        else:
            # For higher dims, use finite differences as fallback
            # In production, implement proper n-D rotation derivatives
            return self._finite_diff_fallback(theta_i, theta_j, true_gradient, distortion_transform)
        
        # Chain rule: combine all gradients
        # ∂IR/∂θ = ∂IR/∂π * ∂π/∂u * ∂u/∂g * ∂g/∂θ
        grad = dKL_dpi @ dsoftmax_du @ du_dg @ dg_dtheta
        
        return grad
    
    def _finite_diff_fallback(self,
                            theta_i: np.ndarray,
                            theta_j: np.ndarray,
                            true_gradient: np.ndarray,
                            distortion_transform,
                            eps: float = 1e-6) -> np.ndarray:
        """Fallback to finite differences for n>2 dimensions."""
        from .irrationality import IrrationalityComputer
        
        ir_computer = IrrationalityComputer(self.state_dim, self.n_discrete_actions, self.action_temperature)
        grad = np.zeros(self.worldview_dim)
        
        # Base irrationality
        T_i = distortion_transform(theta_i)
        T_j = distortion_transform(theta_j)
        g_i = T_i @ true_gradient
        g_j = T_j @ true_gradient
        base_ir = ir_computer.compute_irrationality(g_i, g_j)
        
        # Finite differences
        for k in range(self.worldview_dim):
            theta_plus = theta_i.copy()
            theta_plus[k] += eps
            
            T_i_plus = distortion_transform(theta_plus)
            g_i_plus = T_i_plus @ true_gradient
            ir_plus = ir_computer.compute_irrationality(g_i_plus, g_j)
            
            grad[k] = (ir_plus - base_ir) / eps
        
        return grad
    
    def compute_total_irrationality_gradient(self,
                                           agent_idx: int,
                                           all_thetas: np.ndarray,
                                           true_gradient: np.ndarray,
                                           distortion_transform) -> np.ndarray:
        """
        Compute gradient of total perceived irrationality ∑_j IR_{i←j}.
        
        Args:
            agent_idx: Index of agent i
            all_thetas: All worldview positions (n_agents, worldview_dim)
            true_gradient: True utility gradient
            distortion_transform: Distortion transform object
            
        Returns:
            Total gradient vector
        """
        n_agents = all_thetas.shape[0]
        total_grad = np.zeros(self.worldview_dim)
        
        theta_i = all_thetas[agent_idx]
        
        for j in range(n_agents):
            if j != agent_idx:
                theta_j = all_thetas[j]
                grad_ij = self.compute_irrationality_gradient(
                    theta_i, theta_j, true_gradient, distortion_transform
                )
                total_grad += grad_ij
        
        return total_grad