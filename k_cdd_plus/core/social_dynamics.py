"""Vectorized social dynamics computations."""

import numpy as np
from typing import List, Tuple
from .kernels import SocialKernel
from .masks import IssueMasks


class VectorizedSocialDynamics:
    """
    Vectorized computation of social forces for all agents simultaneously.
    Eliminates O(N²) per-agent loops.
    """
    
    def __init__(self, kernel: SocialKernel, masks: IssueMasks):
        """
        Initialize vectorized social dynamics.
        
        Args:
            kernel: Social kernel for computing weights
            masks: Issue masks generator
        """
        self.kernel = kernel
        self.masks = masks
    
    def compute_pairwise_mask_matrix(self, 
                                   value_masks: np.ndarray) -> np.ndarray:
        """
        Compute pairwise mask matrix M[i,j] = mask_i ⊙ mask_j.
        
        Args:
            value_masks: Value masks of shape (n_agents, worldview_dim)
            
        Returns:
            Pairwise mask products of shape (n_agents, n_agents, worldview_dim)
        """
        n_agents, worldview_dim = value_masks.shape
        
        # Broadcast multiplication: (n_agents, 1, worldview_dim) * (1, n_agents, worldview_dim)
        mask_products = value_masks[:, np.newaxis, :] * value_masks[np.newaxis, :, :]
        
        return mask_products
    
    def compute_vectorized_social_forces(self,
                                       theta_positions: np.ndarray,
                                       value_masks: np.ndarray,
                                       lambda_align: float) -> np.ndarray:
        """
        Compute social forces for all agents using vectorized operations.
        
        Implementation of: F_i = λ ∑_j w_ij (M_i ⊙ M_j) ⊙ (θ_j - θ_i)
        
        Args:
            theta_positions: Worldview positions of shape (n_agents, worldview_dim)
            value_masks: Value masks of shape (n_agents, worldview_dim)
            lambda_align: Social alignment strength
            
        Returns:
            Social forces of shape (n_agents, worldview_dim)
        """
        n_agents, worldview_dim = theta_positions.shape
        
        # Compute pairwise weights: W[i,j] = w(||θ_i - θ_j||)
        weights = self.kernel.compute_weights_matrix(theta_positions)  # (n_agents, n_agents)
        
        # Compute pairwise mask products: M[i,j,k] = mask_i[k] * mask_j[k]
        mask_products = self.compute_pairwise_mask_matrix(value_masks)  # (n_agents, n_agents, worldview_dim)
        
        # Compute position differences: D[i,j,k] = θ_j[k] - θ_i[k]
        position_diffs = theta_positions[np.newaxis, :, :] - theta_positions[:, np.newaxis, :]  # (n_agents, n_agents, worldview_dim)
        
        # Weighted masked differences: WMD[i,j,k] = w_ij * (mask_i[k] * mask_j[k]) * (θ_j[k] - θ_i[k])
        weighted_masked_diffs = weights[:, :, np.newaxis] * mask_products * position_diffs
        
        # Sum over j to get force on each agent i: F_i[k] = ∑_j WMD[i,j,k]
        social_forces = lambda_align * np.sum(weighted_masked_diffs, axis=1)
        
        return social_forces
    
    def compute_vectorized_irrationality_gradients(self,
                                                 theta_positions: np.ndarray,
                                                 true_gradient: np.ndarray,
                                                 distortion_transform,
                                                 ir_gradient_computer) -> np.ndarray:
        """
        Compute irrationality gradients for all agents using vectorized operations.
        
        Args:
            theta_positions: Worldview positions of shape (n_agents, worldview_dim)
            true_gradient: True utility gradient
            distortion_transform: Distortion transform object
            ir_gradient_computer: Gradient computer for analytic derivatives
            
        Returns:
            Irrationality gradients of shape (n_agents, worldview_dim)
        """
        n_agents, worldview_dim = theta_positions.shape
        gradients = np.zeros((n_agents, worldview_dim))
        
        for i in range(n_agents):
            if hasattr(ir_gradient_computer, 'compute_total_irrationality_gradient'):
                gradients[i] = ir_gradient_computer.compute_total_irrationality_gradient(
                    i, theta_positions, true_gradient, distortion_transform
                )
            else:
                # Fallback to finite differences
                gradients[i] = self._finite_diff_gradient(
                    i, theta_positions, true_gradient, distortion_transform
                )
        
        return gradients
    
    def _finite_diff_gradient(self,
                            agent_idx: int,
                            theta_positions: np.ndarray,
                            true_gradient: np.ndarray,
                            distortion_transform,
                            eps: float = 1e-6) -> np.ndarray:
        """Fallback finite difference gradient computation."""
        from .irrationality import IrrationalityComputer
        
        worldview_dim = theta_positions.shape[1]
        state_dim = len(true_gradient)
        ir_computer = IrrationalityComputer(state_dim)
        
        grad = np.zeros(worldview_dim)
        theta_i = theta_positions[agent_idx]
        
        # Base irrationality
        base_ir = 0.0
        for j in range(len(theta_positions)):
            if j != agent_idx:
                T_i = distortion_transform(theta_i)
                T_j = distortion_transform(theta_positions[j])
                g_i = T_i @ true_gradient
                g_j = T_j @ true_gradient
                base_ir += ir_computer.compute_irrationality(g_i, g_j)
        
        # Finite differences
        for k in range(worldview_dim):
            theta_plus = theta_i.copy()
            theta_plus[k] += eps
            
            ir_plus = 0.0
            for j in range(len(theta_positions)):
                if j != agent_idx:
                    T_i_plus = distortion_transform(theta_plus)
                    T_j = distortion_transform(theta_positions[j])
                    g_i_plus = T_i_plus @ true_gradient
                    g_j = T_j @ true_gradient
                    ir_plus += ir_computer.compute_irrationality(g_i_plus, g_j)
            
            grad[k] = (ir_plus - base_ir) / eps
        
        return grad