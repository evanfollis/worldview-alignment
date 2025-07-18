"""Kernels module: Social coupling and density computation."""

import numpy as np
from typing import Optional, Callable
from scipy.spatial.distance import cdist


def gaussian_kernel(distance: float, sigma: float) -> float:
    """
    Gaussian kernel function w(r) = exp(-r²/(2σ²)).
    
    Args:
        distance: Distance between two points
        sigma: Kernel width parameter
        
    Returns:
        Kernel value between 0 and 1
    """
    return np.exp(-0.5 * (distance / sigma) ** 2)


def gaussian_kernel_vectorized(distances: np.ndarray, sigma: float) -> np.ndarray:
    """
    Vectorized Gaussian kernel function.
    
    Args:
        distances: Distance array or matrix
        sigma: Kernel width parameter
        
    Returns:
        Kernel values
    """
    return np.exp(-0.5 * (distances / sigma) ** 2)


def compute_pairwise_distances(positions: np.ndarray) -> np.ndarray:
    """
    Compute pairwise distances between all positions using vectorized operations.
    
    Args:
        positions: Array of shape (n_agents, dim)
        
    Returns:
        Distance matrix of shape (n_agents, n_agents)
    """
    return cdist(positions, positions, metric='euclidean')


class SocialKernel:
    """
    Manages social coupling through kernel functions.
    Computes alignment weights and local density.
    """
    
    def __init__(self,
                 sigma_align: float = 1.0,
                 sigma_density: float = 2.0,
                 kernel_type: str = "gaussian"):
        """
        Initialize social kernel.
        
        Args:
            sigma_align: Width parameter for alignment kernel
            sigma_density: Width parameter for density computation
            kernel_type: Type of kernel function ("gaussian" or custom)
        """
        self.sigma_align = sigma_align
        self.sigma_density = sigma_density
        self.kernel_type = kernel_type
    
    def alignment_weight(self, distance: float) -> float:
        """
        Compute alignment weight between two agents.
        
        Args:
            distance: Distance in worldview space
            
        Returns:
            Weight value between 0 and 1
        """
        if self.kernel_type == "gaussian":
            return gaussian_kernel(distance, self.sigma_align)
        else:
            raise NotImplementedError(f"Kernel type {self.kernel_type} not implemented")
    
    def compute_weights_matrix(self, theta_positions: np.ndarray) -> np.ndarray:
        """
        Compute full weights matrix for all agent pairs.
        
        Args:
            theta_positions: Worldview positions of shape (n_agents, worldview_dim)
            
        Returns:
            Symmetric weights matrix of shape (n_agents, n_agents)
        """
        distances = compute_pairwise_distances(theta_positions)
        
        if self.kernel_type == "gaussian":
            weights = gaussian_kernel_vectorized(distances, self.sigma_align)
        else:
            raise NotImplementedError(f"Kernel type {self.kernel_type} not implemented")
        
        np.fill_diagonal(weights, 0)
        return weights
    
    def local_density(self, theta: np.ndarray, all_thetas: np.ndarray) -> float:
        """
        Compute local population density at a worldview position.
        
        Args:
            theta: Query worldview position of shape (worldview_dim,)
            all_thetas: All worldview positions of shape (n_agents, worldview_dim)
            
        Returns:
            Local density value
        """
        density = 0.0
        for other_theta in all_thetas:
            if not np.array_equal(theta, other_theta):
                distance = np.linalg.norm(theta - other_theta)
                if self.kernel_type == "gaussian":
                    density += gaussian_kernel(distance, self.sigma_density)
                else:
                    raise NotImplementedError(f"Kernel type {self.kernel_type} not implemented")
        
        return density
    
    def density_field(self, all_thetas: np.ndarray) -> np.ndarray:
        """
        Compute density at each agent's position using vectorized operations.
        
        Args:
            all_thetas: All worldview positions of shape (n_agents, worldview_dim)
            
        Returns:
            Density values of shape (n_agents,)
        """
        distances = compute_pairwise_distances(all_thetas)
        
        if self.kernel_type == "gaussian":
            # Compute kernel values for all pairs
            kernel_values = gaussian_kernel_vectorized(distances, self.sigma_density)
            # Zero out diagonal (don't count self)
            np.fill_diagonal(kernel_values, 0)
            # Sum across rows to get density at each position
            densities = kernel_values.sum(axis=1)
        else:
            raise NotImplementedError(f"Kernel type {self.kernel_type} not implemented")
        
        return densities
    
    def gradient_weight(self, theta_i: np.ndarray, theta_j: np.ndarray) -> np.ndarray:
        """
        Compute gradient of alignment weight w.r.t. theta_i.
        
        Args:
            theta_i: First worldview position
            theta_j: Second worldview position
            
        Returns:
            Gradient vector of shape (worldview_dim,)
        """
        diff = theta_i - theta_j
        distance = np.linalg.norm(diff)
        
        if distance < 1e-10:
            return np.zeros_like(theta_i)
        
        if self.kernel_type == "gaussian":
            w = gaussian_kernel(distance, self.sigma_align)
            grad_factor = -w / (self.sigma_align ** 2)
            return grad_factor * diff
        else:
            raise NotImplementedError(f"Gradient for {self.kernel_type} not implemented")