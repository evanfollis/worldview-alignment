"""Unit tests for kernels module."""

import pytest
import numpy as np
from ..core.kernels import SocialKernel, gaussian_kernel, gaussian_kernel_vectorized, compute_pairwise_distances


class TestGaussianKernel:
    """Test Gaussian kernel functions."""
    
    def test_gaussian_kernel_at_zero(self):
        """Test Gaussian kernel value at distance 0."""
        assert np.isclose(gaussian_kernel(0.0, 1.0), 1.0)
    
    def test_gaussian_kernel_decreasing(self):
        """Test Gaussian kernel decreases with distance."""
        sigma = 1.0
        d1 = gaussian_kernel(0.5, sigma)
        d2 = gaussian_kernel(1.0, sigma)
        d3 = gaussian_kernel(2.0, sigma)
        
        assert d1 > d2 > d3
    
    def test_gaussian_kernel_vectorized(self):
        """Test vectorized kernel matches scalar version."""
        distances = np.array([0.0, 0.5, 1.0, 2.0])
        sigma = 1.5
        
        vectorized = gaussian_kernel_vectorized(distances, sigma)
        scalar = np.array([gaussian_kernel(d, sigma) for d in distances])
        
        assert np.allclose(vectorized, scalar)


class TestPairwiseDistances:
    """Test pairwise distance computation."""
    
    def test_distance_matrix_shape(self):
        """Test distance matrix has correct shape."""
        positions = np.random.randn(5, 3)
        distances = compute_pairwise_distances(positions)
        assert distances.shape == (5, 5)
    
    def test_distance_matrix_symmetric(self):
        """Test distance matrix is symmetric."""
        positions = np.random.randn(4, 2)
        distances = compute_pairwise_distances(positions)
        assert np.allclose(distances, distances.T)
    
    def test_distance_matrix_diagonal_zero(self):
        """Test diagonal of distance matrix is zero."""
        positions = np.random.randn(3, 2)
        distances = compute_pairwise_distances(positions)
        assert np.allclose(np.diag(distances), 0.0)
    
    def test_known_distances(self):
        """Test distances for known positions."""
        positions = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0]
        ])
        distances = compute_pairwise_distances(positions)
        
        # Distance from (0,0) to (1,0) should be 1
        assert np.isclose(distances[0, 1], 1.0)
        # Distance from (0,0) to (0,1) should be 1
        assert np.isclose(distances[0, 2], 1.0)
        # Distance from (1,0) to (0,1) should be sqrt(2)
        assert np.isclose(distances[1, 2], np.sqrt(2))


class TestSocialKernel:
    """Test SocialKernel class."""
    
    def test_initialization(self):
        """Test social kernel initialization."""
        kernel = SocialKernel(sigma_align=1.5, sigma_density=2.0)
        assert kernel.sigma_align == 1.5
        assert kernel.sigma_density == 2.0
        assert kernel.kernel_type == "gaussian"
    
    def test_alignment_weight(self):
        """Test alignment weight computation."""
        kernel = SocialKernel(sigma_align=1.0)
        
        # Weight at zero distance should be 1
        assert np.isclose(kernel.alignment_weight(0.0), 1.0)
        
        # Weight should decrease with distance
        w1 = kernel.alignment_weight(0.5)
        w2 = kernel.alignment_weight(1.0)
        assert w1 > w2
    
    def test_weights_matrix(self):
        """Test full weights matrix computation."""
        kernel = SocialKernel(sigma_align=1.0)
        positions = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0]
        ])
        
        weights = kernel.compute_weights_matrix(positions)
        
        # Should be symmetric
        assert np.allclose(weights, weights.T)
        
        # Diagonal should be zero
        assert np.allclose(np.diag(weights), 0.0)
        
        # All weights should be between 0 and 1
        assert np.all(weights >= 0.0)
        assert np.all(weights <= 1.0)
    
    def test_local_density(self):
        """Test local density computation."""
        kernel = SocialKernel(sigma_density=1.0)
        
        # Query point
        theta = np.array([0.0, 0.0])
        
        # Other positions
        all_thetas = np.array([
            [0.0, 0.0],  # Same position (should be ignored)
            [0.5, 0.0],  # Close
            [2.0, 0.0],  # Far
        ])
        
        density = kernel.local_density(theta, all_thetas)
        
        # Should be positive
        assert density > 0
        
        # Adding more nearby points should increase density
        all_thetas_dense = np.array([
            [0.0, 0.0],
            [0.1, 0.0],
            [0.0, 0.1],
            [0.1, 0.1],
            [2.0, 0.0],
        ])
        
        density_dense = kernel.local_density(theta, all_thetas_dense)
        assert density_dense > density
    
    def test_density_field(self):
        """Test vectorized density field computation."""
        kernel = SocialKernel(sigma_density=1.0)
        positions = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0]
        ])
        
        densities = kernel.density_field(positions)
        
        # Should have one density per position
        assert len(densities) == 4
        
        # All densities should be positive
        assert np.all(densities > 0)
        
        # Verify against individual computation
        for i in range(4):
            expected = kernel.local_density(positions[i], positions)
            assert np.isclose(densities[i], expected)
    
    def test_gradient_weight(self):
        """Test gradient of weight function."""
        kernel = SocialKernel(sigma_align=1.0)
        
        theta_i = np.array([0.0, 0.0])
        theta_j = np.array([1.0, 0.0])
        
        grad = kernel.gradient_weight(theta_i, theta_j)
        
        # Should have same dimension as theta
        assert len(grad) == len(theta_i)
        
        # Finite difference check
        eps = 1e-6
        grad_fd = np.zeros_like(theta_i)
        
        for k in range(len(theta_i)):
            theta_plus = theta_i.copy()
            theta_plus[k] += eps
            
            w_plus = kernel.alignment_weight(np.linalg.norm(theta_plus - theta_j))
            w_base = kernel.alignment_weight(np.linalg.norm(theta_i - theta_j))
            
            grad_fd[k] = (w_plus - w_base) / eps
        
        assert np.allclose(grad, grad_fd, rtol=1e-4)