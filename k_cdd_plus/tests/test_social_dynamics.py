"""Unit tests for vectorized social dynamics."""

import pytest
import numpy as np
from ..core.social_dynamics import VectorizedSocialDynamics
from ..core.kernels import SocialKernel
from ..core.masks import IssueMasks


class TestVectorizedSocialDynamics:
    """Test vectorized social dynamics computation."""
    
    def test_pairwise_mask_matrix(self):
        """Test pairwise mask matrix computation."""
        kernel = SocialKernel()
        masks = IssueMasks(worldview_dim=3)
        dynamics = VectorizedSocialDynamics(kernel, masks)
        
        # Test masks
        value_masks = np.array([
            [1.0, 0.5, 0.0],
            [0.8, 1.0, 0.3],
            [0.2, 0.7, 1.0]
        ])
        
        mask_products = dynamics.compute_pairwise_mask_matrix(value_masks)
        
        # Check shape
        assert mask_products.shape == (3, 3, 3)
        
        # Check diagonal (self-products)
        for i in range(3):
            np.testing.assert_array_equal(mask_products[i, i], value_masks[i] * value_masks[i])
        
        # Check symmetry in the first two dimensions
        for i in range(3):
            for j in range(3):
                np.testing.assert_array_equal(mask_products[i, j], mask_products[j, i])
        
        # Check specific values
        expected_01 = value_masks[0] * value_masks[1]  # [0.8, 0.5, 0.0]
        np.testing.assert_array_equal(mask_products[0, 1], expected_01)
    
    def test_vectorized_social_forces(self):
        """Test vectorized social forces computation."""
        kernel = SocialKernel(sigma_align=1.0)
        masks = IssueMasks(worldview_dim=2)
        dynamics = VectorizedSocialDynamics(kernel, masks)
        
        # Test positions
        theta_positions = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0]
        ])
        
        # Test masks (all ones for simplicity)
        value_masks = np.ones((3, 2))
        
        # Compute forces
        forces = dynamics.compute_vectorized_social_forces(
            theta_positions, value_masks, lambda_align=1.0
        )
        
        # Check shape
        assert forces.shape == (3, 2)
        
        # Forces should sum to zero (conservation)
        total_force = np.sum(forces, axis=0)
        np.testing.assert_allclose(total_force, np.zeros(2), atol=1e-10)
        
        # Agent at origin should be pulled toward both others
        assert forces[0, 0] > 0  # Pulled toward (1,0)
        assert forces[0, 1] > 0  # Pulled toward (0,1)
    
    def test_social_forces_vs_individual(self):
        """Test vectorized forces match individual computation."""
        kernel = SocialKernel(sigma_align=1.0)
        masks = IssueMasks(worldview_dim=2)
        dynamics = VectorizedSocialDynamics(kernel, masks)
        
        # Create simple test case
        theta_positions = np.array([
            [0.0, 0.0],
            [2.0, 0.0]
        ])
        value_masks = np.ones((2, 2))
        
        # Vectorized computation
        forces_vec = dynamics.compute_vectorized_social_forces(
            theta_positions, value_masks, lambda_align=1.0
        )
        
        # Manual computation for comparison
        distance = np.linalg.norm(theta_positions[1] - theta_positions[0])
        weight = kernel.alignment_weight(distance)
        
        # Force on agent 0: λ * w * (θ₁ - θ₀)
        expected_force_0 = 1.0 * weight * (theta_positions[1] - theta_positions[0])
        expected_force_1 = 1.0 * weight * (theta_positions[0] - theta_positions[1])
        
        np.testing.assert_allclose(forces_vec[0], expected_force_0, rtol=1e-10)
        np.testing.assert_allclose(forces_vec[1], expected_force_1, rtol=1e-10)
    
    def test_masked_social_forces(self):
        """Test that masks properly modulate social forces."""
        kernel = SocialKernel(sigma_align=1.0)
        masks = IssueMasks(worldview_dim=2)
        dynamics = VectorizedSocialDynamics(kernel, masks)
        
        theta_positions = np.array([
            [0.0, 0.0],
            [1.0, 1.0]
        ])
        
        # Full masks
        full_masks = np.ones((2, 2))
        forces_full = dynamics.compute_vectorized_social_forces(
            theta_positions, full_masks, lambda_align=1.0
        )
        
        # Partial masks (only first dimension active)
        partial_masks = np.array([
            [1.0, 0.0],
            [1.0, 0.0]
        ])
        forces_partial = dynamics.compute_vectorized_social_forces(
            theta_positions, partial_masks, lambda_align=1.0
        )
        
        # Partial forces should be smaller and only in first dimension
        assert np.linalg.norm(forces_partial) < np.linalg.norm(forces_full)
        assert forces_partial[0, 1] == 0.0  # No force in masked dimension
        assert forces_partial[1, 1] == 0.0
    
    def test_performance_comparison(self):
        """Basic performance test - vectorized should handle larger systems."""
        kernel = SocialKernel()
        masks = IssueMasks(worldview_dim=3)
        dynamics = VectorizedSocialDynamics(kernel, masks)
        
        # Larger system
        n_agents = 50
        theta_positions = np.random.randn(n_agents, 3) * 0.5
        value_masks = np.random.rand(n_agents, 3)
        
        # Should not crash and should be reasonably fast
        forces = dynamics.compute_vectorized_social_forces(
            theta_positions, value_masks, lambda_align=1.0
        )
        
        assert forces.shape == (n_agents, 3)
        assert not np.any(np.isnan(forces))
        assert not np.any(np.isinf(forces))