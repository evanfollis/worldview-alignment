"""Unit tests for gradient accuracy and orthogonality."""

import pytest
import numpy as np
from ..core.distortions import DistortionTransform
from ..core.irrationality_gradient import IrrationalityGradientComputer
from ..core.state import BaseUtility


class TestGradientAccuracy:
    """Test gradient computations for accuracy and consistency."""
    
    def test_rotation_orthogonality(self):
        """Test that rotation matrices are orthogonal."""
        distortion = DistortionTransform(state_dim=2, worldview_dim=3, gamma=0.5)
        
        # Test multiple worldview positions
        test_thetas = [
            np.array([0.0, 0.0, 0.0]),
            np.array([1.0, 0.0, 0.0]),
            np.array([0.5, 0.8, -0.3]),
            np.array([2.0, -1.5, 1.0])
        ]
        
        for theta in test_thetas:
            T = distortion(theta)
            
            # Check orthogonality: T @ T.T = I
            product = T @ T.T
            identity = np.eye(2)
            np.testing.assert_allclose(product, identity, atol=1e-12)
            
            # Check determinant = 1 (proper rotation, not reflection)
            det = np.linalg.det(T)
            assert abs(det - 1.0) < 1e-12
    
    def test_rotation_gradient_finite_diff_2d(self):
        """Test 2D rotation gradient against finite differences."""
        grad_computer = IrrationalityGradientComputer(
            state_dim=2, worldview_dim=3, gamma=0.5
        )
        
        theta = np.array([0.8, -0.6, 0.4])
        gradient = np.array([1.0, 0.5])
        
        # Analytic gradient
        jac_analytic = grad_computer.rotation_gradient_2d(theta, gradient)
        
        # Finite difference gradient
        eps = 1e-7
        jac_fd = np.zeros((2, 3))
        
        from ..core.distortions import DistortionTransform
        distortion = DistortionTransform(state_dim=2, worldview_dim=3, gamma=0.5)
        
        # Base rotation
        T_base = distortion(theta)
        rotated_base = T_base @ gradient
        
        for k in range(3):
            theta_plus = theta.copy()
            theta_plus[k] += eps
            
            T_plus = distortion(theta_plus)
            rotated_plus = T_plus @ gradient
            
            jac_fd[:, k] = (rotated_plus - rotated_base) / eps
        
        # Should match within numerical precision
        np.testing.assert_allclose(jac_analytic, jac_fd, rtol=1e-5, atol=1e-8)
    
    def test_rotation_gradient_higher_dims(self):
        """Test n-dimensional rotation gradient doesn't crash."""
        grad_computer = IrrationalityGradientComputer(
            state_dim=4, worldview_dim=3, gamma=0.5
        )
        
        theta = np.array([0.5, -0.3, 0.8])
        gradient = np.array([1.0, 0.5, -0.2, 0.7])
        
        # Should not crash and return correct shape
        jac = grad_computer.rotation_gradient_nd(theta, gradient)
        assert jac.shape == (4, 3)
        assert not np.any(np.isnan(jac))
        assert not np.any(np.isinf(jac))
    
    def test_irrationality_gradient_consistency(self):
        """Test irrationality gradient is consistent."""
        from ..core.distortions import DistortionTransform
        
        grad_computer = IrrationalityGradientComputer(
            state_dim=2, worldview_dim=3, gamma=0.5
        )
        distortion = DistortionTransform(state_dim=2, worldview_dim=3, gamma=0.5)
        
        theta_i = np.array([0.3, 0.4, -0.2])
        theta_j = np.array([-0.1, 0.6, 0.5])
        true_gradient = np.array([1.0, 0.8])
        
        # Compute gradient (should not crash)
        grad = grad_computer.compute_irrationality_gradient(
            theta_i, theta_j, true_gradient, distortion
        )
        
        assert grad.shape == (3,)
        assert not np.any(np.isnan(grad))
        assert not np.any(np.isinf(grad))
        
        # Gradient should be non-zero for different worldviews
        if not np.allclose(theta_i, theta_j):
            assert np.linalg.norm(grad) > 1e-10
    
    def test_gradient_zero_at_identical_worldviews(self):
        """Test gradient is zero when worldviews are identical."""
        from ..core.distortions import DistortionTransform
        
        grad_computer = IrrationalityGradientComputer(
            state_dim=2, worldview_dim=3, gamma=0.5
        )
        distortion = DistortionTransform(state_dim=2, worldview_dim=3, gamma=0.5)
        
        theta = np.array([0.3, 0.4, -0.2])
        true_gradient = np.array([1.0, 0.8])
        
        # Gradient w.r.t. self should be zero (no irrationality)
        grad = grad_computer.compute_irrationality_gradient(
            theta, theta, true_gradient, distortion
        )
        
        # Should be very close to zero
        assert np.linalg.norm(grad) < 1e-6
    
    def test_utility_gradient_accuracy(self):
        """Test utility gradient against finite differences."""
        utility = BaseUtility(dim=3, center=np.array([1.0, -0.5, 0.8]))
        
        test_points = [
            np.array([0.0, 0.0, 0.0]),
            np.array([1.0, -0.5, 0.8]),  # At center
            np.array([2.0, 1.0, -1.0])
        ]
        
        for point in test_points:
            # Analytic gradient
            grad_analytic = utility.gradient(point)
            
            # Finite difference
            grad_fd = utility.finite_diff_gradient(point, eps=1e-8)
            
            np.testing.assert_allclose(grad_analytic, grad_fd, rtol=1e-6)
    
    def test_distortion_jacobian_chain_rule(self):
        """Test that distortion Jacobians satisfy chain rule."""
        distortion = DistortionTransform(state_dim=2, worldview_dim=3, gamma=0.3)
        
        theta = np.array([0.5, -0.2, 0.7])
        gradient = np.array([1.0, 0.5])
        
        # Test chain rule: d/dθ[T(θ)g] should match Jacobian computation
        eps = 1e-7
        
        # Base transformation
        T_base = distortion(theta)
        result_base = T_base @ gradient
        
        # Manual finite difference
        jac_manual = np.zeros((2, 3))
        for k in range(3):
            theta_plus = theta.copy()
            theta_plus[k] += eps
            T_plus = distortion(theta_plus)
            result_plus = T_plus @ gradient
            jac_manual[:, k] = (result_plus - result_base) / eps
        
        # Jacobian method
        jac_method = distortion.jacobian(theta, gradient)
        
        np.testing.assert_allclose(jac_method, jac_manual, rtol=1e-4)