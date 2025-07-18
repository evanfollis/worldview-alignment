"""Unit tests for distortions module."""

import pytest
import numpy as np
from ..core.distortions import DistortionTransform, rotation_matrix_2d


class TestDistortionTransform:
    """Test DistortionTransform class."""
    
    def test_initialization(self):
        """Test distortion initialization."""
        distortion = DistortionTransform(state_dim=2, worldview_dim=3, gamma=0.5)
        assert distortion.state_dim == 2
        assert distortion.worldview_dim == 3
        assert distortion.gamma == 0.5
    
    def test_rotation_matrix_2d(self):
        """Test 2D rotation matrix."""
        angle = np.pi / 4  # 45 degrees
        R = rotation_matrix_2d(angle)
        
        # Check orthogonality
        assert np.allclose(R @ R.T, np.eye(2))
        
        # Check determinant = 1
        assert np.isclose(np.linalg.det(R), 1.0)
    
    def test_compute_angle(self):
        """Test angle computation from worldview."""
        distortion = DistortionTransform(state_dim=2, worldview_dim=3, gamma=0.5)
        theta = np.array([1.0, 0.0, 0.0])
        
        angle = distortion.compute_angle(theta)
        expected = 0.5 * 1.0  # gamma * ||theta||
        assert np.isclose(angle, expected)
    
    def test_distortion_orthogonality(self):
        """Test that T(θ) is orthogonal."""
        distortion = DistortionTransform(state_dim=2, worldview_dim=3, gamma=0.5)
        theta = np.array([1.0, 2.0, -1.0])
        
        assert distortion.is_orthogonal(theta)
    
    def test_jacobian_angle(self):
        """Test Jacobian of angle w.r.t. theta."""
        distortion = DistortionTransform(state_dim=2, worldview_dim=3, gamma=0.5)
        theta = np.array([3.0, 4.0, 0.0])
        
        jac = distortion.jacobian_angle(theta)
        
        # Verify by finite differences
        eps = 1e-6
        jac_fd = np.zeros_like(theta)
        for i in range(len(theta)):
            theta_plus = theta.copy()
            theta_plus[i] += eps
            angle_plus = distortion.compute_angle(theta_plus)
            angle_base = distortion.compute_angle(theta)
            jac_fd[i] = (angle_plus - angle_base) / eps
        
        assert np.allclose(jac, jac_fd, rtol=1e-5)
    
    def test_zero_worldview(self):
        """Test behavior at zero worldview (no distortion)."""
        distortion = DistortionTransform(state_dim=2, worldview_dim=3)
        theta = np.zeros(3)
        
        T = distortion(theta)
        assert np.allclose(T, np.eye(2))  # Should be identity