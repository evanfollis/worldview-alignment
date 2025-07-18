"""Unit tests for state module."""

import pytest
import numpy as np
from ..core.state import BaseUtility, StateSpace, quadratic_utility, quadratic_gradient


class TestBaseUtility:
    """Test BaseUtility class."""
    
    def test_initialization(self):
        """Test utility initialization."""
        utility = BaseUtility(dim=3)
        assert utility.dim == 3
        assert utility.utility_type == "quadratic"
        assert np.allclose(utility.center, np.zeros(3))
    
    def test_quadratic_utility(self):
        """Test quadratic utility computation."""
        center = np.array([1.0, 2.0])
        s = np.array([0.0, 0.0])
        
        u = quadratic_utility(s, center, scale=1.0)
        expected = -5.0  # -(1^2 + 2^2)
        assert np.isclose(u, expected)
    
    def test_quadratic_gradient(self):
        """Test quadratic gradient computation."""
        center = np.array([1.0, 2.0])
        s = np.array([0.0, 0.0])
        
        grad = quadratic_gradient(s, center, scale=1.0)
        expected = np.array([2.0, 4.0])  # -2 * (s - center)
        assert np.allclose(grad, expected)
    
    def test_gradient_vs_finite_diff(self):
        """Test analytical gradient against finite differences."""
        utility = BaseUtility(dim=3, center=np.array([1.0, -1.0, 0.5]))
        s = np.array([0.5, 0.5, -0.5])
        
        analytical = utility.gradient(s)
        finite_diff = utility.finite_diff_gradient(s, eps=1e-6)
        
        assert np.allclose(analytical, finite_diff, rtol=1e-5)


class TestStateSpace:
    """Test StateSpace class."""
    
    def test_initialization(self):
        """Test state space initialization."""
        state_space = StateSpace(dim=4)
        assert state_space.dim == 4
        assert np.allclose(state_space.get_state(), np.zeros(4))
        assert state_space.t == 0
    
    def test_custom_initial_state(self):
        """Test initialization with custom state."""
        initial = np.array([1.0, 2.0, 3.0])
        state_space = StateSpace(dim=3, initial_state=initial)
        assert np.allclose(state_space.get_state(), initial)
    
    def test_update(self):
        """Test state update (fixed for v1.0)."""
        state_space = StateSpace(dim=2)
        initial_state = state_space.get_state().copy()
        
        state_space.update()
        assert state_space.t == 1
        assert np.allclose(state_space.get_state(), initial_state)