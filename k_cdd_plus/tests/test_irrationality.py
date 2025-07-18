"""Unit tests for irrationality computation."""

import pytest
import numpy as np
from ..core.irrationality import IrrationalityComputer, softmax, kl_divergence


class TestSoftmax:
    """Test softmax function."""
    
    def test_softmax_sums_to_one(self):
        """Test that softmax outputs sum to 1."""
        x = np.array([1.0, 2.0, 3.0])
        p = softmax(x)
        assert np.isclose(np.sum(p), 1.0)
    
    def test_softmax_temperature(self):
        """Test temperature effect on softmax."""
        x = np.array([1.0, 2.0])
        
        # High temperature -> more uniform
        p_high = softmax(x, temperature=10.0)
        
        # Low temperature -> more peaked
        p_low = softmax(x, temperature=0.1)
        
        # Low temp should have higher max value
        assert np.max(p_low) > np.max(p_high)


class TestKLDivergence:
    """Test KL divergence computation."""
    
    def test_kl_zero_for_identical(self):
        """Test KL divergence is zero for identical distributions."""
        p = np.array([0.5, 0.3, 0.2])
        kl = kl_divergence(p, p)
        assert np.isclose(kl, 0.0, atol=1e-10)
    
    def test_kl_positive(self):
        """Test KL divergence is positive for different distributions."""
        p = np.array([0.8, 0.1, 0.1])
        q = np.array([0.1, 0.8, 0.1])
        kl = kl_divergence(p, q)
        assert kl > 0
    
    def test_kl_asymmetric(self):
        """Test KL divergence is asymmetric."""
        p = np.array([0.8, 0.1, 0.1])
        q = np.array([0.1, 0.8, 0.1])
        
        kl_pq = kl_divergence(p, q)
        kl_qp = kl_divergence(q, p)
        
        assert not np.isclose(kl_pq, kl_qp)


class TestIrrationalityComputer:
    """Test IrrationalityComputer class."""
    
    def test_initialization(self):
        """Test irrationality computer initialization."""
        computer = IrrationalityComputer(state_dim=2, n_discrete_actions=8)
        assert computer.state_dim == 2
        assert computer.n_discrete_actions == 8
        assert computer.action_directions.shape == (8, 2)
    
    def test_action_directions_normalized(self):
        """Test that action directions are unit vectors."""
        computer = IrrationalityComputer(state_dim=2, n_discrete_actions=6)
        norms = np.linalg.norm(computer.action_directions, axis=1)
        assert np.allclose(norms, 1.0)
    
    def test_gradient_to_action_distribution(self):
        """Test conversion of gradient to action distribution."""
        computer = IrrationalityComputer(state_dim=2, n_discrete_actions=4)
        gradient = np.array([1.0, 0.0])  # Point in +x direction
        
        action_dist = computer.gradient_to_action_distribution(gradient)
        
        # Should sum to 1
        assert np.isclose(np.sum(action_dist), 1.0)
        
        # Should have highest probability for action most aligned with gradient
        best_action = np.argmax(action_dist)
        best_alignment = computer.action_directions[best_action] @ gradient
        
        # Check that this is indeed the best aligned action
        all_alignments = computer.action_directions @ gradient
        assert np.isclose(best_alignment, np.max(all_alignments))
    
    def test_irrationality_zero_for_identical(self):
        """Test irrationality is zero for identical gradients."""
        computer = IrrationalityComputer(state_dim=2)
        gradient = np.array([1.0, 1.0])
        
        ir = computer.compute_irrationality(gradient, gradient)
        assert np.isclose(ir, 0.0, atol=1e-10)
    
    def test_irrationality_positive_for_different(self):
        """Test irrationality is positive for different gradients."""
        computer = IrrationalityComputer(state_dim=2)
        gradient1 = np.array([1.0, 0.0])
        gradient2 = np.array([0.0, 1.0])
        
        ir = computer.compute_irrationality(gradient1, gradient2)
        assert ir > 0
    
    def test_irrationality_matrix(self):
        """Test computation of full irrationality matrix."""
        computer = IrrationalityComputer(state_dim=2)
        
        # Three gradients
        perceived = np.array([
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0]
        ])
        
        # Each agent expects others to have same gradient as themselves
        expected = np.tile(perceived[:, np.newaxis, :], (1, 3, 1))
        
        ir_matrix = computer.compute_irrationality_matrix(perceived, expected)
        
        # Diagonal should be zero (self-irrationality)
        assert np.allclose(np.diag(ir_matrix), 0.0, atol=1e-10)
        
        # Off-diagonal should be positive
        for i in range(3):
            for j in range(3):
                if i != j:
                    assert ir_matrix[i, j] > 0