"""Unit tests for masks module."""

import pytest
import numpy as np
from ..core.masks import IssueMasks, sigmoid


class TestSigmoid:
    """Test sigmoid function."""
    
    def test_sigmoid_range(self):
        """Test sigmoid output is in [0, 1]."""
        x = np.array([-10, -1, 0, 1, 10])
        y = sigmoid(x)
        assert np.all(y >= 0.0)
        assert np.all(y <= 1.0)
    
    def test_sigmoid_at_zero(self):
        """Test sigmoid(0) = 0.5."""
        assert np.isclose(sigmoid(np.array([0.0]))[0], 0.5)
    
    def test_sigmoid_temperature(self):
        """Test temperature effect on sigmoid."""
        x = np.array([1.0])
        
        # High temperature -> closer to 0.5
        y_high = sigmoid(x, temperature=10.0)
        
        # Low temperature -> closer to 1.0
        y_low = sigmoid(x, temperature=0.1)
        
        assert y_low > y_high


class TestIssueMasks:
    """Test IssueMasks class."""
    
    def test_initialization(self):
        """Test issue masks initialization."""
        masks = IssueMasks(worldview_dim=5, tau_value=0.5, tau_participation=0.3)
        assert masks.worldview_dim == 5
        assert masks.tau_value == 0.5
        assert masks.tau_participation == 0.3
    
    def test_generate_event_shape(self):
        """Test event generation produces correct shape."""
        masks = IssueMasks(worldview_dim=10)
        event = masks.generate_event()
        assert event.shape == (10,)
    
    def test_generate_event_sparsity(self):
        """Test sparse events activate ~30% of dimensions."""
        masks = IssueMasks(worldview_dim=100)
        event = masks.generate_event(sparse=True)
        
        # Count non-zero elements
        active_dims = np.sum(event > 0)
        
        # Should be around 30% (allow some variation)
        expected = 30
        assert 20 <= active_dims <= 40
    
    def test_generate_event_dense(self):
        """Test dense events activate all dimensions."""
        masks = IssueMasks(worldview_dim=10)
        event = masks.generate_event(sparse=False)
        
        # All dimensions should be active
        assert np.all(event > 0)
    
    def test_generate_event_focus_dims(self):
        """Test focused events only activate specified dimensions."""
        masks = IssueMasks(worldview_dim=10)
        focus_dims = np.array([1, 3, 7])
        event = masks.generate_event(focus_dims=focus_dims)
        
        # Only specified dims should be active
        for i in range(10):
            if i in focus_dims:
                assert event[i] > 0
            else:
                assert event[i] == 0
    
    def test_sample_sensitivity_profile(self):
        """Test sensitivity profile sampling."""
        masks = IssueMasks(worldview_dim=8)
        profile = masks.sample_sensitivity_profile(mean_sensitivity=0.6, heterogeneity=0.1)
        
        assert profile.shape == (8,)
        assert np.all(profile >= 0.0)
        assert np.all(profile <= 1.0)
        
        # Mean should be approximately correct
        assert 0.4 <= np.mean(profile) <= 0.8
    
    def test_compute_value_mask(self):
        """Test value mask computation."""
        masks = IssueMasks(worldview_dim=4, tau_value=1.0)
        
        sensitivity = np.array([0.8, 0.2, 0.9, 0.1])
        event = np.array([1.0, 0.0, 1.0, 0.0])
        
        mask = masks.compute_value_mask(sensitivity, event, add_noise=False)
        
        assert mask.shape == (4,)
        assert np.all(mask >= 0.0)
        assert np.all(mask <= 1.0)
        
        # High sensitivity + high event should give high mask value
        assert mask[0] > 0.5  # 0.8 * 1.0 = 0.8 > 0
        assert mask[2] > 0.5  # 0.9 * 1.0 = 0.9 > 0
        
        # Low activation should give low mask value
        assert mask[1] < 0.5  # 0.2 * 0.0 = 0 < 0
        assert mask[3] < 0.5  # 0.1 * 0.0 = 0 < 0
    
    def test_compute_participation_gate(self):
        """Test participation gate computation."""
        masks = IssueMasks(worldview_dim=3, tau_participation=1.0)
        
        sensitivity = np.array([0.8, 0.6, 0.7])  # Mean = 0.7
        event = np.array([1.0, 1.0, 0.0])  # Sum = 2.0
        
        gate = masks.compute_participation_gate(sensitivity, event, add_noise=False)
        
        assert 0.0 <= gate <= 1.0
        
        # High activation should give high participation
        assert gate > 0.5  # 0.7 * 2.0 = 1.4 > 0
    
    def test_compute_masks(self):
        """Test combined mask computation."""
        masks = IssueMasks(worldview_dim=3)
        
        sensitivity = np.array([0.8, 0.4, 0.6])
        event = np.array([1.0, 0.5, 0.0])
        
        value_mask, participation = masks.compute_masks(sensitivity, event, add_noise=False)
        
        assert value_mask.shape == (3,)
        assert 0.0 <= participation <= 1.0
        assert np.all(value_mask >= 0.0)
        assert np.all(value_mask <= 1.0)
    
    def test_mask_product(self):
        """Test element-wise mask product."""
        masks = IssueMasks(worldview_dim=3)
        
        mask1 = np.array([0.8, 0.2, 0.6])
        mask2 = np.array([0.5, 0.9, 0.3])
        
        product = masks.mask_product(mask1, mask2)
        expected = mask1 * mask2
        
        assert np.allclose(product, expected)
    
    def test_effective_dimensions(self):
        """Test counting of effective dimensions."""
        masks = IssueMasks(worldview_dim=5)
        
        mask = np.array([0.8, 0.3, 0.7, 0.1, 0.6])
        threshold = 0.5
        
        effective = masks.effective_dimensions(mask, threshold)
        
        # Should count 0.8, 0.7, 0.6 (3 values > 0.5)
        assert effective == 3
    
    def test_sparsity_target(self):
        """Test that default parameters achieve ~30% sparsity."""
        masks = IssueMasks(worldview_dim=100)
        
        # Generate many events and check sparsity
        sparsities = []
        for _ in range(50):
            event = masks.generate_event(sparse=True)
            sparsity = np.sum(event > 0) / len(event)
            sparsities.append(sparsity)
        
        mean_sparsity = np.mean(sparsities)
        
        # Should be around 30% ± 10%
        assert 0.2 <= mean_sparsity <= 0.4