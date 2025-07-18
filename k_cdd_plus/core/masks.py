"""Masks module: Issue-triggered activation and value masks."""

import numpy as np
from typing import Tuple, Optional
from numba import jit


@jit(nopython=True)
def sigmoid(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """
    Sigmoid activation function.
    
    Args:
        x: Input array
        temperature: Temperature parameter (lower = sharper transition)
        
    Returns:
        Sigmoid values between 0 and 1
    """
    return 1.0 / (1.0 + np.exp(-x / temperature))


class IssueMasks:
    """
    Manages issue-triggered masks for selective agent activation.
    """
    
    def __init__(self,
                 worldview_dim: int,
                 tau_value: float = 0.5,
                 tau_participation: float = 0.3,
                 noise_scale: float = 0.1):
        """
        Initialize issue mask generator.
        
        Args:
            worldview_dim: Dimension of worldview space (D)
            tau_value: Temperature for value mask sigmoid
            tau_participation: Temperature for participation gate sigmoid
            noise_scale: Scale of noise added to masks
        """
        self.worldview_dim = worldview_dim
        self.tau_value = tau_value
        self.tau_participation = tau_participation
        self.noise_scale = noise_scale
    
    def generate_event(self,
                      focus_dims: Optional[np.ndarray] = None,
                      intensity: float = 1.0,
                      sparse: bool = True) -> np.ndarray:
        """
        Generate event salience vector e_t.
        
        Args:
            focus_dims: Indices of dimensions to activate (None = random)
            intensity: Event intensity
            sparse: If True, activate only ~30% of dimensions
            
        Returns:
            Event vector of shape (worldview_dim,)
        """
        event = np.zeros(self.worldview_dim)
        
        if focus_dims is None:
            if sparse:
                n_active = max(1, int(0.3 * self.worldview_dim))
                focus_dims = np.random.choice(self.worldview_dim, n_active, replace=False)
            else:
                focus_dims = np.arange(self.worldview_dim)
        
        for dim in focus_dims:
            event[dim] = intensity * (0.5 + 0.5 * np.random.rand())
        
        return event
    
    def sample_sensitivity_profile(self, 
                                 mean_sensitivity: float = 0.5,
                                 heterogeneity: float = 0.2) -> np.ndarray:
        """
        Sample agent sensitivity profile a_i.
        
        Args:
            mean_sensitivity: Mean sensitivity across dimensions
            heterogeneity: Variation in sensitivity
            
        Returns:
            Sensitivity vector of shape (worldview_dim,)
        """
        profile = np.random.normal(mean_sensitivity, heterogeneity, self.worldview_dim)
        return np.clip(profile, 0.0, 1.0)
    
    def compute_value_mask(self,
                          sensitivity: np.ndarray,
                          event: np.ndarray,
                          add_noise: bool = True) -> np.ndarray:
        """
        Compute value mask M_{i,t} = σ((a_i ⊙ e_t + ξ) / τ).
        
        Args:
            sensitivity: Agent sensitivity profile a_i
            event: Event salience vector e_t
            add_noise: Whether to add noise ξ
            
        Returns:
            Value mask of shape (worldview_dim,)
        """
        activation = sensitivity * event
        
        if add_noise:
            noise = np.random.normal(0, self.noise_scale, self.worldview_dim)
            activation += noise
        
        return sigmoid(activation, self.tau_value)
    
    def compute_participation_gate(self,
                                 sensitivity: np.ndarray,
                                 event: np.ndarray,
                                 add_noise: bool = True) -> float:
        """
        Compute participation gate Z_{i,t} = σ((ā_i e_t^T + ζ) / τ_p).
        
        Args:
            sensitivity: Agent sensitivity profile a_i
            event: Event salience vector e_t
            add_noise: Whether to add noise ζ
            
        Returns:
            Participation probability between 0 and 1
        """
        mean_sensitivity = np.mean(sensitivity)
        activation = mean_sensitivity * np.sum(event)
        
        if add_noise:
            noise = np.random.normal(0, self.noise_scale)
            activation += noise
        
        return float(sigmoid(np.array([activation]), self.tau_participation)[0])
    
    def compute_masks(self,
                     sensitivity: np.ndarray,
                     event: np.ndarray,
                     add_noise: bool = True) -> Tuple[np.ndarray, float]:
        """
        Compute both value mask and participation gate.
        
        Args:
            sensitivity: Agent sensitivity profile
            event: Event salience vector
            add_noise: Whether to add noise
            
        Returns:
            Tuple of (value_mask, participation_gate)
        """
        value_mask = self.compute_value_mask(sensitivity, event, add_noise)
        participation = self.compute_participation_gate(sensitivity, event, add_noise)
        
        return value_mask, participation
    
    def mask_product(self, mask1: np.ndarray, mask2: np.ndarray) -> np.ndarray:
        """
        Compute element-wise product of two masks.
        Used for M_{i,t} ⊙ M_{j,t} in social coupling.
        
        Args:
            mask1: First mask
            mask2: Second mask
            
        Returns:
            Product mask
        """
        return mask1 * mask2
    
    def effective_dimensions(self, mask: np.ndarray, threshold: float = 0.5) -> int:
        """
        Count number of effectively active dimensions in a mask.
        
        Args:
            mask: Value mask
            threshold: Activation threshold
            
        Returns:
            Number of dimensions above threshold
        """
        return int(np.sum(mask > threshold))