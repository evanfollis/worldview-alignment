"""State module: Base utility function and gradient computation."""

import numpy as np
from typing import Callable, Tuple, Optional
from numba import jit


@jit(nopython=True)
def quadratic_utility(s: np.ndarray, center: np.ndarray, scale: float = 1.0) -> float:
    """
    Quadratic utility function U_base(s) = -scale * ||s - center||^2.
    
    Args:
        s: State vector of shape (d,)
        center: Optimal state vector of shape (d,)
        scale: Scale factor for utility magnitude
        
    Returns:
        Utility value (scalar)
    """
    diff = s - center
    return -scale * np.sum(diff * diff)


@jit(nopython=True)
def quadratic_gradient(s: np.ndarray, center: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """
    Gradient of quadratic utility: ∇U_base(s) = -2 * scale * (s - center).
    
    Args:
        s: State vector of shape (d,)
        center: Optimal state vector of shape (d,)
        scale: Scale factor for utility magnitude
        
    Returns:
        Gradient vector of shape (d,)
    """
    return -2.0 * scale * (s - center)


class BaseUtility:
    """
    Base utility function U_base that all agents share.
    Default implementation is quadratic, but can be extended.
    """
    
    def __init__(self, 
                 dim: int,
                 utility_type: str = "quadratic",
                 center: Optional[np.ndarray] = None,
                 scale: float = 1.0):
        """
        Initialize base utility function.
        
        Args:
            dim: Dimension of state space
            utility_type: Type of utility function ("quadratic" or custom)
            center: Optimal state (defaults to origin)
            scale: Scale factor for utility
        """
        self.dim = dim
        self.utility_type = utility_type
        self.scale = scale
        
        if center is None:
            self.center = np.zeros(dim)
        else:
            assert len(center) == dim, f"Center dimension {len(center)} != state dimension {dim}"
            self.center = np.array(center, dtype=np.float64)
    
    def __call__(self, s: np.ndarray) -> float:
        """Compute utility U_base(s)."""
        if self.utility_type == "quadratic":
            return quadratic_utility(s, self.center, self.scale)
        else:
            raise NotImplementedError(f"Utility type {self.utility_type} not implemented")
    
    def gradient(self, s: np.ndarray) -> np.ndarray:
        """
        Compute gradient ∇U_base(s).
        
        Args:
            s: State vector of shape (d,)
            
        Returns:
            Gradient vector of shape (d,)
        """
        if self.utility_type == "quadratic":
            return quadratic_gradient(s, self.center, self.scale)
        else:
            raise NotImplementedError(f"Gradient for {self.utility_type} not implemented")
    
    def finite_diff_gradient(self, s: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        """
        Compute gradient using finite differences (for testing).
        
        Args:
            s: State vector
            eps: Finite difference step size
            
        Returns:
            Approximate gradient vector
        """
        grad = np.zeros_like(s)
        for i in range(len(s)):
            s_plus = s.copy()
            s_minus = s.copy()
            s_plus[i] += eps
            s_minus[i] -= eps
            grad[i] = (self(s_plus) - self(s_minus)) / (2 * eps)
        return grad


class StateSpace:
    """
    Manages the external state s_t and its dynamics.
    For v1.0, state is fixed, but this allows future extensions.
    """
    
    def __init__(self, dim: int, initial_state: Optional[np.ndarray] = None):
        """
        Initialize state space.
        
        Args:
            dim: Dimension of state space
            initial_state: Initial state vector (defaults to origin)
        """
        self.dim = dim
        if initial_state is None:
            self.state = np.zeros(dim)
        else:
            assert len(initial_state) == dim
            self.state = np.array(initial_state, dtype=np.float64)
        
        self.t = 0
    
    def get_state(self) -> np.ndarray:
        """Get current state."""
        return self.state.copy()
    
    def update(self, dt: float = 1.0):
        """
        Update state dynamics (fixed for v1.0).
        
        Args:
            dt: Time step
        """
        self.t += 1