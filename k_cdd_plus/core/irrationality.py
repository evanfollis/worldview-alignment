"""Irrationality module: KL divergence computation for perceived irrationality."""

import numpy as np
from typing import Tuple
from numba import jit


@jit(nopython=True)
def softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """
    Compute softmax probabilities.
    
    Args:
        x: Logits/utilities of shape (n_actions,)
        temperature: Temperature parameter (lower = more peaked)
        
    Returns:
        Probability distribution of shape (n_actions,)
    """
    # Numerical stability: subtract max
    x_scaled = x / temperature
    x_shifted = x_scaled - np.max(x_scaled)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x)


@jit(nopython=True)
def kl_divergence(p: np.ndarray, q: np.ndarray, epsilon: float = 1e-10) -> float:
    """
    Compute KL divergence KL(p || q).
    
    Args:
        p: Reference distribution
        q: Comparison distribution
        epsilon: Small value to avoid log(0)
        
    Returns:
        KL divergence value
    """
    # Add epsilon for numerical stability
    p_safe = p + epsilon
    q_safe = q + epsilon
    
    # Renormalize
    p_safe = p_safe / np.sum(p_safe)
    q_safe = q_safe / np.sum(q_safe)
    
    return np.sum(p_safe * np.log(p_safe / q_safe))


def compute_action_distribution(
    gradient: np.ndarray,
    state_dim: int,
    n_discrete_actions: int = 10,
    temperature: float = 1.0
) -> np.ndarray:
    """
    Convert gradient to action distribution via discretization.
    
    For continuous control, we discretize into directions:
    - For 2D: n_discrete_actions angles around unit circle
    - For higher dims: sample directions on unit sphere
    
    Args:
        gradient: Perceived gradient of shape (state_dim,)
        state_dim: Dimension of state space
        n_discrete_actions: Number of discrete actions
        temperature: Softmax temperature
        
    Returns:
        Action probability distribution of shape (n_discrete_actions,)
    """
    if state_dim == 2:
        # Create action directions as angles around circle
        angles = np.linspace(0, 2 * np.pi, n_discrete_actions, endpoint=False)
        action_dirs = np.column_stack([np.cos(angles), np.sin(angles)])
    else:
        # For higher dims, use random directions on unit sphere
        # In practice, could use fixed directions (e.g., axis-aligned + diagonals)
        np.random.seed(42)  # Fixed seed for reproducibility
        action_dirs = np.random.randn(n_discrete_actions, state_dim)
        action_dirs = action_dirs / np.linalg.norm(action_dirs, axis=1, keepdims=True)
    
    # Compute utility of each action as dot product with gradient
    utilities = action_dirs @ gradient
    
    # Convert to probabilities via softmax
    return softmax(utilities, temperature)


class IrrationalityComputer:
    """
    Computes perceived irrationality between agents using KL divergence.
    """
    
    def __init__(self,
                 state_dim: int,
                 n_discrete_actions: int = 10,
                 action_temperature: float = 1.0):
        """
        Initialize irrationality computer.
        
        Args:
            state_dim: Dimension of state space
            n_discrete_actions: Number of discrete actions
            action_temperature: Temperature for action softmax
        """
        self.state_dim = state_dim
        self.n_discrete_actions = n_discrete_actions
        self.action_temperature = action_temperature
        
        # Pre-compute action directions for efficiency
        if state_dim == 2:
            angles = np.linspace(0, 2 * np.pi, n_discrete_actions, endpoint=False)
            self.action_directions = np.column_stack([np.cos(angles), np.sin(angles)])
        else:
            # Use fixed random directions for reproducibility
            rng = np.random.RandomState(42)
            dirs = rng.randn(n_discrete_actions, state_dim)
            self.action_directions = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
    
    def gradient_to_action_distribution(self, gradient: np.ndarray) -> np.ndarray:
        """
        Convert gradient to action distribution.
        
        Args:
            gradient: Perceived gradient
            
        Returns:
            Action probability distribution
        """
        utilities = self.action_directions @ gradient
        return softmax(utilities, self.action_temperature)
    
    def compute_irrationality(self,
                            gradient_i: np.ndarray,
                            gradient_j_perceived_by_i: np.ndarray) -> float:
        """
        Compute IR_{i←j} = KL(π_j || π_i(·|θ_j_perceived_by_i)).
        
        Args:
            gradient_i: Agent i's own perceived gradient
            gradient_j_perceived_by_i: How agent i thinks agent j perceives gradient
            
        Returns:
            Irrationality measure
        """
        # Convert gradients to action distributions
        pi_i = self.gradient_to_action_distribution(gradient_i)
        pi_j_expected = self.gradient_to_action_distribution(gradient_j_perceived_by_i)
        
        # KL divergence: how different is j's actual behavior from i's expectation
        return kl_divergence(pi_i, pi_j_expected)
    
    def compute_irrationality_matrix(self,
                                   perceived_gradients: np.ndarray,
                                   expected_gradients: np.ndarray) -> np.ndarray:
        """
        Compute full irrationality matrix for all agent pairs.
        
        Args:
            perceived_gradients: Each agent's perceived gradient (n_agents, state_dim)
            expected_gradients: Expected gradients matrix (n_agents, n_agents, state_dim)
                               where expected_gradients[i, j] is how i expects j to perceive
            
        Returns:
            Irrationality matrix (n_agents, n_agents) where IR[i,j] = IR_{i←j}
        """
        n_agents = perceived_gradients.shape[0]
        ir_matrix = np.zeros((n_agents, n_agents))
        
        for i in range(n_agents):
            for j in range(n_agents):
                if i != j:
                    ir_matrix[i, j] = self.compute_irrationality(
                        perceived_gradients[i],
                        expected_gradients[i, j]
                    )
        
        return ir_matrix