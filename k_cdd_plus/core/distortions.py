"""Distortions module: Worldview-dependent transformations T(θ)."""

import numpy as np
from typing import Tuple
from numba import jit


@jit(nopython=True)
def rotation_matrix_2d(angle: float) -> np.ndarray:
    """
    Create 2D rotation matrix.
    
    Args:
        angle: Rotation angle in radians
        
    Returns:
        2x2 rotation matrix
    """
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array([[c, -s], [s, c]])


@jit(nopython=True)
def rotation_matrix_nd(angle: float, dim: int, plane: Tuple[int, int] = (0, 1)) -> np.ndarray:
    """
    Create n-dimensional rotation matrix for rotation in specified plane.
    
    Args:
        angle: Rotation angle in radians
        dim: Dimension of space
        plane: Tuple of two indices specifying rotation plane
        
    Returns:
        dim x dim rotation matrix
    """
    R = np.eye(dim)
    i, j = plane
    c = np.cos(angle)
    s = np.sin(angle)
    
    R[i, i] = c
    R[i, j] = -s
    R[j, i] = s
    R[j, j] = c
    
    return R


class DistortionTransform:
    """
    Implements worldview-dependent distortion T(θ).
    Default: rotation by angle γ||θ|| in 2D or first plane in higher dims.
    """
    
    def __init__(self, 
                 state_dim: int,
                 worldview_dim: int,
                 gamma: float = 0.5,
                 distortion_type: str = "rotation"):
        """
        Initialize distortion transform.
        
        Args:
            state_dim: Dimension of state space (d)
            worldview_dim: Dimension of worldview space (D)
            gamma: Distortion strength parameter
            distortion_type: Type of distortion ("rotation" or custom)
        """
        self.state_dim = state_dim
        self.worldview_dim = worldview_dim
        self.gamma = gamma
        self.distortion_type = distortion_type
        
        if state_dim < 2 and distortion_type == "rotation":
            raise ValueError("Rotation requires at least 2D state space")
    
    def compute_angle(self, theta: np.ndarray) -> float:
        """
        Compute rotation angle from worldview vector.
        
        Args:
            theta: Worldview vector of shape (D,)
            
        Returns:
            Rotation angle γ||θ||
        """
        return self.gamma * np.linalg.norm(theta)
    
    def __call__(self, theta: np.ndarray) -> np.ndarray:
        """
        Compute distortion matrix T(θ).
        
        Args:
            theta: Worldview vector of shape (D,)
            
        Returns:
            Distortion matrix of shape (d, d)
        """
        if self.distortion_type == "rotation":
            angle = self.compute_angle(theta)
            if self.state_dim == 2:
                return rotation_matrix_2d(angle)
            else:
                return rotation_matrix_nd(angle, self.state_dim)
        else:
            raise NotImplementedError(f"Distortion type {self.distortion_type} not implemented")
    
    def jacobian_angle(self, theta: np.ndarray) -> np.ndarray:
        """
        Compute Jacobian of rotation angle w.r.t. theta.
        ∂(γ||θ||)/∂θ = γ * θ/||θ||
        
        Args:
            theta: Worldview vector
            
        Returns:
            Jacobian vector of shape (D,)
        """
        theta_norm = np.linalg.norm(theta)
        if theta_norm < 1e-10:
            return np.zeros_like(theta)
        return self.gamma * theta / theta_norm
    
    def jacobian(self, theta: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        """
        Compute Jacobian of distorted gradient w.r.t. theta.
        Used for worldview update computation.
        
        Args:
            theta: Worldview vector of shape (D,)
            gradient: True gradient vector of shape (d,)
            
        Returns:
            Jacobian matrix of shape (d, D)
        """
        if self.distortion_type == "rotation":
            angle = self.compute_angle(theta)
            dangle_dtheta = self.jacobian_angle(theta)
            
            if self.state_dim == 2:
                c = np.cos(angle)
                s = np.sin(angle)
                
                dR_dangle = np.array([[-s, -c], [c, -s]])
                
                jacobian = np.zeros((2, self.worldview_dim))
                for i in range(self.worldview_dim):
                    jacobian[:, i] = dR_dangle @ gradient * dangle_dtheta[i]
                
                return jacobian
            else:
                jacobian = np.zeros((self.state_dim, self.worldview_dim))
                dR_dangle = self._rotation_derivative(angle, self.state_dim)
                
                for i in range(self.worldview_dim):
                    jacobian[:, i] = dR_dangle @ gradient * dangle_dtheta[i]
                
                return jacobian
        else:
            raise NotImplementedError(f"Jacobian for {self.distortion_type} not implemented")
    
    def _rotation_derivative(self, angle: float, dim: int, plane: Tuple[int, int] = (0, 1)) -> np.ndarray:
        """
        Derivative of rotation matrix w.r.t. angle.
        
        Args:
            angle: Current rotation angle
            dim: Dimension
            plane: Rotation plane
            
        Returns:
            Derivative matrix dR/dangle
        """
        dR = np.zeros((dim, dim))
        i, j = plane
        c = np.cos(angle)
        s = np.sin(angle)
        
        dR[i, i] = -s
        dR[i, j] = -c
        dR[j, i] = c
        dR[j, j] = -s
        
        return dR
    
    def is_orthogonal(self, theta: np.ndarray, tol: float = 1e-10) -> bool:
        """
        Check if T(θ) is orthogonal (preserves norms).
        
        Args:
            theta: Worldview vector
            tol: Tolerance for orthogonality check
            
        Returns:
            True if T(θ) is orthogonal
        """
        T = self(theta)
        should_be_identity = T @ T.T
        identity = np.eye(self.state_dim)
        return np.allclose(should_be_identity, identity, atol=tol)