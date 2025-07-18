"""Configuration dataclasses for cleaner API."""

from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class UtilityConfig:
    """Configuration for base utility function."""
    dim: int
    utility_type: str = "quadratic"
    center: Optional[np.ndarray] = None
    scale: float = 1.0


@dataclass
class DistortionConfig:
    """Configuration for worldview distortion."""
    state_dim: int
    worldview_dim: int
    gamma: float = 0.5
    distortion_type: str = "rotation"


@dataclass
class KernelConfig:
    """Configuration for social kernel."""
    sigma_align: float = 1.0
    sigma_density: float = 2.0
    kernel_type: str = "gaussian"


@dataclass
class MaskConfig:
    """Configuration for issue masks."""
    worldview_dim: int
    tau_value: float = 0.5
    tau_participation: float = 0.3
    noise_scale: float = 0.1


@dataclass
class AgentConfig:
    """Configuration for agent dynamics."""
    alpha: float = 0.01
    lambda_align: float = 1.0
    eta_momentum: float = 0.1
    kappa_noise: float = 1.0
    sigma_base: float = 0.01
    gamma: float = 0.5
    use_analytic_gradient: bool = True


@dataclass
class SimulationConfig:
    """Complete simulation configuration."""
    n_agents: int
    state_dim: int
    worldview_dim: int
    seed: Optional[int] = None
    
    # Component configurations
    utility: Optional[UtilityConfig] = None
    distortion: Optional[DistortionConfig] = None
    kernel: Optional[KernelConfig] = None
    mask: Optional[MaskConfig] = None
    agent: Optional[AgentConfig] = None
    
    def __post_init__(self):
        """Set default configurations if not provided."""
        if self.utility is None:
            self.utility = UtilityConfig(dim=self.state_dim)
        
        if self.distortion is None:
            self.distortion = DistortionConfig(
                state_dim=self.state_dim,
                worldview_dim=self.worldview_dim
            )
        
        if self.kernel is None:
            self.kernel = KernelConfig()
        
        if self.mask is None:
            self.mask = MaskConfig(worldview_dim=self.worldview_dim)
        
        if self.agent is None:
            self.agent = AgentConfig()


@dataclass
class ExperimentConfig:
    """Configuration for experiments."""
    n_steps: int = 1000
    event_prob: float = 0.1
    event_intensity: float = 1.0
    verbose: bool = True
    save_path: Optional[str] = None
    metrics_every: int = 1  # Compute metrics every N steps


@dataclass
class PhaseExperimentConfig:
    """Configuration for phase diagram experiments."""
    lambda_range: tuple = (0.5, 4.0)
    lambda_steps: int = 8
    kappa_range: tuple = (0.0, 3.0)
    kappa_steps: int = 8
    n_agents: int = 100
    n_steps: int = 2000
    n_replicates: int = 3
    seed_base: int = 42