# K C D D-PLUS: Kernel-Coupled, Directionally-Distorted, Issue-Triggered Multi-Agent Simulator

Version 1.0

## Overview

K C D D-PLUS models how rational agents with a shared utility function can diverge into distinct worldview clusters due to directional perception distortions. All agents maximize the same latent utility U_base, but each perceives its gradient through a worldview-dependent lens T(θ).

## Key Features

- **Shared Utility, Different Perceptions**: All agents share the same goal but perceive gradients differently
- **Social Coupling**: Agents are attracted to others with similar worldviews via kernel weighting
- **Issue-Triggered Dynamics**: Events selectively activate agents based on sensitivity profiles
- **Population-Dependent Noise**: Crowded regions experience higher volatility
- **Emergent Clustering**: Agents self-organize into worldview clusters without explicit coordination

## Installation

```bash
# Clone repository
git clone <repository-url>
cd worldview-alignment

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install package
pip install -e .
```

## Quick Start

```python
from k_cdd_plus.core.world import Simulation

# Create simulation with 50 agents
sim = Simulation(
    n_agents=50,
    state_dim=2,
    worldview_dim=3,
    seed=42
)

# Run for 1000 steps
metrics = sim.run(n_steps=1000, verbose=True)

# Save results
sim.save_state('outputs/simulation_results.npz')
```

### Command Line Usage

```bash
# Run baseline experiment (quick test)
python -m k_cdd_plus.experiments.baseline --quick

# Run full baseline with custom parameters
python -m k_cdd_plus.experiments.baseline \
    --n_agents 100 \
    --n_steps 5000 \
    --gamma 0.8 \
    --lambda_align 2.0 \
    --kappa 1.5
```

## Core Components

### 1. State Module (`core/state.py`)
- `BaseUtility`: Shared utility function U_base(s)
- `StateSpace`: External state management

### 2. Distortions Module (`core/distortions.py`)
- `DistortionTransform`: Implements T(θ) rotations
- Preserves gradient magnitude while rotating direction

### 3. Kernels Module (`core/kernels.py`)
- `SocialKernel`: Gaussian attraction weights
- Local density computation for noise scaling

### 4. Masks Module (`core/masks.py`)
- `IssueMasks`: Event-triggered activation
- Value masks M and participation gates Z

### 5. Agent Module (`core/agent.py`)
- `Agent`: Individual agent dynamics
- Worldview updates via gradient descent + social forces

### 6. World Module (`core/world.py`)
- `Simulation`: Main orchestration class
- Metrics tracking and clustering analysis

## Mathematical Framework

### Worldview Update Equation

```
θ_{i,t+1} = θ_{i,t} + Z_{i,t}[
    -α∇_θ(∑_j IR_{i←j,t})
    + αλ∑_j w_{ij}(M_{i,t}⊙M_{j,t})⊙(θ_{j,t}-θ_{i,t})
    - 2αη(θ_{i,t}-θ_{i,t-1})
] + ε_{i,t}
```

Where:
- IR: Perceived irrationality
- w: Social kernel weights
- M: Value masks
- Z: Participation gate
- ε: Population-dependent noise

## Baseline Experiments

### B-1: Phase Diagram (λ × κ)
Explores transitions between gas → pluralism → monoculture phases.

### B-2: Curvature Sweep (γ)
Tests how distortion strength affects clustering and irrationality.

### B-3: Population Shock
Examines stability under sudden population increases.

## Metrics

- **Number of Clusters**: DBSCAN-based worldview clustering
- **Mean Irrationality**: Average perceived irrationality across agent pairs
- **Diversity**: Mean pairwise distance in worldview space
- **Participation Rate**: Fraction of active agents
- **Gradient Alignment**: Cosine similarity of perceived gradients

## Testing

```bash
# Run all tests
pytest

# Run specific test module
pytest k_cdd_plus/tests/test_state.py -v

# Run with coverage
pytest --cov=k_cdd_plus
```

## Project Structure

```
k_cdd_plus/
├── core/
│   ├── state.py            # Utility and state
│   ├── distortions.py      # Worldview transformations
│   ├── kernels.py          # Social coupling
│   ├── masks.py            # Issue triggers
│   ├── agent.py            # Agent dynamics
│   └── world.py            # Simulation orchestration
├── experiments/
│   ├── baseline.py         # Basic experiments
│   ├── phase_diagram.py    # Parameter sweeps
│   └── interventions.py    # Intervention tests
├── interventions/          # (v1.1 - Coming soon)
│   ├── lens_tilt.py
│   ├── mediator.py
│   └── noise_spike.py
└── tests/                  # Unit tests
```

## Roadmap

- **v1.0** (Current): Core simulation with baseline experiments
- **v1.1**: Intervention framework (lens tilt, mediators, noise spikes)
- **v1.2**: Dynamic world states and agent actions
- **v1.3**: Empirical validation with real-world data

## Citation

If you use this code in your research, please cite:

```bibtex
@software{kcdd_plus,
  title = {K C D D-PLUS: A Multi-Agent Simulator for Worldview Dynamics},
  version = {1.0},
  year = {2024}
}
```

## License

MIT License - see LICENSE file for details.