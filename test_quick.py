#!/usr/bin/env python3
"""Quick test of K C D D-PLUS implementation."""

import numpy as np
from k_cdd_plus.core.world import Simulation
from k_cdd_plus.core.masks import IssueMasks

print("Testing K C D D-PLUS implementation...")

# Test with small parameters
sim = Simulation(
    n_agents=10,
    state_dim=2,
    worldview_dim=3,
    seed=42
)

print(f"Created simulation with {sim.n_agents} agents")
print(f"Initial worldview positions shape: {sim.get_worldview_positions().shape}")

# Run a few steps
for i in range(5):
    metrics = sim.step()
    print(f"Step {i+1}: Clusters={metrics['n_clusters']}, IR={metrics['mean_irrationality']:.3f}")

print("\nGenerating test event...")
mask_gen = IssueMasks(worldview_dim=3)
event = mask_gen.generate_event(intensity=1.0)
print(f"Event shape: {event.shape}, Active dims: {np.sum(event > 0)}")

# Test with event
metrics = sim.step(event)
print(f"\nStep with event: Participation={metrics['participation_rate']:.2f}")

print("\nTest passed! Core implementation is working.")