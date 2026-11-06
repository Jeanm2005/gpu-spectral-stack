"""Two-dimensional incompressible flow, the physics behind the Phase 2 training data.

Vorticity form on the periodic box [0, 2 pi)^2, x along the last array axis, y along the one
before it:

    d omega / dt + u . grad omega = nu lap omega - alpha omega + f
    u = (d psi / dy, -d psi / dx),   lap psi = -omega

with Kolmogorov forcing f(y) = -A k_f cos(k_f y), the curl of the body force A sin(k_f y) in x.

    flow.params      Flow: the physical parameters, and the spectral grid they live on
    flow.reference   the float64 NumPy solver (Task 2.1): the ground truth, runs anywhere
    flow.gpu         the batched float32 GPU solver on our FFTs (Task 2.2)
"""

from .params import Flow, Grid

__all__ = ["Flow", "Grid"]
