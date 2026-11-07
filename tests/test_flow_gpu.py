"""The GPU Navier-Stokes solver (Task 2.2): exact solutions, budgets, and agreement with the
float64 reference solver, for each FFT backend, in the full and the pruned spectral layout."""

import numpy as np
import pytest
import torch

from spectral.flow import Flow, Grid
from spectral.flow import reference as ref
from spectral.flow.gpu import GpuSolver
from spectral.flow.params import laminar_kolmogorov, taylor_green

pytestmark = pytest.mark.gpu

CONFIGS = [
    ("cufft", False),
    ("cufft", True),
    ("stockham_r8", False),
    ("stockham_r8", True),
    ("stockham_r4", True),
    ("naive", False),
]
IDS = [f"{impl}-{'pruned' if pruned else 'full'}" for impl, pruned in CONFIGS]


def smooth_random_field(rng, n, batch=()):
    grid = Grid(n)
    filtered = ref.rfft2(rng.standard_normal((*batch, n, n))) * np.exp(-grid.k2 / 18.0)
    filtered[..., 0, 0] = 0
    w = ref.irfft2(filtered, n)
    return w / w.std(axis=(-2, -1), keepdims=True)


def run(w0, flow, dt, steps, impl, pruned, record_every=None):
    solver = GpuSolver(w0.shape[-1], flow, dt, fft_impl=impl, pruned=pruned)
    w = torch.as_tensor(w0, dtype=torch.float32, device="cuda")
    out = solver.simulate(w, steps, record_every or max(steps, 1))
    return out.double().cpu().numpy()


def relative(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


@pytest.mark.parametrize(("impl", "pruned"), CONFIGS, ids=IDS)
def test_taylor_green_decays_exactly(impl, pruned):
    grid, nu, alpha = Grid(32), 0.01, 0.1
    flow = Flow(viscosity=nu, drag=alpha, forcing_amplitude=0.0)
    frames = run(taylor_green(grid, 0.0, nu, alpha), flow, 0.01, 200, impl, pruned)
    assert relative(frames[-1], taylor_green(grid, 2.0, nu, alpha)) < 1e-5


@pytest.mark.parametrize(("impl", "pruned"), CONFIGS, ids=IDS)
def test_laminar_kolmogorov_flow_is_a_fixed_point(impl, pruned):
    grid, flow = Grid(32), Flow(viscosity=0.01)
    w = laminar_kolmogorov(grid, flow)
    assert relative(run(w, flow, 0.05, 100, impl, pruned)[-1], w) < 1e-5


@pytest.mark.parametrize(("impl", "pruned"), CONFIGS, ids=IDS)
def test_agrees_with_the_float64_reference(rng, impl, pruned):
    # A forced, nonlinear flow; short enough that chaos does not amplify float32 rounding.
    flow = Flow(viscosity=1e-3, drag=0.1)
    w0 = smooth_random_field(rng, 64)
    gpu = run(w0, flow, 5e-3, 200, impl, pruned)
    cpu = ref.simulate(w0, flow, 5e-3, 200, record_every=200)
    assert relative(gpu[-1], cpu[-1]) < 1e-4


@pytest.mark.parametrize(("impl", "pruned"), CONFIGS, ids=IDS)
def test_energy_budget_holds(rng, impl, pruned):
    flow = Flow(viscosity=0.01, drag=0.1)
    dt = 1e-3
    frames = run(smooth_random_field(rng, 32), flow, dt, 2, impl, pruned, record_every=1)
    d0, d2 = ref.diagnostics(frames[0], flow), ref.diagnostics(frames[2], flow)
    de, dz = ref.budget_rates(frames[1], flow)
    assert (d2["energy"] - d0["energy"]) / (2 * dt) == pytest.approx(de, rel=2e-3)
    assert (d2["enstrophy"] - d0["enstrophy"]) / (2 * dt) == pytest.approx(dz, rel=2e-3)


def test_pruned_and_full_layouts_agree(rng):
    flow = Flow(viscosity=1e-3)
    w0 = smooth_random_field(rng, 128, batch=(2,))
    full = run(w0, flow, 2e-3, 100, "stockham_r8", False)
    pruned = run(w0, flow, 2e-3, 100, "stockham_r8", True)
    assert relative(pruned, full) < 1e-5


def test_batch_with_one_viscosity_each_matches_the_reference(rng):
    nus = np.array([1e-3, 4e-3, 1e-2])
    flow = Flow(viscosity=nus)
    w0 = smooth_random_field(rng, 32, batch=(3,))
    gpu = run(w0, flow, 5e-3, 100, "stockham_r8", True)
    cpu = ref.simulate(w0, flow, 5e-3, 100, record_every=100)
    for b in range(3):
        assert relative(gpu[-1, b], cpu[-1, b]) < 1e-4


def test_frames_streams_every_record_every_steps(rng):
    solver = GpuSolver(16, Flow(), 0.01)
    w0 = torch.as_tensor(smooth_random_field(rng, 16), dtype=torch.float32, device="cuda")
    frames = list(solver.frames(w0, steps=10, record_every=5))
    assert len(frames) == 3 and frames[0].shape == (16, 16)
    assert frames[0].dtype == torch.float32


def test_rejects_an_unknown_backend():
    with pytest.raises(ValueError, match="unknown fft_impl"):
        GpuSolver(16, Flow(), 0.01, fft_impl="fftw")
