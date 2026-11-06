"""The float64 NumPy Navier-Stokes solver (Task 2.1) against exact solutions and its own budgets.
CPU only, so CI runs all of it."""

import numpy as np
import pytest

from spectral.flow import Flow, Grid
from spectral.flow import reference as ref
from spectral.flow.params import laminar_kolmogorov, taylor_green


def smooth_random_field(rng: np.random.Generator, n: int, batch: tuple = ()) -> np.ndarray:
    """A random vorticity field with energy at wavenumbers below about 6, zero mean."""
    grid = Grid(n)
    noise = rng.standard_normal((*batch, n, n))
    filtered = ref.rfft2(noise) * np.exp(-grid.k2 / 18.0)
    filtered[..., 0, 0] = 0
    w = ref.irfft2(filtered, n)
    return w / w.std(axis=(-2, -1), keepdims=True)


def relative(a, b) -> float:
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


# --- Grid -------------------------------------------------------------------------------------


def test_grid_wavenumbers_follow_the_rfft2_layout():
    g = Grid(8)
    assert g.kx.ravel().tolist() == [0, 1, 2, 3, 4]
    assert g.ky.ravel().tolist() == [0, 1, 2, 3, -4, -3, -2, -1]
    assert g.inverse_k2[0, 0] == 0 and g.inverse_k2[1, 2] == pytest.approx(1 / 5)


def test_two_thirds_rule_keeps_wavenumbers_up_to_n_over_3():
    mask = Grid(64).dealias
    kept_kx = np.flatnonzero(mask[0])
    assert kept_kx.max() == 21  # 64 / 3 = 21.3
    assert mask[21, 0] and not mask[22, 0] and mask[64 - 21, 0] and not mask[64 - 22, 0]


def test_rejects_a_grid_that_is_not_a_power_of_two():
    with pytest.raises(ValueError, match="power of two"):
        Grid(48)


# --- Exact solutions --------------------------------------------------------------------------


def test_taylor_green_decays_exactly():
    grid, nu, alpha, dt, steps = Grid(32), 0.01, 0.1, 0.01, 200
    flow = Flow(viscosity=nu, drag=alpha, forcing_amplitude=0.0)
    w0 = taylor_green(grid, 0.0, nu, alpha)
    frames = ref.simulate(w0, flow, dt, steps, record_every=steps)
    exact = taylor_green(grid, dt * steps, nu, alpha)
    assert relative(frames[-1], exact) < 1e-6


def test_laminar_kolmogorov_flow_is_a_fixed_point():
    grid = Grid(32)
    flow = Flow(viscosity=0.01)
    w = laminar_kolmogorov(grid, flow)
    frames = ref.simulate(w, flow, dt=0.05, steps=100, record_every=100)
    assert relative(frames[-1], w) < 1e-12


def test_flow_from_rest_settles_onto_the_laminar_solution():
    # Pure shear has no nonlinear term, so from rest the flow stays a shear flow and relaxes at
    # rate nu k_f^2 + alpha onto the laminar solution.
    grid = Grid(16)
    flow = Flow(viscosity=0.01, drag=0.1)
    frames = ref.simulate(np.zeros((16, 16)), flow, dt=0.05, steps=4000, record_every=4000)
    assert relative(frames[-1], laminar_kolmogorov(grid, flow)) < 1e-6


# --- Budgets and conservation -----------------------------------------------------------------


def test_energy_and_enstrophy_budgets_hold(rng):
    flow = Flow(viscosity=0.01, drag=0.1, forcing_amplitude=1.0)
    dt = 1e-3
    frames = ref.simulate(smooth_random_field(rng, 32), flow, dt, steps=2)
    d0, d2 = ref.diagnostics(frames[0], flow), ref.diagnostics(frames[2], flow)
    de, dz = ref.budget_rates(frames[1], flow)  # central difference around the middle frame
    measured_de = (d2["energy"] - d0["energy"]) / (2 * dt)
    measured_dz = (d2["enstrophy"] - d0["enstrophy"]) / (2 * dt)
    assert measured_de == pytest.approx(de, rel=1e-4, abs=1e-8)
    assert measured_dz == pytest.approx(dz, rel=1e-4, abs=1e-8)


def test_inviscid_unforced_flow_conserves_energy_and_enstrophy(rng):
    # With nu = alpha = A = 0 only the dealiased nonlinear term acts, and it conserves both
    # quadratic invariants; what is left is Heun's time-stepping error, O(dt^3) per step.
    flow = Flow(viscosity=0.0, drag=0.0, forcing_amplitude=0.0)
    frames = ref.simulate(smooth_random_field(rng, 32), flow, dt=1e-3, steps=200, record_every=200)
    d0, d1 = ref.diagnostics(frames[0], flow), ref.diagnostics(frames[-1], flow)
    assert d1["energy"] == pytest.approx(d0["energy"], rel=1e-6)
    assert d1["enstrophy"] == pytest.approx(d0["enstrophy"], rel=1e-6)


def test_mean_vorticity_stays_zero_and_output_is_real(rng):
    frames = ref.simulate(smooth_random_field(rng, 16), Flow(), dt=0.01, steps=20, record_every=5)
    assert frames.shape == (5, 16, 16)
    assert frames.dtype == np.float64
    assert np.abs(frames.mean(axis=(-2, -1))).max() < 1e-12


def test_second_order_in_time(rng):
    flow = Flow(viscosity=0.01, drag=0.1)
    w0 = smooth_random_field(rng, 32)
    t = 0.2

    def final(dt):
        return ref.simulate(w0, flow, dt, steps=round(t / dt), record_every=round(t / dt))[-1]

    fine = final(0.00125)
    e1 = relative(final(0.01), fine)
    e2 = relative(final(0.005), fine)
    assert 3.0 < e1 / e2 < 5.0  # halving dt divides the error by about 4


# --- Batches ----------------------------------------------------------------------------------


def test_a_batch_with_one_viscosity_each_matches_separate_runs(rng):
    w0 = smooth_random_field(rng, 16, batch=(3,))
    nus = np.array([1e-3, 5e-3, 2e-2])
    together = ref.simulate(w0, Flow(viscosity=nus), dt=0.01, steps=30, record_every=30)
    for b, nu in enumerate(nus):
        alone = ref.simulate(w0[b], Flow(viscosity=float(nu)), dt=0.01, steps=30, record_every=30)
        np.testing.assert_allclose(together[:, b], alone, rtol=0, atol=1e-12)


def test_diagnostics_work_on_batches(rng):
    w = smooth_random_field(rng, 16, batch=(2, 3))
    d = ref.diagnostics(w, Flow())
    assert d["energy"].shape == (2, 3)
    assert np.all(d["enstrophy"] > 0)


def test_rejects_bad_step_counts():
    with pytest.raises(ValueError, match="steps"):
        ref.simulate(np.zeros((8, 8)), Flow(), dt=0.1, steps=-1)
