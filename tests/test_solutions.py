#!/usr/bin/env python3

__doc__ = """Test solution classes"""

import pytest
import numpy as np
from numpy.testing import assert_allclose

from typing import Dict

# our
from parallel_slab.solutions import (
    NeoHookeanSolution,
    GeneralizedMooneyRivlinSolution,
    SolutionGenerator,
    ProblemSolution,
)


# Neo
# 1. check for smoothness
# 2. check for couette flow solution when c_1 = 0

# Gen
# 0. check with coutte Stokes flow when c_1 = c_3 = 0
# 1. check with Neo for c_3 = 0


class StokesCouetteSolution:
    """Generates stokes coutte solution
    see Landau, L. D., & Lifshitz, E. M. (1987). Fluid Mechanics: Vol 6. pp. 88
    """

    def __init__(self, h, nu, omega, V_wall):
        self.h = h
        self.nu = nu
        self.omega = omega
        self.T = 2.0 * np.pi / omega
        self.V_wall = V_wall

    @classmethod
    def from_params(cls, params: Dict[str, float]):
        return cls(
            params["L_s"] + params["L_f"],
            params["mu_f"] / params["rho_f"],
            params["omega"],
            params["V_wall"],
        )

    def get_velocity(self, grid: np.ndarray) -> SolutionGenerator:
        # check
        assert np.all(np.logical_and(grid >= 0.0, grid <= self.h))
        # Note : to accomodate the cos -> sin transition, two changse are neede
        # First k is 1.0 - 1.0j instead of 1.0 + 1.0j
        # Then change np.real to np.imag

        k = (1.0 - 1.0j) * np.sqrt(self.omega / self.nu) / np.sqrt(2)

        def __solution(time_v):
            return self.V_wall * np.imag(
                np.exp(1j * (self.omega * time_v))
                * np.sin(k * grid)
                / np.sin(k * self.h)
            )

        return __solution


@pytest.fixture(scope="module")
def grid():
    from parallel_slab.utils import generate_regular_grid

    def generator(L_s, L_f):
        # Generate a fixed mesh of 101 points
        sg, fg = generate_regular_grid(101, L_s, L_f)
        tg = np.hstack((sg, L_s + fg))
        return sg, fg, tg

    return generator


def _internal_test(sol: ProblemSolution):
    # First run till some time
    sol.run_till(sol.time_period * 1.1)

    # The solution should be ready to post process immediately
    assert sol.ready()

    from parallel_slab.utils import generate_regular_grid

    sg, fg = generate_regular_grid(101, sol.L_s, sol.L_f)

    vsg, vfg = sol.get_velocities(sg, fg)

    # Check at 20 random points
    times = np.random.random(20) * sol.time_period
    for t in times:
        assert_allclose(vsg(t), 0.0)
        assert_allclose(vfg(t), 0.0)


class TestNeoHookeanSolution:
    default_params = {
        "L_s": 0.2,
        "n_modes": 64,
        "L_f": 0.2,
        "rho_f": 1.0,
        "mu_f": 0.02,
        "rho_s": 1.0,
        "mu_s": 0.02,
        "c_1": 0.01,
        "V_wall": 0.4,
        "omega": np.pi,
    }

    @pytest.fixture(scope="function")
    def params(self):
        params = self.default_params.copy()
        for k, _ in params.items():
            if k != "n_modes":
                params[k] = np.random.random()

        # Fix it so that time-period is a perfect float
        params["omega"] = np.random.randint(1, 10) * np.pi

        return params

    def test_behavior(self, grid, params):
        params["V_wall"] = 0.0

        # Test behavior of solution characteristics
        sol = NeoHookeanSolution(params=params)
        assert sol.ready()
        _internal_test(sol)

        # Test behavior of solution description
        assert str(sol) == "NeoHookeanSolution"

    def test_against_stokes_coutte_flow(self, grid, params):
        sg, fg, tg = grid(params["L_s"], params["L_f"])

        # Make it a fluid
        params["c_1"] = 0.0
        params["mu_s"] = params["mu_f"]
        params["rho_s"] = params["rho_f"]

        sol = NeoHookeanSolution(params=params)
        vsg, vfg = sol.get_velocities(sg, fg)

        ref_sol = StokesCouetteSolution.from_params(params=params)
        ref_v = ref_sol.get_velocity(tg)

        # Check at 20 random points
        # times = np.random.random(20) * sol.time_period
        times = np.linspace(0.0, 1.0, 21) * sol.time_period
        for t in times:
            try:
                assert_allclose(
                    np.hstack((vsg(t), vfg(t))), ref_v(t), rtol=1e-5, atol=1e-2
                )
            except AssertionError:
                print("Time of mismatch :", t)
                print(params)
                raise

    def test_warning_for_extraneous_c3_parameter(self, params):
        # Set random parameter to c_3
        params["c_3"] = 0.4

        with pytest.warns(UserWarning):
            _ = NeoHookeanSolution(params=params)

    def test_data_serialization(self, params):
        # No behavior here, just check the state is equal or not
        sol = NeoHookeanSolution(params)
        sol_copy = sol.__dict__.copy()
        import tempfile

        with tempfile.TemporaryDirectory() as dirpath:
            sol.save_data(dirpath)
            sol.load_data(dirpath)

        assert sol.__dict__ == sol_copy


class TestGeneralizedMooneyRivlinSolution:
    default_params = {
        "L_s": 0.2,
        "n_modes": 64,
        "L_f": 0.2,
        "rho_f": 1.0,
        "mu_f": 0.02,
        "rho_s": 1.0,
        "mu_s": 0.02,
        "c_1": 0.01,
        "c_3": 0.04,
        "V_wall": 0.4,
        "omega": np.pi,
    }

    @pytest.fixture(scope="function")
    def params(self):
        params = self.default_params.copy()
        for k, _ in params.items():
            if k != "n_modes":
                params[k] = np.random.random()

        # Fix it so that time-period is a perfect float
        params["omega"] = np.random.randint(1, 10) * np.pi

        return params

    def test_behavior(self, grid, params):
        params["V_wall"] = 0.0

        # Test behavior of solution characterization
        sol = GeneralizedMooneyRivlinSolution(params=params)
        _internal_test(sol)

        # Test behavior of solution description
        assert str(sol) == "GeneralizedMooneyRivlinSolution"

    def test_against_stokes_coutte_flow(self, grid, params):
        sg, fg, tg = grid(params["L_s"], params["L_f"])

        # Make it a fluid
        params["c_1"] = 0.0
        params["c_3"] = 0.0
        params["mu_s"] = params["mu_f"]
        params["rho_s"] = params["rho_f"]
        params["n_modes"] = 16

        sol = GeneralizedMooneyRivlinSolution(params=params)
        # First run till some time
        sol.run_till(10.0 * sol.time_period)
        vsg, vfg = sol.get_velocities(sg, fg)

        ref_sol = StokesCouetteSolution.from_params(params=params)
        ref_v = ref_sol.get_velocity(tg)

        # Check at 20 random points
        # times = np.random.random(20) * sol.time_period
        times = np.linspace(0.0, 1.0, 21) * sol.time_period
        for t in times:
            try:
                assert_allclose(
                    np.hstack((vsg(t), vfg(t))), ref_v(t), rtol=1e-5, atol=1e-2
                )
            except AssertionError:
                print("Time of mismatch :", t)
                print(params)
                raise

    def test_against_neo_hookean_material(self, grid, params):
        sg, fg, _ = grid(params["L_s"], params["L_f"])

        # Make it a fluid
        params["c_3"] = 0.0
        params["n_modes"] = 16

        # These two are not strictly needed, but otherwise this
        # thing blows up
        params["mu_f"] = 0.02
        params["mu_s"] = 0.002

        sol = GeneralizedMooneyRivlinSolution(params=params)
        # First run till some time
        sol.run_till(10.0 * sol.time_period)
        vsg, vfg = sol.get_velocities(sg, fg)

        ref_sol = NeoHookeanSolution(params=params)
        ref_sol.run_till(10.0 * sol.time_period)
        ref_vsg, ref_vfg = ref_sol.get_velocities(sg, fg)

        # Check at 20 random points
        # times = np.random.random(20) * sol.time_period
        times = np.linspace(0.0, 1.0, 21) * sol.time_period
        for t in times:
            try:
                assert_allclose(vsg(t), ref_vsg(t), rtol=1e-4, atol=1e-2)
                assert_allclose(vfg(t), ref_vfg(t), rtol=1e-4, atol=1e-2)
            except AssertionError:
                print("Time of mismatch :", t)
                print(params)
                raise

    def test_data_serialization(self, params):
        params["n_modes"] = 16

        # Data
        sol = GeneralizedMooneyRivlinSolution(params=params)
        # No data
        new_sol = GeneralizedMooneyRivlinSolution(params=params)

        sol.run_till(2.0 * sol.time_period)

        import tempfile

        with tempfile.TemporaryDirectory() as dirpath:
            sol.save_data(dirpath)
            new_sol.load_data(dirpath)

            assert new_sol.ready()
            assert new_sol.recorded_data.keys() == sol.recorded_data.keys()

            # We don't implement an equality operator by design so do custom
            # checks
            K = sol.recorded_data.keys()
            for k in K:
                # recoreded data is a dict of dicts
                sol_dict = sol.recorded_data[k]
                new_sol_dict = new_sol.recorded_data[k]

                deepK = sol_dict.keys()
                assert new_sol_dict.keys() == deepK

                for dk in deepK:
                    if isinstance(
                        sol_dict[dk],
                        (
                            np.ndarray,
                            np.float64,
                        ),
                    ):
                        assert_allclose(sol_dict[dk], new_sol_dict[dk])
                    else:
                        assert sol_dict[dk] == new_sol_dict[dk]
