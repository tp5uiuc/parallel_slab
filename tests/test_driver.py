#!/usr/bin/env python3

__doc__ = """Test driving interface"""

import pytest
import numpy as np
import tempfile
import os
import sys

# Simple contextual Timer
from timeit import default_timer

# our
from parallel_slab.solutions import GeneralizedMooneyRivlinSolution, NeoHookeanSolution
from parallel_slab.driver import _internal_load, run, plot_solution


class Timer(object):
    def __init__(self, description):
        self.description = description

    def __enter__(self):
        self.start = default_timer()

    def __exit__(self, type, value, traceback):
        self.end = default_timer()

    def __call__(self, *args, **kwargs):
        return self.end - self.start


@pytest.fixture(scope="module")
def params():
    params = {
        "L_s": 0.2,
        "n_modes": 16,
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

    for k, _ in params.items():
        if k != "n_modes":
            params[k] = np.random.random()

    # Fix it so that time-period is a perfect float
    params["omega"] = np.random.randint(1, 10) * np.pi

    return params


class TestInternalLoad:
    def test_internal_throws_on_no_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml") as f:
            # with pytest.raises(IOError):
            _internal_load(f.name)

    def test_internal_throws_on_wrong_file(self):
        import yaml

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml") as f:
            f.write("s : 2: 4")
            # with pytest.raises(yaml.YAMLError):
            _ = _internal_load(f.name)

    def test_internal_load(self):
        import yaml

        r = range(65, 97)
        keys = [chr(i) for i in r]
        values = [float(i) for i in r]
        test_dict = dict(zip(keys, values))

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml") as f:
            yaml.safe_dump(test_dict, f)
            returned_dict = _internal_load(f.name)
            assert test_dict == returned_dict


class TestRun:
    t = Timer("RunTimer")

    def test_NeoHookean_run(self, params):
        # Ensure it takes no-little time
        with tempfile.TemporaryDirectory() as dirpath:
            with self.t:
                run(NeoHookeanSolution(params), 100.0, dirpath)
            # less than 0.01s to run
            assert self.t() < 0.01

    def test_GeneralizedMooneyRivlinSolution_run_with_existing_file(self, params):
        from pathlib import Path

        # Ensure it takes no-little time
        with tempfile.TemporaryDirectory() as dirpath:
            sol = GeneralizedMooneyRivlinSolution(params)

            # Create file
            Path(sol.get_file_id(dirpath)).touch()

            with self.t:
                run(sol, 100.0, dirpath)
            # less than 0.01s to run
            assert self.t() < 0.01

    def test_GeneralizedMooneyRivlinSolution_run_on_new_path(self, params):
        # Ensure it takes some time
        with tempfile.TemporaryDirectory() as dirpath:
            sol = GeneralizedMooneyRivlinSolution(params)

            with self.t:
                run(sol, 2.0 * sol.time_period, dirpath)
            # Ensure it takes time
            assert self.t() > 1.0
            # Ensure it has saved some data as a pickle file
            os.path.isfile(sol.get_file_id(dirpath))


# class Param:
#     def __init__(self, cp, d):
#         self.cls_param = cp
#         self.dir = d


class TestPlot:
    cls_params = [NeoHookeanSolution, GeneralizedMooneyRivlinSolution]

    # dirs = [tempfile.mkdtemp() for _ in cls_params]
    # A consistent setup tardown seems difficult here, so we run the simulations many times

    # @pytest.fixture(scope="class", params=[Param(x, y) for x, y in zip(cls_params, dirs)])
    @pytest.fixture(scope="class", params=cls_params)
    def solution_gen(self, params, request):
        sol = request.param(params)
        return lambda td: run(sol, 2.0 * sol.time_period, td)
        # return run(sol, 2.0 * sol.time_period, request.param.dir), request.param.dir

    # def run_solution(solution):
    #     return solution(td)

    def test_default_plot(self, solution_gen):
        with tempfile.TemporaryDirectory() as td:
            solution = solution_gen(td)

            n_samples = 5
            # Ensure it takes no-little time
            plot_times = (
                np.linspace(0.0, 1.0, n_samples, endpoint=False) * solution.time_period
            )
            plot_solution(solution, plot_times, td)

            # Check for velocities.pdf file in td
            assert os.path.isfile(os.path.join(td, "velocities.pdf"))

            # Check for n_samples csv file in td
            n_outputs = [x for x in os.listdir(td) if x.endswith(".csv")]

            assert len(n_outputs) == n_samples

    def test_default_plot_does_not_bug_out_for_empty_samples(self, solution_gen):
        # solution, td = solution_gen
        with tempfile.TemporaryDirectory() as td:
            solution = solution_gen(td)

            # Ensure it takes no-little time
            plot_solution(solution, [], td)

            # Check for velocities.pdf file in td
            assert os.path.isfile(os.path.join(td, "velocities.pdf"))

            # Check for n_samples csv file in td
            n_outputs = [x for x in os.listdir(td) if x.endswith(".csv")]

            assert len(n_outputs) == 0

    def test_default_plot_when_solution_is_not_ready(self, solution_gen):
        # solution, td = solution_gen
        with tempfile.TemporaryDirectory() as td:
            solution = solution_gen(td)

            n_samples = 5
            # Ensure it takes no-little time
            solution._data_loaded = False
            plot_times = (
                np.linspace(0.0, 1.0, n_samples, endpoint=False) * solution.time_period
            )
            plot_solution(solution, plot_times, td)

            # Check for velocities.pdf file in td
            assert os.path.isfile(os.path.join(td, "velocities.pdf"))

            # Check for n_samples csv file in td
            n_outputs = [x for x in os.listdir(td) if x.endswith(".csv")]

            assert len(n_outputs) == n_samples

    @pytest.mark.mpl_image_compare
    @pytest.mark.parametrize("cls", cls_params)
    def test_plot_output(self, cls):
        params = {
            "L_s": 0.2,
            "n_modes": 16,
            "L_f": 0.2,
            "rho_f": 1.0,
            "mu_f": 0.02,
            "rho_s": 1.0,
            "mu_s": 0.002,
            "c_1": 0.02,
            "c_3": 0.04,
            "V_wall": 0.4,
            "omega": np.pi,
        }
        with tempfile.TemporaryDirectory() as td:
            solution = cls(params=params)

            run(solution, 10.0 * solution.time_period, td)

            n_samples = 21
            # Ensure it takes no-little time
            plot_times = (
                np.linspace(0.0, 1.0, n_samples, endpoint=False) * solution.time_period
            )

            solution, fig = plot_solution(solution, plot_times, td)

        return fig

    @pytest.mark.xfail(
        sys.platform in ["darwin", "linux"], reason="no ffmpeg in travis CI"
    )
    def test_animated_plot(self, solution_gen):
        with tempfile.TemporaryDirectory() as td:
            solution = solution_gen(td)

            n_samples = 21
            # Ensure it takes no-little time
            plot_times = (
                np.linspace(0.0, 1.0, n_samples, endpoint=False) * solution.time_period
            )
            plot_solution(solution, plot_times, td, animate_flag=True)

            # Check for velocities.pdf file in td
            assert os.path.isfile(os.path.join(td, "movie.mp4"))

    @pytest.mark.xfail(
        sys.platform in ["darwin", "linux"], reason="no ffmpeg in travis CI"
    )
    def test_animated_plot_does_not_bug_out_for_empty_samples(self, solution_gen):
        # solution, td = solution_gen
        with tempfile.TemporaryDirectory() as td:
            solution = solution_gen(td)

            # Ensure it takes no-little time
            plot_solution(solution, [], td, animate_flag=True)

            # Check for velocities.pdf file in td
            assert os.path.isfile(os.path.join(td, "movie.mp4"))
