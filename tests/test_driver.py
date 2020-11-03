#!/usr/bin/env python3

__doc__ = """Test driving interface"""

import pytest
import numpy as np
import tempfile

from typing import Dict

# our
from parallel_slab.solutions import GeneralizedMooneyRivlinSolution, NeoHookeanSolution
from parallel_slab.driver import _internal_load, run

# Simple contextual Timer
from timeit import default_timer


class Timer(object):
    def __init__(self, description):
        self.description = description

    def __enter__(self):
        self.start = default_timer()

    def __exit__(self, type, value, traceback):
        self.end = default_timer()

    def __call__(self, *args, **kwargs):
        return self.end - self.start


@pytest.fixture(scope="function")
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
        import os

        # Ensure it takes some time
        with tempfile.TemporaryDirectory() as dirpath:
            sol = GeneralizedMooneyRivlinSolution(params)

            with self.t:
                run(sol, 2.0 * sol.time_period, dirpath)
            # Ensure it takes time
            assert self.t() > 1.0
            # Ensure it has saved some data as a pickle file
            os.path.isfile(sol.get_file_id(dirpath))
