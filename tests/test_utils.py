#!/usr/bin/env python3

__doc__ = """Test utility scripts"""

import pytest
import numpy as np
from numpy.testing import assert_allclose

# our
from parallel_slab.utils import (
    generate_grid_like_pycfs,
    generate_regular_grid,
    dict_hash,
)


class TestGrids:
    @pytest.mark.parametrize("gen", [generate_regular_grid, generate_grid_like_pycfs])
    @pytest.mark.parametrize("npoints", [2 ** i for i in range(7, 11)])
    def test_range_of_grid(self, gen, npoints):
        total_length = 0.5
        l_half_solid = total_length * np.random.random()
        # l_half_solid = total_length * 0.5
        l_half_fluid = total_length - l_half_solid
        sg, fg = gen(
            n_total_points=npoints, l_half_solid=l_half_solid, l_half_fluid=l_half_fluid
        )

        def test_grid(grid, max_condition):
            idx1 = (
                grid >= 0.0
            )  # Equals for testing some case where we exactly specify start location
            idx2 = grid <= max_condition
            idx = np.logical_and(idx1, idx2)
            assert np.all(idx)

        # Test extent
        test_grid(sg, l_half_solid)
        test_grid(fg, l_half_fluid)

    @pytest.mark.parametrize("npoints", [2 ** i for i in range(7, 11)])
    def test_one_away(self, npoints):
        total_length = 0.5
        l_half_solid = total_length * np.random.random()
        # l_half_solid = total_length * 0.5
        l_half_fluid = total_length - l_half_solid
        sg, fg = generate_grid_like_pycfs(
            n_total_points=npoints, l_half_solid=l_half_solid, l_half_fluid=l_half_fluid
        )

        # Test they are one point away
        assert_allclose(fg[0] + l_half_solid - sg[-1], 1.0 / npoints)


class TestDictionaryHash:
    hashes = []

    @pytest.fixture(scope="class")
    def generate_dictionary(self):
        def gen(n_entries):
            keys = []
            values = []
            for _ in range(n_entries):
                key = ""
                for _ in range(5):
                    code = np.random.randint(65, 90 + 1)
                    key += chr(code)
                keys.append(key)
                code = np.random.randint(48, 57 + 1)
                values.append(code)
                # number = 48, 57
                # large_str = 65, 90
                # small_str = 97, 122
            return dict(zip(keys, values))

        return gen

    @pytest.mark.parametrize("n_entries", range(2, 5))
    def test_unique_hash(self, n_entries, generate_dictionary):
        k = dict_hash(generate_dictionary(n_entries))
        assert k not in self.hashes
        self.hashes.append(k)

    @pytest.mark.parametrize("n_entries", range(2, 5))
    def same_hash(self, n_entries, generate_dictionary):
        D = generate_dictionary(n_entries)
        K = reversed(D.keys())
        ND = {k: D[k] for k in K}
        assert dict_hash(D) == dict_hash(ND)
