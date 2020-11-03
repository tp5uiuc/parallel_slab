#!/usr/bin/env python3

__doc__ = """Test driving interface"""

import pytest
import numpy as np
import tempfile

from typing import Dict

# our
from parallel_slab.driver import _internal_load


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
