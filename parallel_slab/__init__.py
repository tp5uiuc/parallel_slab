#!/usr/bin/env python3

__doc__ = """Shear benchmark for an elastic solid--fluid layers sandwiched between two oscillatory moving walls"""

from .solutions import (
    ProblemSolution,
    NeoHookeanSolution,
    GeneralizedMooneyRivlinSolution,
)
from .driver import run_and_plot, run, plot_solution
from .driver import run_and_plot_from_yaml, run_from_yaml, plot_from_yaml
