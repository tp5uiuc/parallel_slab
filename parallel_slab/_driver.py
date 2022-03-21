#!/usr/bin/env python3

__doc__ = """Implementation details to access solutions"""

import os
from functools import partial
from typing import Callable, Dict, List, Tuple, Union

import numpy as np

from .solutions import (
    GeneralizedMooneyRivlinSolution,
    NeoHookeanSolution,
    ProblemSolution,
)
from .utils import generate_regular_grid

Grids = Tuple[np.ndarray, np.ndarray]
FilePath = Union[str, bytes, os.PathLike]


def run(solution: ProblemSolution, final_time: float, file_path) -> ProblemSolution:
    if isinstance(solution, GeneralizedMooneyRivlinSolution):
        if not os.path.exists(solution.get_file_id(file_path)):
            # Run solution till terminal time
            solution.run_till(final_time)

            # Save data, useful for nonlinear solutions
            solution.save_data(file_path)

    return solution


def run_NeoHookeanSolution_from(
    params: Dict[str, float], final_time: float, file_path
) -> ProblemSolution:
    """
    For interfacing with the online sandbox

    Parameters
    ----------
    params
    final_time
    file_path

    Returns
    -------

    """
    return run(
        solution=NeoHookeanSolution(params), final_time=final_time, file_path=file_path
    )


class SandboxInterface:
    """
    An interface for the online sandbox
    """

    def __init__(
        self,
        nd_plot_times: np.ndarray,
        index: int,
        trace_location: np.ndarray,
        trace_data: np.ndarray,
    ):
        self.nd_trace_times = np.copy(nd_plot_times)
        self.separation_index = index
        self.trace_location = np.copy(trace_location)
        self.trace_data = np.copy(trace_data)

        self.colors = np.array(
            [
                [0.01960784, 0.18823529, 0.38039216, 1.0],
                [0.14422658, 0.41960784, 0.68453159, 1.0],
                [0.33159041, 0.62004357, 0.78823529, 1.0],
                [0.65490196, 0.81437908, 0.89411765, 1.0],
                [0.88583878, 0.92941176, 0.95337691, 1.0],
                [0.98169935, 0.90762527, 0.86405229, 1.0],
                [0.96862745, 0.71764706, 0.6, 1.0],
                [0.86535948, 0.43660131, 0.34814815, 1.0],
                [0.71372549, 0.1254902, 0.18344227, 1.0],
                [0.40392157, 0.0, 0.12156863, 1.0],
                [0.40392157, 0.0, 0.12156863, 1.0],
                [0.71372549, 0.1254902, 0.18344227, 1.0],
                [0.86535948, 0.43660131, 0.34814815, 1.0],
                [0.96862745, 0.71764706, 0.6, 1.0],
                [0.98169935, 0.90762527, 0.86405229, 1.0],
                [0.88583878, 0.92941176, 0.95337691, 1.0],
                [0.65490196, 0.81437908, 0.89411765, 1.0],
                [0.33159041, 0.62004357, 0.78823529, 1.0],
                [0.14422658, 0.41960784, 0.68453159, 1.0],
                [0.01960784, 0.18823529, 0.38039216, 1.0],
            ],
        )

        # # twilight color scheme, 20 values
        # self.colors = np.array(
        #     [
        #         [0.88575016, 0.85000925, 0.88797365, 1.0],
        #         [0.79454305, 0.82245116, 0.84392185, 1.0],
        #         [0.63802587, 0.74331377, 0.791151, 1.0],
        #         [0.50288535, 0.64897217, 0.76521447, 1.0],
        #         [0.41572233, 0.54494883, 0.74809204, 1.0],
        #         [0.37841364, 0.43080859, 0.72259932, 1.0],
        #         [0.36943335, 0.30692924, 0.67205053, 1.0],
        #         [0.35537415, 0.18380393, 0.58093191, 1.0],
        #         [0.29942484, 0.08968225, 0.4202162, 1.0],
        #         [0.21482844, 0.06594039, 0.26191676, 1.0],
        #         [0.21930504, 0.06746579, 0.22451024, 1.0],
        #         [0.33533353, 0.08353243, 0.27632534, 1.0],
        #         [0.47384144, 0.12426354, 0.3122547, 1.0],
        #         [0.58894862, 0.19997021, 0.31288922, 1.0],
        #         [0.67907474, 0.30769498, 0.31665546, 1.0],
        #         [0.74259922, 0.42994254, 0.35352709, 1.0],
        #         [0.7817418, 0.55975098, 0.44624687, 1.0],
        #         [0.81269922, 0.68713034, 0.60005215, 1.0],
        #         [0.85775663, 0.79801963, 0.77829572, 1.0],
        #         [0.88571155, 0.85002186, 0.88572539, 1.0],
        #     ]
        # )

    # should also defined call based on the actual time, rather than the
    # index,
    def __call__(self, temporal_index: int):
        # if in between times interpolate the velocities, skip for now
        # solid y, solid v, fluid y, fluid v
        return [
            self.trace_data[temporal_index, : self.separation_index],
            self.trace_location[temporal_index, : self.separation_index],
            self.trace_data[temporal_index, self.separation_index :],
            self.trace_location[temporal_index, self.separation_index :],
        ]

    def get_colormap(self, temporal_index: int):
        # hardcode the values and color

        n_colors = self.colors.shape[0]

        # sample from 0.0 to 1.0, convert to next 1/20
        def col_idx(sample):
            return int(np.floor(n_colors * sample))

        def to_hex(rgba):
            r, g, b, _ = rgba
            return "#%02x%02x%02x" % tuple(map(lambda x: int(255 * x), (r, g, b)))

        def to_js_color(sample):
            # print(col_idx(float(sample) / n_samples))
            return to_hex(self.colors[col_idx(float(sample))])

        return to_js_color(self.nd_trace_times[temporal_index])


def get_NeoHookeanSolution_data_from(
    params: Dict[str, float],
    plot_times: List[float],
    file_path: FilePath = "",  # does this work?
    grid_generator: Callable[[float, float], Grids] = partial(
        generate_regular_grid, 128
    ),
):
    # Assume plot_times is ordered and between [0, T)
    solution = run_NeoHookeanSolution_from(
        params, final_time=plot_times[-1], file_path=file_path
    )

    # Load data from disk and get ready for plotting
    if not solution.ready():
        solution.load_data(file_path=file_path)

    solid_grid, fluid_grid = grid_generator(solution.L_s, solution.L_f)
    v_solid_gen, v_fluid_gen = solution.get_velocities(solid_grid, fluid_grid)

    max_velocity = solution.V_wall
    max_length = solution.L_s + solution.L_f

    v_solid = v_solid_gen(plot_times[0])
    v_fluid = v_fluid_gen(plot_times[0])

    # Gets it at one time level
    total_v = np.hstack((v_solid, v_fluid))
    total_y = np.hstack(
        (solid_grid / max_length, (solution.L_s + fluid_grid) / max_length)
    )

    # Gets it at multiple time levels of shape (plot_times, n_data)
    cache_total_v = np.zeros((len(plot_times), total_v.shape[0]))
    cache_total_y = np.repeat(total_y.reshape(1, -1), repeats=len(plot_times), axis=0)

    for i, t in enumerate(plot_times):
        cache_total_v[i, :] = np.hstack((v_solid_gen(t), v_fluid_gen(t)))

    cache_total_v /= max_velocity

    return SandboxInterface(
        np.array(plot_times) / solution.time_period,
        index=v_solid.shape[0],
        trace_location=cache_total_y,
        trace_data=cache_total_v,
    )
