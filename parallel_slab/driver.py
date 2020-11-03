#!/usr/bin/env python3

__doc__ = """Main interface for accessing solutions"""

from .utils import generate_regular_grid
from .solutions import (
    ProblemSolution,
    NeoHookeanSolution,
    GeneralizedMooneyRivlinSolution,
)
from functools import partial

import numpy as np
from typing import List, Dict, Type, Callable, Tuple, Union
import os

Grids = Tuple[np.ndarray, np.ndarray]
FilePath = Union[str, bytes, os.PathLike]


def _get_stylized_plot():
    from matplotlib import pyplot as plt

    fig = plt.figure(figsize=(7, 6), dpi=300, tight_layout=True)
    ax = fig.add_subplot()
    ax.set_aspect("auto")
    ax.set_xlim([-1.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    from matplotlib.colors import to_rgb

    x = np.linspace(10000.0, 10001.0, 51)
    (solid_line,) = ax.plot(x, x, color=to_rgb("xkcd:reddish"), linewidth=3)
    (fluid_line,) = ax.plot(
        x,
        x,
        color=to_rgb("xkcd:bluish"),
        linewidth=3,
    )
    ax.set_xlabel("v / V_wall")
    ax.set_ylabel("y / (L_s + L_f)")
    ax.tick_params(direction="out")
    # ax.grid(True)

    return fig, ax, solid_line, fluid_line


def _internal_load(param_file_name: FilePath) -> Dict[str, float]:
    from yaml import safe_load, YAMLError

    try:
        with open(param_file_name) as f:
            try:
                params = safe_load(f)
            except YAMLError:
                raise
    except IOError:
        raise
    return params


def run(
    solution: ProblemSolution, final_time: float, file_path: FilePath
) -> ProblemSolution:
    if isinstance(solution, GeneralizedMooneyRivlinSolution):
        if not os.path.exists(solution.get_file_id(file_path)):
            # Run solution till terminal time
            solution.run_till(final_time)

            # Save data, useful for nonlinear solutions
            solution.save_data(file_path)

    return solution


def run_from_yaml(
    solution_type: Type[ProblemSolution],
    final_time: float,
    param_file_name: FilePath = "params.yaml",
    rel_file_path: str = "data",
):
    """
    Thin convenience wrapper around YAML files

    Parameters
    ----------
    solution_type
    final_time
    param_file_name

    Returns
    -------

    """
    file_path = os.path.join(os.path.dirname(param_file_name), rel_file_path)
    os.makedirs(file_path, exist_ok=True)
    return run(solution_type(_internal_load(param_file_name)), final_time, file_path)


def plot_solution(
    solution: ProblemSolution,
    plot_times: List[float],
    file_path: FilePath,
    animate_flag: bool = False,
    write_flag: bool = True,
    grid_generator: Callable[[float, float], Grids] = partial(
        generate_regular_grid, 512
    ),
) -> ProblemSolution:
    from matplotlib import pyplot as plt
    from matplotlib import cm
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    # Load data from disk and get ready for plotting
    if not solution.ready():
        solution.load_data(file_path=file_path)

    solid_grid, fluid_grid = grid_generator(solution.L_s, solution.L_f)
    v_solid_gen, v_fluid_gen = solution.get_velocities(solid_grid, fluid_grid)

    fig, ax, solid_line, fluid_line = _get_stylized_plot()
    plot_times.sort()

    v_solid = v_solid_gen(plot_times[0])
    v_fluid = v_fluid_gen(plot_times[0])

    max_velocity = solution.V_wall
    max_length = solution.L_s + solution.L_f
    v_solid /= max_velocity
    v_fluid /= max_velocity
    total_v = np.hstack((v_solid, v_fluid))
    total_y = np.hstack(
        (solid_grid / max_length, (solution.L_s + fluid_grid) / max_length)
    )

    # Plot horizontal line at the interface position
    ax.axhline(y=solution.L_s / max_length, c="k", linestyle="--", linewidth=2)

    if animate_flag:
        solid_line.set_ydata(solid_grid / max_length)
        fluid_line.set_ydata((solution.L_s + fluid_grid) / max_length)

        import matplotlib.animation as animation

        def animate(time_v):
            s_v = v_solid_gen(time_v)
            f_v = v_fluid_gen(time_v)

            solid_line.set_xdata(s_v / max_velocity)
            fluid_line.set_xdata(f_v / max_velocity)

            return (
                solid_line,
                fluid_line,
            )

        anim = animation.FuncAnimation(
            fig,
            animate,
            frames=plot_times,
            blit=True,
            repeat=True,
        )

        writer = animation.FFMpegWriter(fps=30, metadata=dict(artist="Me"))
        anim.save(os.path.join(file_path, "movie.mp4"), writer=writer)
    else:
        # Not animate flag, so plot according to colormap
        time_period = solution.time_period

        cmap = cm.twilight_shifted
        img = plt.imshow(np.array([[0, 1]]), cmap=cmap)
        img.set_visible(False)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)

        fig.colorbar(img, cax=cax, orientation="vertical")
        ax.set_aspect("auto")

        # Plot with colormap and save
        for t in plot_times:
            v_solid = v_solid_gen(t) / max_velocity
            v_fluid = v_fluid_gen(t) / max_velocity

            total_v = np.hstack((v_solid, v_fluid))

            non_dim_t = t / time_period
            non_dim_t -= np.floor(non_dim_t)

            c = cmap(non_dim_t)
            ax.plot(
                total_v,
                total_y,
                color=c,
                linewidth=3,
            )

            if write_flag:
                # Write the data
                np.savetxt(
                    os.path.join(
                        file_path,
                        "{0}_velocity_at_{1:e}.csv".format(str(solution), non_dim_t),
                    ),
                    np.c_[total_y, total_v],
                    delimiter=",",
                )

        # Write the figure too
        if write_flag:
            # Write the image with all the lines
            fig.savefig(os.path.join(file_path, "velocities.pdf"), dpi=100)
        else:
            plt.show()

    return solution


def plot_from_yaml(
    solution_type: Type[ProblemSolution],
    plot_times: List[float],
    param_file_name: FilePath = "params.yaml",
    rel_file_path: str = "data",
    animate_flag: bool = False,
    write_flag: bool = True,
    grid_generator: Callable[[float, float], Grids] = partial(
        generate_regular_grid, 512
    ),
) -> ProblemSolution:
    file_path = os.path.join(os.path.dirname(param_file_name), rel_file_path)
    os.makedirs(file_path, exist_ok=True)
    return plot_solution(
        solution_type(_internal_load(param_file_name)),
        plot_times,
        file_path,
        animate_flag,
        write_flag,
        grid_generator,
    )


def run_and_plot(
    solution: ProblemSolution,
    final_time: float,
    plot_times: List[float] = None,
    file_path: FilePath = "data",
    animate_flag: bool = False,
    write_flag: bool = True,
    grid_generator: Callable[[float, float], Grids] = partial(
        generate_regular_grid, 512
    ),
) -> ProblemSolution:
    # file_path = os.path.join(os.getcwd(), rel_file_path)
    os.makedirs(file_path, exist_ok=True)
    solution = run(solution, final_time, file_path)

    if plot_times is None:
        plot_times = np.linspace(0.0, 1.0, 21) * solution.time_period

    solution = plot_solution(
        solution, plot_times, file_path, animate_flag, write_flag, grid_generator
    )

    return solution


def run_and_plot_from_yaml(
    solution_type: Type[ProblemSolution],
    final_time: float,
    plot_times: List[float] = None,
    param_file_name: FilePath = "params.yaml",
    rel_file_path: FilePath = "data",
    animate_flag: bool = False,
    write_flag: bool = True,
    grid_generator: Callable[[float, float], Grids] = partial(
        generate_regular_grid, 512
    ),
) -> ProblemSolution:
    return run_and_plot(
        solution_type(_internal_load(param_file_name)),
        final_time,
        plot_times,
        os.path.join(os.getcwd(), rel_file_path),
        animate_flag,
        write_flag,
        grid_generator,
    )
