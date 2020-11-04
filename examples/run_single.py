from parallel_slab import NeoHookeanSolution, GeneralizedMooneyRivlinSolution
from parallel_slab import run_and_plot_from_yaml
import numpy as np

if __name__ == "__main__":
    ###############################################
    ## 1. Load from YAML
    ###############################################
    run_and_plot_from_yaml(
        NeoHookeanSolution,  # the type of solid material
        final_time=20.0,  # final time of simulation, till periodic steady state
        plot_times=np.linspace(
            10.0, 12.0, 20
        ),  # time at which you want to plot the solutions
        param_file_name="params.yaml",  # name of the parameters YAML file
        rel_file_path="data",  # folder to store simulation artefacts
        write_flag=False,  # write data files (as csv) and images (as pdf) at plot_times
        animate_flag=True,
    )  # animate a movie sampled at plot_time and dump it as mp4

    ###############################################
    ## 2. Load from YAML, default parameters
    ###############################################
    # Many of the parameters are defaulted to the values shown above
    # so that you only need to bother with the first two parameters
    run_and_plot_from_yaml(
        GeneralizedMooneyRivlinSolution, final_time=2.0  # the type of solid material
    )  # final time of simulation, till periodic steady state

    ###############################################
    ## 3. Output Grid
    ###############################################
    # Finally we can control the details of the output grid
    # using the grid_generator keyword parameter which accepts a
    # callable object with two parameters (length of solid and fluid zones from YAML file)
    # and outputs a tuple of solid and fluid grids
    # This is for easily obtaining the solution at your simulation grid points
    from parallel_slab.driver import generate_regular_grid
    from functools import partial

    # generate_regular_grid generates two grids with total n_total_points
    grid_generator = partial(generate_regular_grid, 128)

    run_and_plot_from_yaml(
        NeoHookeanSolution,  # the type of solid material
        final_time=20.0,  # final time of simulation, till periodic steady state
        grid_generator=grid_generator,
    )
