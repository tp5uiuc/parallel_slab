from parallel_slab import NeoHookeanSolution, GeneralizedMooneyRivlinSolution
from parallel_slab import run_and_plot_from_yaml
import os
import numpy as np


# # def run_from_yaml(cls: Type[ProblemSolution] = NeoHookeanSolution):
# #     param_file = os.path.join(os.getcwd(), "params.yaml")
# #     # plot_times = [10.0, 11.0, 12.0]
# #     plot_times = np.linspace(10.0, 12.0, 20)
# #     return plot_from_yaml(cls, plot_times=plot_times, param_file_name=param_file)
# #
# #
# def lmr_from_yaml():
#     from parallel_slab import _internal_load
#     param_file = os.path.join(os.getcwd(), "params.yaml")
#     params = _internal_load(param_file)
#     solution = GeneralizedMooneyRivlinSolution(params)
#     return solution
#
#
if __name__ == "__main__":
    # solution = lmr_from_yaml()
    # solution.run_till(20.0)
    # solution.save_data(os.path.join(os.getcwd(), "data"))
    # solution.load_data(os.path.join(os.getcwd(), "data"))

    # 2b3322d55d7ed2ed6a331483b9fcd399
    # 2b3322d55d7ed2ed6a331483b9fcd399
    # a = run_from_yaml(NeoHookeanSolution)

    # from parallel_slab import internal_load
    # param_file = os.path.join(os.getcwd(), "params.yaml")
    # params = internal_load(param_file)
    # solution = NeoHookeanSolution(params)
    # plot_times = np.linspace(10.0, 12.0, 20)
    plot_times = np.array([0.75, 1.0]) * 2.0
    run_and_plot_from_yaml(NeoHookeanSolution, 20.0, plot_times=plot_times)
