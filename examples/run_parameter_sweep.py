import os

import numpy as np
import psweep as ps

from parallel_slab import (
    GeneralizedMooneyRivlinSolution,
    NeoHookeanSolution,
    run_and_plot,
)


def evaluate(pset):
    solution = NeoHookeanSolution(pset)
    final_time = 20.0
    file_path = os.path.join(pset["_calc_dir"], pset["_pset_id"])
    time_period = 2.0 * np.pi / pset["omega"]
    plot_times = np.linspace(0.0, 1.0, 21) * time_period
    run_and_plot(
        solution=solution,
        final_time=final_time,
        plot_times=plot_times,
        file_path=file_path,
    )
    return pset


def run_sweep():
    params = [None for _ in range(11)]
    params[0] = ps.plist("L_s", [0.2])
    params[1] = ps.plist("n_modes", [64])
    params[2] = ps.plist("L_f", [0.2])
    params[3] = ps.plist("rho_f", [1.0])
    params[4] = ps.plist("mu_f", [0.02])
    params[5] = ps.plist("rho_s", [1.0])
    params[6] = ps.plist("mu_s", [0.002])
    params[7] = ps.plist("c_1", [0.01, 0.02, 0.03, 0.04])
    params[8] = ps.plist("c_3", [0.04])
    params[9] = ps.plist("V_wall", [0.4])
    params[10] = ps.plist("omega", [np.pi])

    params = ps.pgrid(*params)

    df = ps.run(
        evaluate,
        params,
        poolsize=4,
        backup_calc_dir=False,
        backup_script=__file__,
        simulate=False,
    )

    return df


def load_database():
    results_fn_base = "results.pk"
    calc_dir = "calc"
    pdf = os.path.join(calc_dir, results_fn_base)
    df = ps.df_read(pdf)
    return df


if __name__ == "__main__":
    df = run_sweep()
    # df = load_database()
