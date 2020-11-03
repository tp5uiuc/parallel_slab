#!/usr/bin/env python3

__doc__ = """Solution for neo-Hookean solids"""

import numpy as np
from scipy.interpolate import interp1d
import os
from typing import Dict, TypeVar, Callable, Tuple, List, Any
from .utils import dict_hash

SolutionGenerator = Callable[[float], np.ndarray]


class SolutionBase:
    """

    Can do ABC but unnecessary problems
    """

    def __init__(self, params: Dict[str, float]):
        self.interface_velocity = None
        self.solid_velocity_k = None
        self.fluid_velocity_k = None

        self.k = np.arange(1, params["n_modes"], dtype=np.float64)

        ### For documentation of these parameters, see the main driver file
        self.L_s = params["L_s"]
        self.L_f = params["L_f"]
        self.rho_f = params["rho_f"]
        self.mu_f = params["mu_f"]
        self.nu_f = self.mu_f / self.rho_f

        # The main parameters determining the solid behavior
        self.rho_s = params["rho_s"]
        self.mu_s = params["mu_s"]
        self.nu_s = self.mu_s / self.rho_s

        omega = params["omega"]
        self.omega = omega
        self.time_period = 2.0 * np.pi / omega

        self.V_wall = params["V_wall"]
        self.wall_velocity = lambda time_v: np.imag(
            self.V_wall * SolutionBase._internal_activation(omega, time_v)
        )

        self.data_loaded = False

    @staticmethod
    def _internal_activation(omega, time_v):
        return np.exp(1j * omega * time_v)

    def ready(self):
        return self.data_loaded

    def _solid_velocity(self, solid_grid) -> SolutionGenerator:
        """
        Generate solid velocity

        Parameters
        ----------
        solid_grid :  Simulation grid for the solid, a np array

        Returns
        -------
        A functor that can generate physical solid velocity

        """

        def __solution(time_v: float) -> np.ndarray:
            """
            Returns the physical solid velocity given temporal driving velocities

            Parameters
            ----------
            time_v :  The time at which the solution is requested

            Returns
            -------
            Solid velocities sampled at the grid points
            """

            sin_array = np.sin(
                np.pi * self.k.reshape(1, -1) / self.L_s * solid_grid.reshape(-1, 1)
            )
            # Can use DST to acclerate multiplication, but skip
            solid_harmonic_velocity = sin_array @ self.solid_velocity_k(time_v)
            # No np.real here! affects the phase
            return (
                self.interface_velocity(time_v) * solid_grid / self.L_s
            ) + solid_harmonic_velocity

        return __solution

    def _fluid_velocity(self, fluid_grid: np.ndarray) -> SolutionGenerator:
        """
        Returns the physical fluid velocity given temporal driving velocities

        Parameters
        ----------
        fluid_grid :  Simulation grid for the solid, a np array

        Returns
        -------
        A functor that can generate physical fluid velocity at grid points
        """

        def __solution(time_v: float) -> np.ndarray:
            """
            Returns the physical fluid velocity given temporal driving velocities

            Parameters
            ----------
            time_v :  The time at which the solution is requested

            Returns
            -------
            Fluid velocities sampled at the grid points
            """

            sin_array = np.sin(
                np.pi * self.k.reshape(1, -1) / self.L_f * fluid_grid.reshape(-1, 1)
            )
            # Can use DST to acclerate multiplication, but skip
            fluid_harmonic_velocity = sin_array @ self.fluid_velocity_k(time_v)
            interface_velocity = self.interface_velocity(time_v)
            # No np.real here! affects the phase
            return (
                interface_velocity
                + fluid_grid
                * (self.wall_velocity(time_v) - interface_velocity)
                / self.L_f
                + fluid_harmonic_velocity
            )

        return __solution

    def get_velocities(
        self, solid_grid, fluid_grid
    ) -> Tuple[SolutionGenerator, SolutionGenerator]:
        """
        Gets the velocity data at the requested time

        Parameters
        ----------
        solid_grid: Grid points across the solid domain
        fluid_grid: Grid points across the fluid domain

        Returns
        -------
        (2, ) tuple containing generators for solid and fluid velocities

        """
        # Check to see if within range
        assert np.all(np.logical_and(solid_grid >= 0.0, solid_grid <= self.L_s))
        assert np.all(np.logical_and(fluid_grid >= 0.0, fluid_grid <= self.L_f))

        return (self._solid_velocity(solid_grid), self._fluid_velocity(fluid_grid))


class NeoHookeanSolution(SolutionBase):
    """
    Analytical solution for a Neo-Hookean solid slab
    updated solution according to the latest draft of TR2
    """

    def __repr__(self):
        return "NeoHookeanSolution"

    def __init__(self, params: Dict[str, float]):
        """

        Parameters
        ----------
        params : Simulation parameters
        """
        SolutionBase.__init__(self, params)

        c_1 = params["c_1"]

        if params.get("c_3", 0.0) > 0.0:
            from warnings import warn

            warn(
                "Intended solution is for a Neo-Hookean solid, but we found parameters corresponding to"
                "a generalized Mooney Rivlin solid!",
                UserWarning,
            )

        k = self.k
        nu_f = self.nu_f
        nu_s = self.nu_s
        rho_s = self.rho_s
        L_f = self.L_f
        L_s = self.L_s
        mu_f = self.mu_f
        mu_s = self.mu_s
        omega = self.omega
        V_wall = self.V_wall

        # Internal parameter setup
        alpha_k = np.zeros(k.shape, dtype=np.complex)

        alpha_k = (1j * 2.0 * omega) / ((1j * omega) + (nu_f * (np.pi * k / L_f) ** 2))
        beta_k = (1j * 2.0 * omega) / (
            (1j * omega)
            + (np.pi * k / L_s) ** 2 * (2 * c_1 / (1j * omega * rho_s) + nu_s)
        )
        alpha_sum = np.sum(alpha_k)
        alternating_alpha_sum = np.sum(((-1) ** k) * alpha_k)
        beta_sum = np.sum(beta_k)

        interface_velocity_hat_denom = (mu_f / L_f) * (1 + alpha_sum) + (
            mu_s / L_s + 2 * c_1 / (1j * omega * L_s)
        ) * (1 + beta_sum)
        interface_velocity_hat = (
            (mu_f * V_wall / L_f)
            * (1 + alternating_alpha_sum)
            / interface_velocity_hat_denom
        )

        solid_displacement_hat_k = -(
            1j * ((-1) ** k) * interface_velocity_hat * beta_k / np.pi / omega / k
        )
        fluid_velocity_hat_k = (
            (((-1) ** k) * V_wall - interface_velocity_hat) * alpha_k / np.pi / k
        )
        solid_velocity_hat_k = 1j * omega * solid_displacement_hat_k

        self.interface_velocity = lambda time_v: np.imag(
            interface_velocity_hat * SolutionBase._internal_activation(omega, time_v)
        )
        self.solid_displacement_k = lambda time_v: np.imag(
            solid_displacement_hat_k * SolutionBase._internal_activation(omega, time_v)
        )
        self.solid_velocity_k = lambda time_v: np.imag(
            solid_velocity_hat_k * SolutionBase._internal_activation(omega, time_v)
        )
        self.fluid_velocity_k = lambda time_v: np.imag(
            fluid_velocity_hat_k * SolutionBase._internal_activation(omega, time_v)
        )

        # finally indicate always ready
        self.data_loaded = True

    def save_data(self, file_path: str) -> None:
        """
        Saves data for post-processing

        Parameters
        ----------
        file_path : Path (Folder) to save custom data in

        Returns
        -------

        """
        # There's nothing to save in this case
        pass

    def load_data(self, file_path: str) -> None:
        """
        Loads data from disk to render

        Parameters
        ----------
        file_path : Path (Folder) to load custom data from

        Returns
        -------

        """
        # There's nothing to load in this case
        self.data_loaded = True

    def run_till(self, final_time: float) -> None:
        # There's nothing to run in this case
        pass


class GeneralizedMooneyRivlinSolution(SolutionBase):
    """
    Analytical solution for a generalized Mooney-Rivlin solid slab
    updated solution according to the latest draft of TR2
    """

    def __repr__(self):
        return "GeneralizedMooneyRivlinSolution"

    def __init__(self, params: Dict[str, float]):
        """
        Parameters
        ----------
        params : Simulation parameters
        """
        SolutionBase.__init__(self, params)

        self.c_1 = params["c_1"]
        self.c_3 = params["c_3"]
        self.n_samples = params.get(
            "n_samples", 200
        )  # Collect n_sample samples in last Time period

        self.get_file_id = lambda path: os.path.join(path, dict_hash(params) + ".pkl")

        # Convergence metrics, lumped into data
        self.time_history = None
        self.interface_displacement_history = None
        self.interface_velocity_history = None

        # Data metrics
        from collections import defaultdict

        # Has the following fields
        # 1. 'modes' :
        #   - 'modes' : gives modes
        # 2. 'history' : gives history
        #   - 'time : time history
        #   - 'interface_displacement' : interface displacement over the course of simulation
        #   - 'interface_velocity' : interface velocity over the course of simulation
        # 3. iter : a number between [0, n_steps_per_period) that stores
        #   - 'fluid' : fluid mode velocities
        #   - 'solid' : solid mode velocities
        #   - 'inter' : interface velocities
        self.recorded_data = defaultdict(dict)
        self.recorded_data["modes"]["modes"] = self.k.copy()

        self.delta_t = 2.5e-4 * self.time_period
        self.n_steps_in_period = int(self.time_period / self.delta_t)

        self.iter2ndtime = lambda itera: itera / self.n_steps_in_period

    def save_data(self, file_path: str) -> None:
        """
        Saves data for post-processing

        Parameters
        ----------
        file_path : Path (Folder) to save custom data in

        Returns
        -------

        """
        from pickle import dump

        dump(self.recorded_data, open(self.get_file_id(file_path), "wb"))

    def load_data(self, file_path) -> None:
        """
        Loads data from disk to render

        Parameters
        ----------
        file_path : Path (Folder) to load custom data from

        Returns
        -------

        """
        from pickle import load

        self.recorded_data = load(open(self.get_file_id(file_path), "rb"))

        from numpy.testing import assert_allclose

        assert_allclose(self.k, self.recorded_data["modes"]["modes"])

        # Unpack to specific members for ease of use
        self.time_history = self.recorded_data["history"]["time"]
        self.interface_displacement_history = self.recorded_data["history"][
            "interface_displacement"
        ]
        self.interface_velocity_history = self.recorded_data["history"][
            "interface_velocity"
        ]

        self._prepare()

    def _prepare(self):
        ndtime = []
        v_fluid = []
        v_solid = []
        u_solid = []
        v_inter = []

        for k, v in self.recorded_data.items():
            # if k is an iteration
            if isinstance(k, int):
                ndtime.append(self.iter2ndtime(k))
                v_fluid.append(v["fluid"].copy())
                v_solid.append(v["solid"].copy())
                u_solid.append(v["solid_d"].copy())
                v_inter.append(v["inter"])

        # (t,) array from [0, T). Does not include the point at T
        # To make sure that values between T-delt and T can be properly interpolated we need to manually
        # pad the last position ty append
        def append_zeroth_position(a_list: List[Any]):
            return np.array(a_list + [a_list[0]])

        # manually append the last non-dimensional time
        ndtime = np.array(ndtime + [1.0])

        # (k, t) arrays
        v_fluid = append_zeroth_position(v_fluid).T
        v_solid = append_zeroth_position(v_solid).T
        u_solid = append_zeroth_position(u_solid).T
        v_inter = append_zeroth_position(v_inter).T

        def time2nd(time_v):
            temp = time_v / self.time_period
            temp -= np.floor(temp)
            return temp

        interface_velocity_gen = interp1d(ndtime, v_inter)
        self.interface_velocity = lambda time_v: interface_velocity_gen(time2nd(time_v))

        solid_velocity_k_gen = interp1d(ndtime, v_solid)
        self.solid_velocity_k = lambda time_v: solid_velocity_k_gen(time2nd(time_v))

        solid_displacement_k_gen = interp1d(ndtime, u_solid)
        self.solid_displacement_k = lambda time_v: solid_displacement_k_gen(
            time2nd(time_v)
        )

        fluid_velocity_k_gen = interp1d(ndtime, v_fluid)
        self.fluid_velocity_k = lambda time_v: fluid_velocity_k_gen(time2nd(time_v))

        self.data_loaded = True

    def run_till(self, final_time: float) -> None:
        # There's nothing to run
        delta_t = self.delta_t
        n_steps_in_period = self.n_steps_in_period

        # Keys
        OLD = 0
        CURR = 1
        NEXT = 2

        CACHE_SIZE = 3
        # Setup initial cache of values
        interface_displacement = np.zeros((CACHE_SIZE,))
        interface_velocity = np.zeros((CACHE_SIZE,))

        solid_velocity = np.zeros((CACHE_SIZE, self.k.size))
        solid_displacement = np.zeros((CACHE_SIZE, self.k.size))
        fluid_velocity = np.zeros((CACHE_SIZE, self.k.size))
        non_linear_solid_stress = np.zeros((CACHE_SIZE, self.k.size))
        zero_mode_non_linear_solid_stress = np.zeros((CACHE_SIZE,))

        # Time stepper details for discretizing the du_dt term in the
        # interface matching condition
        # du_dt @ (n+1) = [A * u(n+1) + B * u(n) + C * u(n - 1)] / dt

        # # Fill in the solid velocity too, using midpoint rule (Nyström)
        # # expanded around t^(n) (less accurate, but still good enough)
        # A = 0.5
        # B = 0.0
        # C = -0.5

        # Fill in the solid velocity too, using BDF2, expanded around t^(n+1)
        # Second order accurate
        A = 1.5
        B = -2.0
        C = 0.5

        zeta_solid_k = (
            1.0 + 0.5 * self.nu_s * delta_t * (np.pi * self.k / self.L_s) ** 2
        )
        zeta_fluid_k = (
            1.0 + 0.5 * self.nu_f * delta_t * (np.pi * self.k / self.L_f) ** 2
        )
        gamma_k = (
            2.0 * self.c_1 * (delta_t * np.pi * self.k / self.L_s) ** 2 / self.rho_s
            - 2.0
        )

        inv_zeta_fluid_sum = np.sum(1.0 / zeta_fluid_k)
        inv_zeta_solid_sum = np.sum(1.0 / zeta_solid_k)

        interface_velocity_denominator = (
            (self.mu_f / self.L_f) * (1.0 + 2.0 * inv_zeta_fluid_sum)
            + (self.c_1 * delta_t / self.L_s) * (1.0 + 2.0 * inv_zeta_solid_sum)
            + (self.mu_s / self.L_s) * (1.0 + A * inv_zeta_solid_sum)
            # + (mu_s / self.L_s) * (1.0 + 0.5 * inv_zeta_solid_sum)
            # + (mu_s / L_s) * (1.0 + 1.5 * inv_zeta_solid_sum)
        )

        l_modes = self.k.copy()
        j_modes = np.hstack((0, l_modes))  # one more

        # To put in loop
        sim_time = 0.0
        n_steps = int(final_time / delta_t)

        data_collection_frequency = n_steps_in_period // self.n_samples
        # Start off exactly here so that the 0th iteration gets collected
        data_counter = data_collection_frequency

        time_tracker = []
        interface_displacement_tracker = []
        interface_velocity_tracker = []

        from tqdm import tqdm

        for iteration in tqdm(range(n_steps)):
            # includes 0 by default
            if not iteration % data_collection_frequency:
                time_tracker.append(sim_time)
                interface_displacement_tracker.append(interface_displacement[CURR])
                interface_velocity_tracker.append(interface_velocity[CURR])

            # if in last period, collect data in a dictionary indexed by phase
            if iteration >= (n_steps - n_steps_in_period):
                #  start collecting data into the folder every certain steps
                if data_counter >= data_collection_frequency:
                    data_counter = 0
                    # self.time_data.append(sim_time)
                    # total_dumping_steps = n_steps - n_steps_in_period
                    curr_iter = iteration % n_steps_in_period

                    self.recorded_data[curr_iter]["fluid"] = fluid_velocity[CURR].copy()
                    self.recorded_data[curr_iter]["solid"] = solid_velocity[CURR].copy()
                    self.recorded_data[curr_iter]["solid_d"] = solid_displacement[
                        CURR
                    ].copy()
                    # Floating point number
                    self.recorded_data[curr_iter]["inter"] = interface_velocity[CURR]

            #####################################
            # Interface velocity update
            #####################################

            interface_velocity[NEXT] = (
                self.mu_f / self.L_f * self.wall_velocity(sim_time + delta_t)
            )
            interface_velocity[NEXT] += (
                -(
                    self.c_1
                    * (
                        2.0 * interface_displacement[CURR]
                        + delta_t * interface_velocity[CURR]
                    )
                )
                / self.L_s
            )

            interface_velocity[NEXT] += (
                -2.0 * zero_mode_non_linear_solid_stress[CURR]
                + zero_mode_non_linear_solid_stress[OLD]
            )

            # Fluid contribution to interface
            delta_wall_velocity = self.wall_velocity(
                sim_time + delta_t
            ) - self.wall_velocity(sim_time)
            fluid_interface_velocity_contribution = np.pi * self.k * fluid_velocity[
                CURR
            ] * (2.0 - zeta_fluid_k) + 2.0 * (
                interface_velocity[CURR] + ((-1) ** self.k) * delta_wall_velocity
            )
            fluid_interface_velocity_contribution *= self.mu_f / self.L_f / zeta_fluid_k

            interface_velocity[NEXT] += np.sum(fluid_interface_velocity_contribution)

            # Solid contribution to interface
            # 1. Purely linear elastic contributions
            solid_interface_velocity_contribution = (
                2.0
                * self.c_1
                / zeta_solid_k
                / self.L_s
                * (
                    delta_t * interface_velocity[OLD]
                    + (
                        ((-1) ** self.k)
                        * np.pi
                        * self.k
                        * (
                            gamma_k * solid_displacement[CURR]
                            + (2.0 - zeta_solid_k) * solid_displacement[OLD]
                        )
                    )
                )
            )

            # 2. Solid viscosity contributions
            solid_interface_velocity_contribution += (
                self.mu_s
                / (zeta_solid_k * self.L_s * delta_t)
                * (
                    A * delta_t * interface_velocity[OLD]
                    + (
                        ((-1) ** self.k)
                        * np.pi
                        * self.k
                        * (
                            (A * gamma_k - B * zeta_solid_k) * solid_displacement[CURR]
                            + (A * (2.0 - zeta_solid_k) - C * zeta_solid_k)
                            * solid_displacement[OLD]
                        )
                    )
                )
            )

            # 3. nonlinear elasticity contributions
            solid_interface_velocity_contribution += ((-1) ** self.k) * (
                (
                    (gamma_k + 2.0) / zeta_solid_k
                    + A
                    * (self.nu_s * delta_t * (np.pi * self.k / self.L_s) ** 2)
                    / zeta_solid_k
                    - 2.0
                )
                * non_linear_solid_stress[CURR]
                + non_linear_solid_stress[OLD]
            )

            interface_velocity[NEXT] += np.sum(solid_interface_velocity_contribution)

            interface_velocity[NEXT] /= interface_velocity_denominator

            #####################################
            # Interface Displacement update
            #####################################
            interface_displacement[NEXT] = interface_displacement[
                CURR
            ] + 0.5 * delta_t * (interface_velocity[NEXT] + interface_velocity[CURR])

            #####################################
            # Fluid Velocity update
            #####################################
            fluid_velocity_numerator = (2.0 - zeta_fluid_k) * fluid_velocity[CURR]
            fluid_velocity_numerator += -(2.0 / np.pi / self.k) * (
                interface_velocity[NEXT]
                - interface_velocity[CURR]
                - (((-1) ** self.k) * delta_wall_velocity)
            )
            fluid_velocity[NEXT] = fluid_velocity_numerator / zeta_fluid_k

            #####################################
            # Solid displacement update
            #####################################
            # E_vf_k term
            solid_displacement[NEXT] = (
                ((-1) ** self.k)
                * (interface_velocity[NEXT] - interface_velocity[OLD])
                * delta_t
                / (np.pi * self.k)
            )
            solid_displacement[NEXT] += 2.0 * solid_displacement[CURR]
            solid_displacement[NEXT] += -(2.0 - zeta_solid_k) * solid_displacement[OLD]
            solid_displacement[NEXT] += -(delta_t ** 2) * (
                2.0
                * self.c_1
                * (np.pi * self.k / self.L_s) ** 2
                / self.rho_s
                * solid_displacement[CURR]
                + np.pi * self.k / self.rho_s / self.L_s * non_linear_solid_stress[CURR]
            )
            solid_displacement[NEXT] /= zeta_solid_k

            #####################################
            # Solid non-linear stress calculation
            #####################################

            # construct the matrix for different js
            # Put l along columns, j along rows and sum over all columns
            # The ( 1 + self.k[-1] ) is supposed to be just K
            K = 1 + self.k[-1]
            pseudo_spectral_contribution = (
                (np.pi * l_modes.reshape(1, -1) / self.L_s)
                * solid_displacement[NEXT].reshape(1, -1)
                * np.cos(
                    np.pi * l_modes.reshape(1, -1) * (j_modes.reshape(-1, 1) + 0.5) / K
                )
            )
            # sum over all l wavemodes
            pseudo_spectral_contribution_sum = np.sum(
                pseudo_spectral_contribution, axis=-1
            )
            # print(interface_displacement[NEXT])
            du_dy_cubed = (
                (interface_displacement[NEXT] / self.L_s)
                + pseudo_spectral_contribution_sum
            ) ** 3

            zero_mode_non_linear_solid_stress[NEXT] = (
                4.0 * self.c_3 / K * np.sum(du_dy_cubed)
            )

            cos_terms = np.cos(
                np.pi * self.k.reshape(-1, 1) * (j_modes.reshape(1, -1) + 0.5) / K
            )
            non_linear_solid_stress[NEXT] = (
                8.0
                * self.c_3
                / K
                * np.sum(du_dy_cubed.reshape(1, -1) * cos_terms, axis=-1)
            )

            #####################################
            # Solid velocity update
            #####################################
            # Fill in the solid velocity too, using BDF2, expanded around t^(n+1)
            solid_velocity[NEXT] = (
                A * solid_displacement[NEXT]
                + B * solid_displacement[CURR]
                + C * solid_displacement[OLD]
            ) / delta_t

            # Finally update the old gods and the new

            for its in [
                interface_displacement,
                interface_velocity,
                solid_velocity,
                solid_displacement,
                fluid_velocity,
                non_linear_solid_stress,
                zero_mode_non_linear_solid_stress,
            ]:
                its[OLD] = its[CURR].copy()
                its[CURR] = its[NEXT].copy()

            # Update time and iterations finally
            sim_time += delta_t
            # iteration += 1
            data_counter += 1

            time_tracker.append(sim_time)
            interface_displacement_tracker.append(interface_displacement[CURR])
            interface_velocity_tracker.append(interface_velocity[CURR])

        self.recorded_data["history"]["time"] = np.array(time_tracker)
        self.recorded_data["history"]["interface_displacement"] = np.array(
            interface_displacement_tracker
        )
        self.recorded_data["history"]["interface_velocity"] = np.array(
            interface_velocity_tracker
        )

        self._prepare()


ProblemSolution = TypeVar(
    "ProblemSolution", NeoHookeanSolution, GeneralizedMooneyRivlinSolution
)
