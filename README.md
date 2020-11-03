Shearing Parallel Slabs Benchmark
&middot;
[![Build Status](https://travis-ci.com/tp5uiuc/parallel_slab.svg?token=ZZkcxuTHm9peGgncAAKa&branch=master)](https://travis-ci.com/tp5uiuc/parallel_slab)
[![codecov](https://codecov.io/gh/tp5uiuc/parallel_slab/branch/master/graph/badge.svg?token=QWZOGBPC83)](https://codecov.io/gh/tp5uiuc/parallel_slab)
[![license](https://img.shields.io/badge/license-MIT-green)](https://mit-license.org/)
[![pyversion](https://img.shields.io/pypi/pyversions/Django)](https://www.python.org/)
=====

Shear flow benchmark for testing elastic solid--fluid coupling algorithms

## Installation
Please clone this repository and use one of the [examples](examples) for running a single simulation or 
a parameter sweep of simulations. For more information see [Usage and examples](#usage-and-examples)

## Physical setup
The two-dimensional setup employs an elastic solid layer sandwiched between two fluid layers, in turn confined by two long planar walls, 
whose horizontal oscillations drive a characteristic system response. This is shown in the figure below. The fluid is
Newtonian while the solid is made of either a neo-Hookean material (with stresses proportional linearly to deformations)
or a generalized Mooney--Rivlin material (with stresses varying non-linearly in response to deformations). This setting 
admits a time periodic, one-dimensional analytical solution for neo-Hookean materials and a semi-analytical solution 
(based on a sharp interface pseudo-spectral method) for a generalized Mooney--Rivlin solid.

![setup](./docs/assets/setup.png)

Overall, this problem entails multiple interfaces, phases and boundary conditions interacting dynamically, and serves as
 a challenging benchmark to validate the long time behaviour, stability and accuracy of FSI solvers.

## Usage and examples
TODO

![linear](./docs/assets/panel_linear_velocities.png)

![nonlinear](./docs/assets/panel_nonlinear_velocities.png)

## Numerical algorithms
Details on the algorithms employed in the two examples shown above can be found in the following technical paper.
If you are employing this benchmark, please cite the work below.

<a id="1">[1]</a> 
TODO : CITE OUR PAPER HERE


## Running tests
Check [Installation](#installation) to see how to install the package. Once that is done, install the test requirements
using 
`pip3 install -r tests/requirements.txt` 
from the repo directory. This package uses `py.test` for running unit and integration tests. 
To run them, simply do 
`python3 -m pytest` 
from the repository directory.