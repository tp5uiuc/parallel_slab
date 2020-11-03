Shearing Parallel Slabs Benchmark
&middot;
[![Build Status](https://travis-ci.com/tp5uiuc/parallel_slab.svg?token=ZZkcxuTHm9peGgncAAKa&branch=master)](https://travis-ci.com/tp5uiuc/parallel_slab)
[![codecov](https://codecov.io/gh/tp5uiuc/parallel_slab/branch/master/graph/badge.svg?token=QWZOGBPC83)](https://codecov.io/gh/tp5uiuc/parallel_slab)
[![license](https://img.shields.io/badge/license-MIT-green)](https://mit-license.org/)
[![pyversion](https://img.shields.io/pypi/pyversions/Django)](https://www.python.org/)
=====

Shear benchmark for an elastic solid--fluid layers sandwiched 
between two oscillatory moving walls

## Installation
Please clone this repository and use one of the [examples](examples) for running a single simulation or 
a parameter sweep of simulations. For more information see [Usage and examples](#usage-and-examples)

## Physical setup
TODO

## Usage and examples
TODO

## Numerical algorithms
Details on the employed algorithms in the two examples shown above can be found in the following technical paper.

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
