# <img src="https://enzyme.mit.edu/logo.svg" width="75" align=left> The Enzyme High-Performance Automatic Differentiator of LLVM

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://enzyme.mit.edu/julia/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://enzyme.mit.edu/julia/dev)
[![Build Status](https://github.com/EnzymeAD/Enzyme.jl/workflows/CI/badge.svg)](https://github.com/EnzymeAD/Enzyme.jl/actions)
[![Coverage](https://codecov.io/gh/EnzymeAD/Enzyme.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/EnzymeAD/Enzyme.jl)

This is a package containing the Julia bindings for [Enzyme](https://github.com/EnzymeAD/enzyme). This is very much a work in progress and bug reports/discussion is greatly appreciated!

Enzyme is a plugin that performs automatic differentiation (AD) of statically analyzable LLVM. It is highly-efficient and its ability perform AD on optimized code allows Enzyme to meet or exceed the performance of state-of-the-art AD tools.

Enzyme.jl can be installed in the usual way Julia packages are installed
```
] add Enzyme
```

Enzyme.jl can be used by calling `autodiff` on a function to be differentiated as shown below:

```julia
using Enzyme, Test

f1(x) = x*x
# Returns a tuple of active returns, which in this case is simply (2.0,)
@test first(autodiff(Reverse, f1, Active(1.0))[1]) ≈ 2.0
```

For details, see the [package documentation](https://enzyme.mit.edu/julia).

More information on installing and using Enzyme directly (not through Julia) can be found on our website: [https://enzyme.mit.edu](https://enzyme.mit.edu).

To get involved or if you have questions, please join our [mailing list](https://groups.google.com/d/forum/enzyme-dev).

If using this code in an academic setting, please cite the following two papers (first for Enzyme as a whole, then for GPU+optimizations):
```
@inproceedings{NEURIPS2020_9332c513,
 author = {Moses, William and Churavy, Valentin},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {H. Larochelle and M. Ranzato and R. Hadsell and M. F. Balcan and H. Lin},
 pages = {12472--12485},
 publisher = {Curran Associates, Inc.},
 title = {Instead of Rewriting Foreign Code for Machine Learning, Automatically Synthesize Fast Gradients},
 url = {https://proceedings.neurips.cc/paper/2020/file/9332c513ef44b682e9347822c2e457ac-Paper.pdf},
 volume = {33},
 year = {2020}
}
@inproceedings{10.1145/3458817.3476165,
author = {Moses, William S. and Churavy, Valentin and Paehler, Ludger and H\"{u}ckelheim, Jan and Narayanan, Sri Hari Krishna and Schanen, Michel and Doerfert, Johannes},
title = {Reverse-Mode Automatic Differentiation and Optimization of GPU Kernels via Enzyme},
year = {2021},
isbn = {9781450384421},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3458817.3476165},
doi = {10.1145/3458817.3476165},
booktitle = {Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis},
articleno = {61},
numpages = {16},
keywords = {CUDA, LLVM, ROCm, HPC, AD, GPU, automatic differentiation},
location = {St. Louis, Missouri},
series = {SC '21}
}
```
