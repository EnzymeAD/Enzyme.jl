# <img src="https://enzyme.mit.edu/logo.svg" width="75" align=left> The Enzyme High-Performance Automatic Differentiator of LLVM

[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://enzyme.mit.edu/julia)
[![Build Status](https://github.com/wsmoses/Enzyme.jl/workflows/CI/badge.svg)](https://github.com/wsmoses/Enzyme.jl/actions)
[![Coverage](https://codecov.io/gh/wsmoses/Enzyme.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/wsmoses/Enzyme.jl)

This is a package containing the Julia bindings for [Enzyme](https://github.com/wsmoses/enzyme). This is very much a work in progress and bug reports/discussion is greatly appreciated!

Enzyme is a plugin that performs automatic differentiation (AD) of statically analyzable LLVM. It is highly-efficient and its ability perform AD on optimized code allows Enzyme to meet or exceed the performance of state-of-the-art AD tools.

Enzyme.jl can be installed in the usual way Julia packages are installed
```
] add Enzyme
```

Enzyme.jl can be used by calling `autodiff` on a function to be differentiated as shown below:

```julia
using Enzyme, Test

f1(x) = x*x
@test autodiff(f1, Active(1.0)) == (2.0,)
```

For details, see the [package documentation](https://enzyme.mit.edu/julia).

More information on installing and using Enzyme directly (not through Julia) can be found on our website: [https://enzyme.mit.edu](https://enzyme.mit.edu).

To get involved or if you have questions, please join our [mailing list](https://groups.google.com/d/forum/enzyme-dev).

If using this code in an academic setting, please cite the following paper to appear in NeurIPS 2020

```
@incollection{enzymeNeurips,
title = {Instead of Rewriting Foreign Code for Machine Learning, Automatically Synthesize Fast Gradients},
author = {Moses, William S. and Churavy, Valentin},
booktitle = {Advances in Neural Information Processing Systems 33},
year = {2020},
note = {To appear in},
}
```
