# Internal API

!!! note

    This is the documentation of Enzymes's internal API. The internal API is
    *not* subject to semantic versioning and may change at any time and
    without deprecation.

```@autodocs
Modules = [Enzyme.Compiler, Enzyme.Compiler.RecursiveMaps]
Order = [:module, :type, :constant, :macro, :function]
Filter = t -> !(t === Enzyme.Compiler.CheckNan)
```
