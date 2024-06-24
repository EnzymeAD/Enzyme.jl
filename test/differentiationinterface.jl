using DifferentiationInterface
using DifferentiationInterfaceTest
using Enzyme: Enzyme

test_differentiation([AutoEnzyme(; mode=Enzyme.Forward), AutoEnzyme(; mode=Enzyme.Reverse)];
                     second_order=false, logging=false)
