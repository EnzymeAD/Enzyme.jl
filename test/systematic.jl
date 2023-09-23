using EnzymeTestUtils

using LinearAlgebra

modules = [LinearAlgebra]

function test_forward_all_activities(f, x)
    @testset "Forward mode: $Tret, $Tx" for Tret in (Const, Duplicated, DuplicatedNoNeed),
        Tx in (Const, Duplicated)
        @info "Testing forward mode for $f, $(typeof(x)), $Tret, $Tx"
        test_forward(f, Tret, (x, Tx))
    end
end

function test_forward_all_activities(f, x, y)
    @testset "Forward mode: $Tret, $Tx, $Ty" for Tret in (Const, Duplicated, DuplicatedNoNeed),
        Tx in (Const, Duplicated),
        Ty in (Const, Duplicated)
        @info "Testing forward mode for $f, $(typeof(x)), $(typeof(y)), $Tret, $Tx, $Ty"
        test_forward(f, Tret, (x, Tx), (y, Ty))
    end
end

function test_reverse_all_activities(f, x)
    @testset "Reverse mode: $Tret, $Tx" for Tret in (Const, Active),
        Tx in activities_rev(x)
        @info "Testing reverse mode for $f, $(typeof(x)), $Tret, $Tx"
        test_reverse(f, Tret, (x, Tx))
    end
end

function test_reverse_all_activities(f, x, y)
    @testset "Reverse mode: $Tret, $Tx, $Ty" for Tret in (Const, Active),
        Tx in activities_rev(x),
        Ty in activities_rev(y)
        @info "Testing reverse mode for $f, $(typeof(x)), $(typeof(y)), $Tret, $Tx, $Ty"
        test_reverse(f, Tret, (x, Tx), (y, Ty))
    end
end

activities_rev(::Real) = (Const, Active)
activities_rev(::AbstractArray{<:Real}) = (Const, BatchDuplicated)

isa_valid_return(::Bool) = false
isa_valid_return(::Real) = true
isa_valid_return(::AbstractArray{Bool}) = false
isa_valid_return(::AbstractArray{<:Real}) = true

@testset "$_module" for _module in modules
    @info "Systematic tests for module $_module"
    function_names = filter(x -> eval(x) isa Function, names(_module))
    possible_args = (rand(), rand(2), rand(2, 2))
    @testset "$name" for name in function_names
        F = eval(name)
        @testset "One argument" begin
            @testset "$(typeof(arg))" for arg in possible_args
                return_value = try
                    F(arg)
                catch e
                    continue
                end
                if isa_valid_return(return_value)
                    test_forward_all_activities(F, arg)
                    test_reverse_all_activities(F, arg)
                else
                    continue
                end
            end
        end
        @testset "$(typeof(arg1)), $(typeof(arg2))" for (arg1, arg2) in Iterators.product(
            possible_args, possible_args
        )
            return_value = try
                F(arg1, arg2)
            catch e
                continue
            end
            if isa_valid_return(return_value)
                test_forward_all_activities(F, arg1, arg2)
                test_reverse_all_activities(F, arg1, arg2)
            else
                continue
            end
        end
    end
end
