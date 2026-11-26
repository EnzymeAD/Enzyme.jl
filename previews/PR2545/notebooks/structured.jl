### A Pluto.jl notebook ###
# v0.20.17

using Markdown
using InteractiveUtils

# ╔═╡ 5b1b8602-8598-11f0-39b3-237b6e3ec14f
begin
    import Pkg
    # careful: this is _not_ a reproducible environment
    # activate the local environment
    Pkg.activate(".")
    Pkg.instantiate()
    using PlutoUI, PlutoLinks
end

# ╔═╡ 94a8cc8f-d510-4a3e-bd7c-116da1a0a021
@revise using Enzyme

# ╔═╡ 363b1faa-5e50-476b-838e-5ea3f3c4ecf5
@revise using EnzymeCore

# ╔═╡ 199c2ada-f13b-4955-a2fa-c1876fb96514
using LinearAlgebra

# ╔═╡ e49d650b-af38-4d2c-91b6-9ac93613bbf0
import EnzymeCore: EnzymeRules

# ╔═╡ 4aa3740e-211c-4fff-9707-b8731a3fa57f
begin
	struct MySymmetric{T,S<:AbstractMatrix{<:T}} <: AbstractMatrix{T}
	    data::S
	    uplo::Char
	
	    function MySymmetric{T,S}(data, uplo::Char) where {T,S<:AbstractMatrix{<:T}}
	        LinearAlgebra.require_one_based_indexing(data)
	        (uplo != 'U' && uplo != 'L') && LinearAlgebra.throw_uplo()
	        new{T,S}(data, uplo)
	    end
	end
	function MySymmetric(A, uplo='U')
		 MySymmetric{eltype(A), typeof(A)}(A, 'U')
	end
end

# ╔═╡ d6572143-fc11-4fdc-9c23-34867b33ad85
@inline function Base.getindex(A::MySymmetric, i::Int, j::Int)
    @boundscheck Base.checkbounds(A, i, j)
    @inbounds if i == j
        return A.data[i, j]
    elseif (A.uplo == 'U') == (i < j)
        return A.data[i, j]
    else
        return A.data[j, i]
    end
end

# ╔═╡ 8f6588f2-754b-435c-9998-38da8f6b14ad
begin
	Base.size(A::MySymmetric) = size(A.data)
	Base.length(A::MySymmetric) = length(A.data)
end

# ╔═╡ de04ba90-a397-4b75-b6b4-977d5881e848
x = [1.0 0.0
	 0.0 1.0]

# ╔═╡ 9beada1d-6281-4dee-9049-8a66af1199a4
norm(x)

# ╔═╡ 9e95ec8a-132d-419e-a295-33383b4cac85
Enzyme.gradient(Reverse, norm, x) |> only

# ╔═╡ 4be9a16a-c14b-4378-aa02-fc4bfa783d10
Enzyme.gradient(Reverse, norm, MySymmetric(x))|> only

# ╔═╡ de7e14eb-4e55-4661-8614-e8026d09e6d3
x2 = [0.0 1.0
	  1.0 0.0]

# ╔═╡ 999fb61b-20c6-4dcf-ad34-eca257bfda9f
d_x2 = Enzyme.gradient(Reverse, norm, x2) |> only

# ╔═╡ c9dce3d8-8d2f-4b27-a4f2-e3b79c1a2e31
d_x2_sym = Enzyme.gradient(Reverse, norm, MySymmetric(x2)) |> only

# ╔═╡ ae83a229-1877-4c4c-937d-3f6781963c41
d_x2 == d_x2_sym

# ╔═╡ 67d08dd1-54dc-42b4-a0c3-e07aec1239fb
sum(d_x2) == sum(d_x2_sym.data)

# ╔═╡ 827485bf-0973-4650-a969-6225f72e5d6a
 Symmetric(x2) |> dump

# ╔═╡ dbe34880-93bf-4a5d-b28b-5e6b76267742
 d_x2_sym |> dump

# ╔═╡ 0122a4df-75d9-444e-8d83-d7a93b6dfeb5
begin
	struct MySymmetric2{T,S<:AbstractMatrix{<:T}} <: AbstractMatrix{T}
	    data::S
	    uplo::Char
	
	    function MySymmetric2{T,S}(data, uplo::Char) where {T,S<:AbstractMatrix{<:T}}
	        LinearAlgebra.require_one_based_indexing(data)
	        (uplo != 'U' && uplo != 'L') && LinearAlgebra.throw_uplo()
	        new{T,S}(data, uplo)
	    end
	end
	function MySymmetric2(A, uplo='U')
		 MySymmetric2{eltype(A), typeof(A)}(A, 'U')
	end
end

# ╔═╡ 8497709d-d123-48bc-a86e-5f58aa1b0ebc
@inline function Base.getindex(A::MySymmetric2, i::Int, j::Int)
    @boundscheck Base.checkbounds(A, i, j)
    @inbounds if i == j
        return A.data[i, j]
    elseif (A.uplo == 'U') == (i < j)
        return A.data[i, j]
    else
        return A.data[j, i]
    end
end

# ╔═╡ b648f8e9-ec43-4883-acd1-3f77f85f23c8
md"""
Let us implement a Symmetric matrix type like it exist in LinearAlgebra.jl
"""

# ╔═╡ 470f5618-109e-4328-9e8c-21d130b0170c
md"""
!!! warning
    The gradient of a Symmetric matrix seem to be right right! But let's check for the correctness of the off-diagonal entries, when they are non-zero!
"""

# ╔═╡ dbbc0d6c-dbab-4e8d-92cc-6c5517b7caac
md"""
So while the Symmetric matrix does actually store a full matrix underneath, it is more treated like a triangular matrix (upper in this case). So the entries `A[1,2]` and `A[2,1]` are aliased. The gradient accumulation is correct, but when we then use this matrix later on in Julia code it appears as if the gradient was twice what it ought to be.
"""

# ╔═╡ bd3ea593-88e2-4f6c-9df7-551950cdf020
md"""
!!! note
    We implement a second equivalent type here, since otherwise we could not show the issue clearly.
"""

# ╔═╡ d0a031e4-99a4-417b-8f57-58a67219fa23
begin
	Base.size(A::MySymmetric2) = size(A.data)
	Base.length(A::MySymmetric2) = length(A.data)
end

# ╔═╡ d047c162-446f-4de4-b2fd-5f2550f0ad78
md"""
Now we can implement a rule where we adjust the gradient contribution to be half since later we will double count them.
"""

# ╔═╡ e81ac66b-75ba-4e88-8e9c-491a60a671dc
begin
	function EnzymeRules.augmented_primal(config, func::Const{typeof(Base.getindex)}, ::Type{<:Active}, S::Duplicated{<:MySymmetric2}, i::Const, j::Const)
	    # Compute primal
	    if needs_primal(config)
	        primal = func.val(S.val, i.val, j.val)
	    else
	        primal = nothing
	    end
	
	    # Return an AugmentedReturn object with shadow = nothing
	    return EnzymeRules.AugmentedReturn(primal, nothing, nothing)
	end

	function EnzymeRules.reverse(config, ::Const{typeof(Base.getindex)}, dret::Active, tape,
	                 S::Duplicated{<:MySymmetric2}, i::Const, j::Const)
		i = i.val
		j = j.val
		A = S.val
		dA = S.dval
		@inbounds if i == j
        	dA.data[i, j] += dret.val
    	elseif (A.uplo == 'U') == (i < j)
        	dA.data[i, j] += dret.val / 2
    	else
	        dA.data[j, i] += dret.val / 2
    	end
		
	    return (nothing, nothing, nothing)
	end
end

# ╔═╡ fe7138ce-950a-4f79-b176-cdb227d4c898
Enzyme.gradient(Reverse, norm, MySymmetric(x2)) |> only

# ╔═╡ 18ee361f-a1eb-4e3c-aa7e-9bd344e96eba
Enzyme.gradient(Reverse, norm, MySymmetric2(x2)) |> only

# ╔═╡ Cell order:
# ╠═5b1b8602-8598-11f0-39b3-237b6e3ec14f
# ╠═94a8cc8f-d510-4a3e-bd7c-116da1a0a021
# ╠═363b1faa-5e50-476b-838e-5ea3f3c4ecf5
# ╠═e49d650b-af38-4d2c-91b6-9ac93613bbf0
# ╠═199c2ada-f13b-4955-a2fa-c1876fb96514
# ╟─b648f8e9-ec43-4883-acd1-3f77f85f23c8
# ╠═4aa3740e-211c-4fff-9707-b8731a3fa57f
# ╠═d6572143-fc11-4fdc-9c23-34867b33ad85
# ╠═8f6588f2-754b-435c-9998-38da8f6b14ad
# ╠═de04ba90-a397-4b75-b6b4-977d5881e848
# ╠═9beada1d-6281-4dee-9049-8a66af1199a4
# ╠═9e95ec8a-132d-419e-a295-33383b4cac85
# ╠═4be9a16a-c14b-4378-aa02-fc4bfa783d10
# ╟─470f5618-109e-4328-9e8c-21d130b0170c
# ╠═de7e14eb-4e55-4661-8614-e8026d09e6d3
# ╠═999fb61b-20c6-4dcf-ad34-eca257bfda9f
# ╠═c9dce3d8-8d2f-4b27-a4f2-e3b79c1a2e31
# ╠═ae83a229-1877-4c4c-937d-3f6781963c41
# ╠═67d08dd1-54dc-42b4-a0c3-e07aec1239fb
# ╠═827485bf-0973-4650-a969-6225f72e5d6a
# ╠═dbe34880-93bf-4a5d-b28b-5e6b76267742
# ╟─dbbc0d6c-dbab-4e8d-92cc-6c5517b7caac
# ╟─bd3ea593-88e2-4f6c-9df7-551950cdf020
# ╠═0122a4df-75d9-444e-8d83-d7a93b6dfeb5
# ╠═8497709d-d123-48bc-a86e-5f58aa1b0ebc
# ╠═d0a031e4-99a4-417b-8f57-58a67219fa23
# ╟─d047c162-446f-4de4-b2fd-5f2550f0ad78
# ╠═e81ac66b-75ba-4e88-8e9c-491a60a671dc
# ╠═fe7138ce-950a-4f79-b176-cdb227d4c898
# ╠═18ee361f-a1eb-4e3c-aa7e-9bd344e96eba
