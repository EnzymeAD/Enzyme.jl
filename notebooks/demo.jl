### A Pluto.jl notebook ###
# v0.15.1

using Markdown
using InteractiveUtils

# ╔═╡ f6b815e8-e813-11eb-0760-9165c6cf726f
begin
	using Enzyme
	using PlutoUI
	using ChainRules
	using BenchmarkTools
	import ForwardDiff
	import Zygote
end

# ╔═╡ 6f099fd4-10c4-43fb-9f7a-cdb8e0378623
md"""
# Activity annotations
- `Const`
- `Active`
- `Duplicated`
- `DuplicatedNoNeed`
"""

# ╔═╡ b3a4c95d-51cc-4db6-b2f7-040cc4135100
square(x) = x^2

# ╔═╡ 8f207356-c29a-4762-a7ce-011b8919f667
autodiff(square, 1.0)

# ╔═╡ 60d880ac-b534-41b6-a868-3aead8cdf827
md"""
Default activity for values is `Const`
"""

# ╔═╡ b8b856f1-9bd6-40fc-a141-280b0f8a34ff
autodiff(square, Const(1.0))

# ╔═╡ cf180dad-90a5-4380-9b74-7725dcaf0e73
autodiff(square, Active(1.0))

# ╔═╡ 6bf01d28-9c5e-4220-bc42-bf446ee7d807
md"""
## Supporting mutating functions

Enzyme can differentiate through mutating functions. This requires that the users passes in the shadow variables with the `Duplicated` or `DuplicatedNoNeed` activity annotation.
"""

# ╔═╡ 7dfe8143-bd09-4046-8141-bd51e12a3e5d
function cube(y, x)
	y[] = x[]^3
	return nothing
end

# ╔═╡ 124e04fe-9393-44de-a556-52fd5d1cf9db
let
	x = Ref(4.0)
	y = Ref(0.0)
	cube(y, x)
	y[]
end

# ╔═╡ bda1fecc-2499-4deb-bc4d-988e29887929
md"""
In order to calculate the gradient of `x`, we have to propagate `1.0` into the
shadow `dy`.
"""

# ╔═╡ fb55cc7a-dc83-4d42-87d3-c1ab3450ca28
let
	x = Ref(4.0)
	dx = Ref(0.0)
	
	y = Ref(0.0)
	dy = Ref(1.0)
	
	autodiff(cube, Duplicated(y, dy), Duplicated(x, dx))
	y[], dy[], x[], dx[]
end

# ╔═╡ 59c51921-a7ce-4014-827f-a460ea0e2e7e
md"""
## `DuplicatedNoNeed`

If we do not care about the return value of our function and are only interested in calculating the gradient we can use `DuplicatedNoNeed`.
"""

# ╔═╡ 258c181e-52f3-4a8f-b2ca-6713c7137766
let
	x = Ref(4.0)
	dx = Ref(0.0)
	
	y = Ref(0.0)
	dy = Ref(1.0)
	
	autodiff(cube, Enzyme.DuplicatedNoNeed(y, dy), Duplicated(x, dx))
	y[], dy[], x[], dx[]
end

# ╔═╡ 6db932aa-1bed-461c-9029-cfd77e783f7a
with_terminal() do
  Enzyme.Compiler.enzyme_code_llvm( cube,
	Tuple{Enzyme.Duplicated{Base.RefValue{Float64}}, 
	Duplicated{Base.RefValue{Float64}}}, debuginfo=:none)
end

# ╔═╡ 39279922-2994-4d2a-83c0-fead3083f247
with_terminal() do
  Enzyme.Compiler.enzyme_code_llvm( cube,
	Tuple{Enzyme.DuplicatedNoNeed{Base.RefValue{Float64}}, 
	Duplicated{Base.RefValue{Float64}}}, debuginfo=:none)
end

# ╔═╡ 5e85ad6a-6508-4bf0-b25b-a64ba65556f5
md"""
# Differentiating through control-flow

Let's differentiate through some control flow. This kind of scalar code is where normally one would use `ForwardDiff.jl` since the machine learning optimized toolkits like Zygote have unacceptable overheads.
"""

# ╔═╡ 6826dc6e-2e9c-4dcc-89c4-cc98ca03cab0
# Taylor series for `-log(1-x)`
# eval at -log(1-1/2) = -log(1/2)
function taylor(f::T, N=10^7) where T
    g = zero(T)
    for i in 1:N
        g += f^i / i
    end
    return g
end

# ╔═╡ 58ca6247-082e-4f66-947b-29181c3ebedf
autodiff(taylor, Active(0.5), Const(10^8))

# ╔═╡ ef02d55d-bc5d-4802-b200-1b0d0a037ec6
fwd_taylor(x) = ForwardDiff.derivative(taylor, 0.5)

# ╔═╡ cf51c8b8-733d-436e-a881-116653126906
zyg_taylor(x) = Zygote.gradient(taylor, x)

# ╔═╡ c0bc6781-00b5-4803-974c-7760dfa140c2
enz_taylor(x) = autodiff(taylor, Active(x))

# ╔═╡ 49f3d740-906a-4987-b79e-a7af9e1b54e5
@benchmark fwd_taylor($(Ref(0.5))[])

# ╔═╡ d996f5e0-0755-4d86-b747-5f21944183bd
@benchmark zyg_taylor($(Ref(0.5))[])

# ╔═╡ 078aa223-1e77-4084-90e2-cd87d71b5988
@benchmark enz_taylor($(Ref(0.5))[])

# ╔═╡ e21a7977-73b2-4c6a-90b7-3c9dbc04e613
md"""
# Differentiating through more complicated codes

## A custom matrix multiply
"""

# ╔═╡ 21858576-c2a0-474f-9aa7-d14ee7f7210c
function mymul!(R, A, B)
    @assert axes(A,2) == axes(B,1)
    @inbounds @simd for i in eachindex(R)
        R[i] = 0
    end
    @inbounds for j in axes(B, 2), i in axes(A, 1)
        @inbounds @simd for k in axes(A,2)
            R[i,j] += A[i,k] * B[k,j]
        end
    end
    nothing
end

# ╔═╡ 29ee6b7a-971e-4d46-ac3b-d793124a4542
begin
	A = rand(1024, 64)
	B = rand(64, 512)

	R = zeros(size(A,1), size(B,2))
	∂z_∂R = rand(size(R)...)  # Some gradient/tangent passed to us

	∂z_∂A = zero(A)
	∂z_∂B = zero(B)
end;

# ╔═╡ 113ca417-ede9-4f15-85ff-7c6fb7d97494
Enzyme.autodiff(mymul!, 
	Duplicated(R, ∂z_∂R),
	Duplicated(A, ∂z_∂A),
	Duplicated(B, ∂z_∂B))

# ╔═╡ 3f3771a0-3d23-4ba5-b5f0-cb49ec40260c
md"""
Let's confirm correctness of result
"""

# ╔═╡ 62182140-d067-4814-a83d-f6c9d6b3a15f
R ≈ A * B

# ╔═╡ bfa753c2-1f61-4c38-b4d3-55f068dd0948
md"""
and correctness of the gradients
"""

# ╔═╡ f6856069-8bfa-49ed-8f5e-f1bd6009c4f4
∂z_∂A ≈ ∂z_∂R * B'

# ╔═╡ e59d8304-603f-4893-8e73-ed23dad2d28f
∂z_∂B ≈ A' * ∂z_∂R

# ╔═╡ c4ded44a-3193-402b-9f10-0e3ec60f0c65
@benchmark Enzyme.autodiff(mymul!, 
	Duplicated(R, ∂z_∂R),
	Duplicated(A, ∂z_∂A),
	Duplicated(B, ∂z_∂B))

# ╔═╡ febf0550-8c6d-4a26-8bc1-3af206294d41
md"""
# Some more fun
"""

# ╔═╡ aa7d84ae-6504-41fe-915d-98a10137d64a
struct LList
    next::Union{LList,Nothing}
	val::Float64
end 

# ╔═╡ ac02df55-6c63-4f5d-b643-ff89930152ee
function sumlist(n::LList)
    sum = 0.0
    while n !== nothing
        sum += n.val
        n = n.next
    end
    sum
end

# ╔═╡ edfe9131-4316-4af3-b6eb-4c0bc89fe0ce
begin    
    regular = LList(LList(nothing, 1.0), 2.0)
    shadow  = LList(LList(nothing, 0.0), 0.0)
    autodiff(sumlist, Duplicated(regular, shadow))
end

# ╔═╡ 398b0cde-ff8e-4238-be65-33699e17abb6
shadow.val ≈ 1.0

# ╔═╡ 686b7891-0ff5-4b7f-ba54-92aff8bacb12
shadow.next.val ≈ 1.0

# ╔═╡ 4335c393-1765-4641-ae4b-52cafac4238d
html"<button onclick='present()'>present</button>"

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
BenchmarkTools = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
ChainRules = "082447d4-558c-5d27-93f4-14fc19e9eca2"
Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"
ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

[compat]
BenchmarkTools = "~1.1.1"
ChainRules = "~0.8.22"
Enzyme = "~0.6.4"
ForwardDiff = "~0.10.18"
PlutoUI = "~0.7.9"
Zygote = "~0.6.16"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

[[AbstractFFTs]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "485ee0867925449198280d4af84bdb46a2a404d0"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.0.1"

[[Adapt]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "84918055d15b3114ede17ac6a7182f68870c16f7"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.3.1"

[[ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"

[[Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[BenchmarkTools]]
deps = ["JSON", "Logging", "Printf", "Statistics", "UUIDs"]
git-tree-sha1 = "c31ebabde28d102b602bada60ce8922c266d205b"
uuid = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
version = "1.1.1"

[[CEnum]]
git-tree-sha1 = "215a9aa4a1f23fbd05b92769fdd62559488d70e9"
uuid = "fa961155-64e5-5f13-b03f-caf6b980ea82"
version = "0.4.1"

[[ChainRules]]
deps = ["ChainRulesCore", "Compat", "LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "dabb81719f820cddd6df4916194d44f1fe282bd1"
uuid = "082447d4-558c-5d27-93f4-14fc19e9eca2"
version = "0.8.22"

[[ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "0b0aa9d61456940511416b59a0e902c57b154956"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "0.10.12"

[[CommonSubexpressions]]
deps = ["MacroTools", "Test"]
git-tree-sha1 = "7b8a93dba8af7e3b42fecabf646260105ac373f7"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.0"

[[Compat]]
deps = ["Base64", "Dates", "DelimitedFiles", "Distributed", "InteractiveUtils", "LibGit2", "Libdl", "LinearAlgebra", "Markdown", "Mmap", "Pkg", "Printf", "REPL", "Random", "SHA", "Serialization", "SharedArrays", "Sockets", "SparseArrays", "Statistics", "Test", "UUIDs", "Unicode"]
git-tree-sha1 = "dc7dedc2c2aa9faf59a55c622760a25cbefbe941"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "3.31.0"

[[CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"

[[DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "4437b64df1e0adccc3e5d1adbc3ac741095e4677"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.9"

[[Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[DelimitedFiles]]
deps = ["Mmap"]
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"

[[DiffResults]]
deps = ["StaticArrays"]
git-tree-sha1 = "c18e98cba888c6c25d1c3b048e4b3380ca956805"
uuid = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
version = "1.0.3"

[[DiffRules]]
deps = ["NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "214c3fcac57755cfda163d91c58893a8723f93e9"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.0.2"

[[Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "a32185f5428d3986f47c2ab78b1f216d5e6cc96f"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.8.5"

[[Downloads]]
deps = ["ArgTools", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"

[[Enzyme]]
deps = ["Adapt", "CEnum", "Enzyme_jll", "GPUCompiler", "LLVM", "Libdl", "ObjectFile"]
git-tree-sha1 = "29a884b30338585a31ca43d3e839f4f8cb8d7498"
uuid = "7da242da-08ed-463a-9acd-ee780be4f1d9"
version = "0.6.4"

[[Enzyme_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "8d60073e8bbcbc5749911fa1583ef573332e32c7"
uuid = "7cc45869-7501-5eee-bdea-0790c847d4ef"
version = "0.0.15+0"

[[ExprTools]]
git-tree-sha1 = "b7e3d17636b348f005f11040025ae8c6f645fe92"
uuid = "e2ba6199-217a-4e67-a87a-7c52f15ade04"
version = "0.1.6"

[[FillArrays]]
deps = ["LinearAlgebra", "Random", "SparseArrays"]
git-tree-sha1 = "25b9cc23ba3303de0ad2eac03f840de9104c9253"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "0.12.0"

[[ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "NaNMath", "Printf", "Random", "SpecialFunctions", "StaticArrays"]
git-tree-sha1 = "e2af66012e08966366a43251e1fd421522908be6"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "0.10.18"

[[GPUCompiler]]
deps = ["DataStructures", "ExprTools", "InteractiveUtils", "LLVM", "Libdl", "Logging", "TimerOutputs", "UUIDs"]
git-tree-sha1 = "e8a09182a4440489e2e3dedff5ad3f6bbe555396"
uuid = "61eb1bfa-7361-4325-ad38-22787b887f55"
version = "0.12.5"

[[IRTools]]
deps = ["InteractiveUtils", "MacroTools", "Test"]
git-tree-sha1 = "95215cd0076a150ef46ff7928892bc341864c73c"
uuid = "7869d1d1-7146-5819-86e3-90919afe41df"
version = "0.4.3"

[[InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "642a199af8b68253517b80bd3bfd17eb4e84df6e"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.3.0"

[[JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "81690084b6198a2e1da36fcfda16eeca9f9f24e4"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.1"

[[LLVM]]
deps = ["CEnum", "LLVMExtra_jll", "Libdl", "Printf", "Unicode"]
git-tree-sha1 = "1b7ba36ea7aa6fa2278118951bad114fbb8359f2"
uuid = "929cbde3-209d-540e-8aea-75f648917ca0"
version = "4.1.0"

[[LLVMExtra_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b36c0677a0549c7d1dc8719899a4133abbfacf7d"
uuid = "dad2f222-ce93-54a1-a47d-0025e8a3acab"
version = "0.0.6+0"

[[LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"

[[LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"

[[LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"

[[Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[LinearAlgebra]]
deps = ["Libdl"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[LogExpFunctions]]
deps = ["DocStringExtensions", "LinearAlgebra"]
git-tree-sha1 = "7bd5f6565d80b6bf753738d2bc40a5dfea072070"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.2.5"

[[Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "6a8a2a625ab0dea913aba95c11370589e0239ff0"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.6"

[[Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"

[[Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"

[[NaNMath]]
git-tree-sha1 = "bfe47e760d60b82b66b61d2d44128b62e3a369fb"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "0.3.5"

[[NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"

[[ObjectFile]]
deps = ["Reexport", "StructIO"]
git-tree-sha1 = "55ce61d43409b1fb0279d1781bf3b0f22c83ab3b"
uuid = "d8793406-e978-5875-9003-1fc021f44a92"
version = "0.3.7"

[[OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[OrderedCollections]]
git-tree-sha1 = "85f8e6578bf1f9ee0d11e7bb1b1456435479d47c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.4.1"

[[Parsers]]
deps = ["Dates"]
git-tree-sha1 = "c8abc88faa3f7a3950832ac5d6e690881590d6dc"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "1.1.0"

[[Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"

[[PlutoUI]]
deps = ["Base64", "Dates", "InteractiveUtils", "JSON", "Logging", "Markdown", "Random", "Reexport", "Suppressor"]
git-tree-sha1 = "44e225d5837e2a2345e69a1d1e01ac2443ff9fcb"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.9"

[[Preferences]]
deps = ["TOML"]
git-tree-sha1 = "00cfd92944ca9c760982747e9a1d0d5d86ab1e5a"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.2.2"

[[Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[Random]]
deps = ["Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[Reexport]]
git-tree-sha1 = "5f6c21241f0f655da3952fd60aa18477cf96c220"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.1.0"

[[Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "4036a3bd08ac7e968e27c203d45f5fff15020621"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.1.3"

[[SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"

[[Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[SpecialFunctions]]
deps = ["ChainRulesCore", "LogExpFunctions", "OpenSpecFun_jll"]
git-tree-sha1 = "a50550fa3164a8c46747e62063b4d774ac1bcf49"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "1.5.1"

[[StaticArrays]]
deps = ["LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "1b9a0f17ee0adde9e538227de093467348992397"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.2.7"

[[Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[StructIO]]
deps = ["Test"]
git-tree-sha1 = "010dc73c7146869c042b49adcdb6bf528c12e859"
uuid = "53d494c1-5632-5724-8f4c-31dff12d585f"
version = "0.3.0"

[[Suppressor]]
git-tree-sha1 = "a819d77f31f83e5792a76081eee1ea6342ab8787"
uuid = "fd094767-a336-5f1f-9728-57cf17d0bbfb"
version = "0.2.0"

[[TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"

[[Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"

[[Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[TimerOutputs]]
deps = ["ExprTools", "Printf"]
git-tree-sha1 = "209a8326c4f955e2442c07b56029e88bb48299c7"
uuid = "a759f4b9-e2f1-59dc-863e-4aeb61b1ea8f"
version = "0.5.12"

[[UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"

[[Zygote]]
deps = ["AbstractFFTs", "ChainRules", "ChainRulesCore", "DiffRules", "Distributed", "FillArrays", "ForwardDiff", "IRTools", "InteractiveUtils", "LinearAlgebra", "MacroTools", "NaNMath", "Random", "Requires", "SpecialFunctions", "Statistics", "ZygoteRules"]
git-tree-sha1 = "4f9a5ba559da1fc7474f2ece6c6c1e21c4ab989c"
uuid = "e88e6eb3-aa80-5325-afca-941959d7151f"
version = "0.6.16"

[[ZygoteRules]]
deps = ["MacroTools"]
git-tree-sha1 = "9e7a1e8ca60b742e508a315c17eef5211e7fbfd7"
uuid = "700de1a5-db45-46bc-99cf-38207098b444"
version = "0.2.1"

[[nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"

[[p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
"""

# ╔═╡ Cell order:
# ╠═f6b815e8-e813-11eb-0760-9165c6cf726f
# ╟─6f099fd4-10c4-43fb-9f7a-cdb8e0378623
# ╠═b3a4c95d-51cc-4db6-b2f7-040cc4135100
# ╠═8f207356-c29a-4762-a7ce-011b8919f667
# ╟─60d880ac-b534-41b6-a868-3aead8cdf827
# ╠═b8b856f1-9bd6-40fc-a141-280b0f8a34ff
# ╠═cf180dad-90a5-4380-9b74-7725dcaf0e73
# ╟─6bf01d28-9c5e-4220-bc42-bf446ee7d807
# ╠═7dfe8143-bd09-4046-8141-bd51e12a3e5d
# ╠═124e04fe-9393-44de-a556-52fd5d1cf9db
# ╟─bda1fecc-2499-4deb-bc4d-988e29887929
# ╠═fb55cc7a-dc83-4d42-87d3-c1ab3450ca28
# ╟─59c51921-a7ce-4014-827f-a460ea0e2e7e
# ╠═258c181e-52f3-4a8f-b2ca-6713c7137766
# ╠═6db932aa-1bed-461c-9029-cfd77e783f7a
# ╠═39279922-2994-4d2a-83c0-fead3083f247
# ╠═5e85ad6a-6508-4bf0-b25b-a64ba65556f5
# ╠═6826dc6e-2e9c-4dcc-89c4-cc98ca03cab0
# ╠═58ca6247-082e-4f66-947b-29181c3ebedf
# ╠═ef02d55d-bc5d-4802-b200-1b0d0a037ec6
# ╠═cf51c8b8-733d-436e-a881-116653126906
# ╠═c0bc6781-00b5-4803-974c-7760dfa140c2
# ╠═49f3d740-906a-4987-b79e-a7af9e1b54e5
# ╠═d996f5e0-0755-4d86-b747-5f21944183bd
# ╠═078aa223-1e77-4084-90e2-cd87d71b5988
# ╠═e21a7977-73b2-4c6a-90b7-3c9dbc04e613
# ╠═21858576-c2a0-474f-9aa7-d14ee7f7210c
# ╠═29ee6b7a-971e-4d46-ac3b-d793124a4542
# ╠═113ca417-ede9-4f15-85ff-7c6fb7d97494
# ╟─3f3771a0-3d23-4ba5-b5f0-cb49ec40260c
# ╠═62182140-d067-4814-a83d-f6c9d6b3a15f
# ╠═bfa753c2-1f61-4c38-b4d3-55f068dd0948
# ╠═f6856069-8bfa-49ed-8f5e-f1bd6009c4f4
# ╠═e59d8304-603f-4893-8e73-ed23dad2d28f
# ╠═c4ded44a-3193-402b-9f10-0e3ec60f0c65
# ╟─febf0550-8c6d-4a26-8bc1-3af206294d41
# ╠═aa7d84ae-6504-41fe-915d-98a10137d64a
# ╠═ac02df55-6c63-4f5d-b643-ff89930152ee
# ╠═edfe9131-4316-4af3-b6eb-4c0bc89fe0ce
# ╠═398b0cde-ff8e-4238-be65-33699e17abb6
# ╠═686b7891-0ff5-4b7f-ba54-92aff8bacb12
# ╟─4335c393-1765-4641-ae4b-52cafac4238d
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
