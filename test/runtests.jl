using Test
using LinearAlgebra
using TriangularPacked

@testset "Packing structure" begin
    A = rand(8,8)
    A = tril(A)  # ensure lower-triangular
    Lpack = pack_lower_levels(A, 2)

    @test isa(Lpack, TriangularPacked{Float64})
    @test Lpack.n == 8
end


@testset "TriangularPacked TRMM" begin
    n = 8
    A = tril(rand(n,n))              # lower-triangular
    X = rand(n,3)

    # build packed structure
    Lpack = TriangularPacked.Pack.pack_lower_levels(A, 2)

    # split X into leaf blocks matching leaves of L
    leaf_sizes = [size(D,1) for D in Lpack.leaves]
    idx = cumsum(vcat(0, leaf_sizes))
    Xdiag = [copy(X[idx[i]+1:idx[i+1], :]) for i in 1:length(leaf_sizes)]

    # run packed TRMM
    TriangularPacked.TRMM.trmm_lower_packed!(Lpack, Xdiag)

    # reconstruct X from leaf blocks
    Xpacked = vcat(Xdiag...)

    # baseline
    Xbaseline = A * X

    @test isapprox(Xpacked, Xbaseline; rtol=1e-10, atol=1e-12)
end

