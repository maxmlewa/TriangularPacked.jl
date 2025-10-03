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
