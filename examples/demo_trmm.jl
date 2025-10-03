using LinearAlgebra, TriangularPacked

# Build random lower-triangular
A = rand(16,16)
A = tril(A)

Lpack = pack_lower_levels(A, 2)

X = rand(16,4)  # RHS
# Split into leaf blocks
Xdiag = [copy(X[1:4,:]), copy(X[5:8,:]), copy(X[9:12,:]), copy(X[13:16,:])]

trmm_lower_packed!(Lpack, Xdiag)
println("Result leaf blocks after TRMM:")
println(Xdiag)



n = 8
A = tril(rand(n,n))
X = rand(n,2)

Lpack = TriangularPacked.Pack.pack_lower_levels(A, 2)

# slice RHS into leaf-aligned blocks
leaf_sizes = [size(D,1) for D in Lpack.leaves]
idx = cumsum(vcat(0, leaf_sizes))
Xdiag = [copy(X[idx[i]+1:idx[i+1], :]) for i in 1:length(leaf_sizes)]

# run packed TRMM
TriangularPacked.TRMM.trmm_lower_packed!(Lpack, Xdiag)

println("Result (packed TRMM):")
println(vcat(Xdiag...))
println("Baseline A*X:")
println(A*X)
