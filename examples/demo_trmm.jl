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
